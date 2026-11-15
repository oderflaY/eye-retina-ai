"""Rasgos de asimetría facial con MediaPipe (para detectar parálisis facial).

Corre en ``.venv-etiquetas`` (MediaPipe choca con TensorFlow)::

    .venv-etiquetas/bin/python src/data/rasgos_rostro.py \
        --palsy ~/TemrariosSpace/datasets/Rostro/palsy/frames --ffhq data/raw/ffhq_listos --max_ffhq 1000

Por qué rasgos y no la foto directa: en PalsyNet los videos con parálisis son caseros y los
sanos son de youtubers; una CNN aprendería "video casero vs. estudio". La asimetría (ojo que no
cierra, comisura que no sube, ceja caída) es lo que mira el médico y no depende de la cámara.

Rasgos por cara (todos adimensionales, normalizados por la distancia entre ojos y con la cara
enderezada por el ángulo de la línea de los ojos):
* ``bs_<par>``: |izq − der| de los blendshapes de MediaPipe en pares (parpadeo, sonrisa, ceja…).
* ``geo_*``: diferencias de altura de comisuras, cejas y apertura de ojos; desvío de la boca y
  de la nariz respecto a la línea media.
* ``yaw``/``pitch``/``roll``: pose (para filtrar caras muy de lado).
"""

from __future__ import annotations

import argparse
import csv
import tarfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

PARES = [
    ("eyeBlink", "Left", "Right"), ("eyeSquint", "Left", "Right"), ("eyeWide", "Left", "Right"),
    ("browDown", "Left", "Right"), ("browOuterUp", "Left", "Right"), ("cheekSquint", "Left", "Right"),
    ("mouthSmile", "Left", "Right"), ("mouthFrown", "Left", "Right"), ("mouthDimple", "Left", "Right"),
    ("mouthStretch", "Left", "Right"), ("mouthPress", "Left", "Right"), ("mouthUpperUp", "Left", "Right"),
    ("mouthLowerDown", "Left", "Right"), ("noseSneer", "Left", "Right"),
]
# Malla de MediaPipe ("derecho" = de la persona)
OJO_D, OJO_I = (33, 133, 159, 145), (263, 362, 386, 374)  # ext, int, párpado sup, inf
COMISURA_D, COMISURA_I = 61, 291
CEJA_D, CEJA_I = 105, 334
NARIZ, BARBILLA, FRENTE, LABIO_SUP, LABIO_INF = 1, 152, 10, 13, 14

COLUMNAS = (["fuente", "grupo", "clase", "archivo"] + [f"bs_{p}" for p, _, _ in PARES]
            + [f"bsm_{p}" for p, _, _ in PARES]
            + ["geo_comisura", "geo_ceja", "geo_ojo", "geo_boca_desvio", "geo_nariz_desvio", "geo_labio_inclin",
               "yaw", "pitch", "roll"])


def rasgos(res) -> dict | None:
    if not res.face_landmarks:
        return None
    pts = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]])[:, :2]
    bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
    f: dict = {}
    for p, l, r in PARES:
        a, b = bs.get(p + l, 0.0), bs.get(p + r, 0.0)
        f[f"bs_{p}"] = abs(a - b)
        f[f"bsm_{p}"] = (a + b) / 2  # cuánto se mueve en total (contexto: sin movimiento no hay asimetría)
    # enderezar por la línea de los ojos y escalar por la distancia entre ojos
    cd, ci = pts[list(OJO_D[:2])].mean(0), pts[list(OJO_I[:2])].mean(0)
    v = ci - cd
    ang = np.arctan2(v[1], v[0])
    rot = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    q = (pts - (cd + ci) / 2) @ rot.T / (np.linalg.norm(v) + 1e-6)
    ap_d = abs(q[OJO_D[3], 1] - q[OJO_D[2], 1]) / (abs(q[OJO_D[0], 0] - q[OJO_D[1], 0]) + 1e-6)
    ap_i = abs(q[OJO_I[3], 1] - q[OJO_I[2], 1]) / (abs(q[OJO_I[0], 0] - q[OJO_I[1], 0]) + 1e-6)
    medio_x = (q[NARIZ, 0] + q[FRENTE, 0] + q[BARBILLA, 0]) / 3
    boca = (q[COMISURA_D] + q[COMISURA_I]) / 2
    f.update(
        geo_comisura=abs(q[COMISURA_D, 1] - q[COMISURA_I, 1]),
        geo_ceja=abs((q[CEJA_D, 1] - q[OJO_D[2], 1]) - (q[CEJA_I, 1] - q[OJO_I[2], 1])),
        geo_ojo=abs(ap_d - ap_i) / (max(ap_d, ap_i) + 1e-6),
        geo_boca_desvio=abs(boca[0] - medio_x),
        geo_nariz_desvio=abs(q[NARIZ, 0] - (q[FRENTE, 0] + q[BARBILLA, 0]) / 2),
        geo_labio_inclin=abs(np.degrees(np.arctan2(q[COMISURA_I, 1] - q[COMISURA_D, 1], q[COMISURA_I, 0] - q[COMISURA_D, 0]))),
    )
    m = np.array(res.facial_transformation_matrixes[0])[:3, :3]
    f["yaw"] = float(np.degrees(np.arctan2(-m[2, 0], np.hypot(m[2, 1], m[2, 2]))))
    f["pitch"] = float(np.degrees(np.arctan2(m[2, 1], m[2, 2])))
    f["roll"] = float(np.degrees(ang))
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--palsy", type=Path, required=True)
    ap.add_argument("--ffhq", type=Path)
    ap.add_argument("--max_ffhq", type=int, default=1000)
    ap.add_argument("--modelo", default="models/face_landmarker.task")
    ap.add_argument("--out", type=Path, default=Path("data/processed/rostro/rasgos.csv"))
    a = ap.parse_args()
    det = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=a.modelo), running_mode=RunningMode.IMAGE, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = sin = 0
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNAS)
        w.writeheader()

        def procesar(img_bgr, **meta):
            nonlocal n, sin
            r = rasgos(det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))))
            if r is None:
                sin += 1
                return
            w.writerow({**meta, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}})
            n += 1

        for f in sorted(a.palsy.glob("*/*/*.jpg")):  # solo cuadros originales (no depth/ ni cropped_img/)
            clase, persona = f.parts[-3], f.parts[-2]
            procesar(cv2.imread(str(f)), fuente="palsynet", grupo=f"{clase}_{persona}",
                     clase=int(clase == "affected"), archivo=str(f))
        hechos = 0
        if a.ffhq:
            for tar in sorted(a.ffhq.glob("*.tar")):
                with tarfile.open(tar) as t:
                    for m in t:
                        if hechos >= a.max_ffhq:
                            break
                        if not m.isfile() or not m.name.lower().endswith((".png", ".jpg", ".webp")):
                            continue
                        img = cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                        procesar(img, fuente="ffhq", grupo=f"ffhq_{Path(m.name).stem}", clase=0, archivo=m.name)
                        hechos += 1
    print(f"LISTO: {n} caras con rasgos, {sin} sin cara")


if __name__ == "__main__":
    main()
