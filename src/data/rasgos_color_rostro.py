"""Color del blanco del ojo (ictericia) y de la piel de la cara (enrojecimiento) con MediaPipe.

Corre en ``.venv-etiquetas``::

    .venv-etiquetas/bin/python src/data/rasgos_color_rostro.py \
        --ictericia ~/TemrariosSpace/datasets/Rostro/ictericia --ffhq data/raw/ffhq_listos --max_ffhq 1500

Por qué medidas de color y no una CNN: en los datasets públicos las fotos enfermas vienen de
bancos clínicos (marca de agua, encuadre) y las sanas de bancos de stock; una CNN aprendería la
fuente. El signo clínico es simple y medible:

* **Ictericia**: la esclerótica (blanco del ojo) se pone amarilla → b* alto en CIELAB. Se toma la
  abertura del ojo (contorno de la malla), se quita el iris (círculo de los puntos 468-477) y se
  quedan los píxeles claros. Se mide también la piel (mejillas) para separar "luz amarilla" (todo
  amarillo) de "ojo amarillo" (el ojo más amarillo que lo esperado para esa luz).
* **Enrojecimiento facial** (rosácea, rubor): a* de mejillas y nariz menos a* de la frente.

``esp`` = esclerótica, ``piel`` = mejillas, ``frente``; ``gw_`` = tras balance de grises.
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

CONTORNO = {
    "d": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "i": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
}
IRIS = {"d": (468, [469, 470, 471, 472]), "i": (473, [474, 475, 476, 477])}
MEJILLAS = [[50, 101, 118, 117, 123, 187, 205, 36], [280, 330, 347, 346, 352, 411, 425, 266]]
NARIZ = [[4, 45, 220, 115, 48, 64, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275]]
FRENTE = [[109, 10, 338, 337, 336, 296, 66, 107, 108]]
COLS = ["fuente", "clase", "archivo", "n_esp", "esp_L", "esp_a", "esp_b", "esp_b_p75", "piel_L", "piel_a", "piel_b",
        "frente_a", "nariz_a", "gw_esp_b", "gw_esp_a", "gw_piel_b", "esp_b_menos_piel", "rojo_mejilla", "rojo_nariz"]


def mascara(shape, pts, idxs) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for poly in idxs:
        cv2.fillPoly(m, [pts[poly].astype(np.int32)], 1)
    return m.astype(bool)


def lab(img_rgb01: np.ndarray, m: np.ndarray) -> np.ndarray:
    px = img_rgb01[m]
    return cv2.cvtColor(px[None].astype(np.float32), cv2.COLOR_RGB2Lab)[0] if len(px) else np.zeros((0, 3), np.float32)


def medir(rgb: np.ndarray, res) -> dict | None:
    if not res.face_landmarks:
        return None
    h, w = rgb.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
    img = rgb.astype(np.float32) / 255.0
    # balance de grises (gray world) para tener una versión con la luz neutralizada
    gw = np.clip(img * (img.reshape(-1, 3).mean(0).mean() / np.clip(img.reshape(-1, 3).mean(0), 1e-3, None)), 0, 1)
    esp = np.zeros((h, w), bool)
    for lado in ("d", "i"):
        abertura = mascara(img.shape, pts, [CONTORNO[lado]])
        c, anillo = IRIS[lado]
        r = np.mean([np.linalg.norm(pts[k] - pts[c]) for k in anillo])
        iris = np.zeros((h, w), np.uint8)
        cv2.circle(iris, tuple(pts[c].astype(int)), int(r * 1.1) + 1, 1, -1)
        esp |= abertura & ~iris.astype(bool)
    if esp.sum() < 30:
        return None
    L = lab(img, esp)
    claros = L[:, 0] >= np.percentile(L[:, 0], 40)  # sin pestañas ni sombras
    L = L[claros]
    Lg = lab(gw, esp)[claros]
    piel = lab(img, mascara(img.shape, pts, MEJILLAS))
    pielg = lab(gw, mascara(img.shape, pts, MEJILLAS))
    fr = lab(img, mascara(img.shape, pts, FRENTE))
    nz = lab(img, mascara(img.shape, pts, NARIZ))
    if len(piel) < 30 or len(fr) < 30:
        return None
    return {
        "n_esp": int(len(L)), "esp_L": float(np.median(L[:, 0])), "esp_a": float(np.median(L[:, 1])),
        "esp_b": float(np.median(L[:, 2])), "esp_b_p75": float(np.percentile(L[:, 2], 75)),
        "piel_L": float(np.median(piel[:, 0])), "piel_a": float(np.median(piel[:, 1])), "piel_b": float(np.median(piel[:, 2])),
        "frente_a": float(np.median(fr[:, 1])), "nariz_a": float(np.median(nz[:, 1])) if len(nz) else 0.0,
        "gw_esp_b": float(np.median(Lg[:, 2])), "gw_esp_a": float(np.median(Lg[:, 1])), "gw_piel_b": float(np.median(pielg[:, 2])),
        "esp_b_menos_piel": float(np.median(L[:, 2]) - np.median(piel[:, 2])),
        "rojo_mejilla": float(np.median(piel[:, 1]) - np.median(fr[:, 1])),
        "rojo_nariz": float((np.median(nz[:, 1]) if len(nz) else 0.0) - np.median(fr[:, 1])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ictericia", type=Path, required=True)
    ap.add_argument("--ffhq", type=Path)
    ap.add_argument("--max_ffhq", type=int, default=1500)
    ap.add_argument("--modelo", default="models/face_landmarker.task")
    ap.add_argument("--out", type=Path, default=Path("data/processed/rostro/color.csv"))
    a = ap.parse_args()
    det = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=a.modelo), running_mode=RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cuenta = {"ok": 0, "sin": 0}
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, COLS)
        wr.writeheader()

        def proc(bgr, **meta):
            if bgr is None:
                return
            s = 1024 / max(bgr.shape[:2])
            if s < 1:
                bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            r = medir(rgb, det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))))
            if r is None:
                cuenta["sin"] += 1
                return
            cuenta["ok"] += 1
            wr.writerow({**meta, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}})

        for f in sorted(a.ictericia.rglob("*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and f.parent.name in ("Jaundice", "Healthy"):
                proc(cv2.imread(str(f)), fuente="ictericia_ds", clase=int(f.parent.name == "Jaundice"), archivo=str(f))
        n = 0
        if a.ffhq:
            for tar in sorted(a.ffhq.glob("*.tar")):
                with tarfile.open(tar) as t:
                    for m in t:
                        if n >= a.max_ffhq:
                            break
                        if m.isfile() and m.name.lower().endswith((".png", ".jpg", ".webp")):
                            proc(cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8), cv2.IMREAD_COLOR),
                                 fuente="ffhq", clase=0, archivo=m.name)
                            n += 1
    print("LISTO", cuenta)


if __name__ == "__main__":
    main()
