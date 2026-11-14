"""Crea el dataset del detector de ojos a partir de caras reales (FFHQ) etiquetadas con MediaPipe.

Corre en el entorno ``.venv-etiquetas`` (MediaPipe choca con las versiones de TensorFlow)::

    .venv-etiquetas/bin/python src/data/etiquetar_ffhq.py \
        --tars ~/TemrariosSpace/datasets/FFHQ/tars --out data/raw/eyes_ffhq

Para cada cara:

* **Positivos**: un recorte cuadrado por ojo, igual que la app OjoScan (2.5 × el ancho del ojo,
  con variación aleatoria de escala y posición), con las 6 etiquetas de docs/FORMATO_ETIQUETAS.md:
  caja de la abertura del ojo, esquina interna/externa, párpado superior/inferior, pupila (centro
  del iris) y borde lateral del iris.
* **Negativos**: un recorte del mismo tamaño en boca / nariz / mejilla / frente / barbilla que no
  contiene ningún ojo (fila sin caja = ``eye_prob 0``).

Se descartan caras con el ojo cerrado o muy de lado (etiquetas poco fiables).
"""

from __future__ import annotations

import argparse
import csv
import io
import tarfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# Índices de la malla de MediaPipe (478 puntos con iris). "R" = ojo derecho de la persona.
CONTORNO = {
    "derecho": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "izquierdo": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
}
ESQUINAS = {"derecho": (133, 33), "izquierdo": (362, 263)}  # (interna, externa)
PARPADOS = {"derecho": (159, 145), "izquierdo": (386, 374)}  # (superior, inferior) a la mitad del ojo
IRIS = {"derecho": (468, [469, 470, 471, 472]), "izquierdo": (473, [474, 475, 476, 477])}
NEGATIVOS = {"boca": 13, "nariz": 1, "mejilla_d": 205, "mejilla_i": 425, "barbilla": 152, "frente": 10,
             "ceja_d": 105, "ceja_i": 334, "puente": 168}

ESCALAS = (1.25, 3.6)  # lado del recorte / ancho del ojo: de "solo el ojo" a "ojo con parte de la cara"
RECORTES_POR_OJO = 2
LADO_MAX = 320
APERTURA_IRIS = 0.18  # alto/ancho de la abertura; menos = ojo (casi) cerrado: sin pupila/iris visibles
COLS = ["image", "x", "y", "w", "h"]
NOMBRES = ["inner_corner", "outer_corner", "upper_lid", "lower_lid", "pupil", "iris"]
for n in NOMBRES:
    COLS += [f"{n}_x", f"{n}_y"]
COLS += ["eye_side", "source"]


def recortar(img: np.ndarray, cx: float, cy: float, lado: float) -> tuple[np.ndarray, int, int, float] | None:
    h, w = img.shape[:2]
    lado = int(round(min(lado, w, h)))
    x0 = int(round(np.clip(cx - lado / 2, 0, w - lado)))
    y0 = int(round(np.clip(cy - lado / 2, 0, h - lado)))
    rec = img[y0 : y0 + lado, x0 : x0 + lado]
    esc = min(1.0, LADO_MAX / lado)
    if esc < 1:
        rec = cv2.resize(rec, (round(lado * esc), round(lado * esc)), interpolation=cv2.INTER_AREA)
    return rec, x0, y0, esc


def procesar(img: np.ndarray, pts: np.ndarray, rng: np.random.Generator, base: str, out: Path) -> list[list]:
    filas: list[list] = []
    centros = []
    lado_ref = 0.0
    for lado_ojo, idx in CONTORNO.items():
        c = pts[idx]
        x0, y0 = c.min(0)
        x1, y1 = c.max(0)
        ancho, alto = x1 - x0, y1 - y0
        centros.append((x0, y0, x1, y1))
        if ancho < 25:
            continue
        e_int, e_ext = (pts[i] for i in ESQUINAS[lado_ojo])
        p_sup, p_inf = (pts[i] for i in PARPADOS[lado_ojo])
        # apertura sin depender de la inclinación: párpado a párpado / esquina a esquina
        abierto = np.linalg.norm(p_inf - p_sup) / (np.linalg.norm(e_ext - e_int) + 1e-6) >= APERTURA_IRIS
        for k in range(RECORTES_POR_OJO):
            # escala log-uniforme: tantos recortes muy cerrados como abiertos
            lado = max(ancho, alto) * float(np.exp(rng.uniform(*np.log(ESCALAS))))
            lado_ref = max(lado_ref, lado)
            holgura = max(0.0, (lado - ancho) / 2) * 0.8  # el ojo siempre queda completo
            cx = (x0 + x1) / 2 + rng.uniform(-holgura, holgura)
            cy = (y0 + y1) / 2 + rng.uniform(-holgura, holgura) * 0.7
            rec, ox, oy, esc = recortar(img, cx, cy, lado)
            if x0 < ox or y0 < oy or x1 > ox + rec.shape[1] / esc or y1 > oy + rec.shape[0] / esc:
                continue

            def rel(p: np.ndarray) -> list:
                return [round((p[0] - ox) * esc, 1), round((p[1] - oy) * esc, 1)]

            interna, externa = (pts[i] for i in ESQUINAS[lado_ojo])
            centro_iris, anillo = IRIS[lado_ojo]
            eje = (externa - interna) / (np.linalg.norm(externa - interna) + 1e-6)
            borde = max(anillo, key=lambda i: float(pts[i] @ eje))
            nombre = f"{base}_{lado_ojo}{k}.jpg"
            cv2.imwrite(str(out / "images" / nombre), rec, [cv2.IMWRITE_JPEG_QUALITY, 92])
            fila: list = [f"images/{nombre}", round((x0 - ox) * esc, 1), round((y0 - oy) * esc, 1),
                          round(ancho * esc, 1), round(alto * esc, 1)]
            # Esquinas por posición EN LA IMAGEN (columna inner_corner = la de la izquierda): en un
            # recorte de solo el ojo no se ve de qué lado está la nariz, así que interna/externa
            # sería ambiguo. Párpados: puntos de la malla a la mitad del ojo (no el extremo en y,
            # que con el ojo inclinado cae en una esquina).
            izq, der = sorted((interna, externa), key=lambda q: float(q[0]))
            for p in (izq, der, p_sup, p_inf):
                fila += rel(p)
            fila += (rel(pts[centro_iris]) + rel(pts[borde])) if abierto else ["", "", "", ""]
            filas.append(fila + [lado_ojo, "ffhq_mediapipe" if abierto else "ffhq_mediapipe_cerrado"])

    if lado_ref > 0:
        # Negativos: zonas de la cara que no tocan ningún ojo, de tamaño parecido a un recorte de ojo.
        hechos_neg = 0
        for nombre_neg in rng.permutation(list(NEGATIVOS)):
            if hechos_neg >= 2:
                break
            p = pts[NEGATIVOS[nombre_neg]]
            lado = lado_ref * rng.uniform(0.4, 1.0)
            cx, cy = p + rng.uniform(-0.1, 0.1, 2) * lado
            rec, ox, oy, esc = recortar(img, cx, cy, lado)
            lado_real = rec.shape[0] / esc
            toca = any(ex1 > ox and ex0 < ox + lado_real and ey1 > oy and ey0 < oy + lado_real
                       for ex0, ey0, ex1, ey1 in centros)
            if toca:
                continue
            nombre = f"{base}_no_{nombre_neg}.jpg"
            cv2.imwrite(str(out / "images" / nombre), rec, [cv2.IMWRITE_JPEG_QUALITY, 92])
            filas.append([f"images/{nombre}", "", "", "", ""] + [""] * 12 + ["", f"negativo_{nombre_neg}"])
            hechos_neg += 1
    return filas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--modelo", default="models/face_landmarker.task")
    ap.add_argument("--max", type=int, default=0, help="máximo de caras (0 = todas)")
    args = ap.parse_args()

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    det = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.modelo), running_mode=RunningMode.IMAGE,
        num_faces=1, min_face_detection_confidence=0.6,
    ))
    csv_path = args.out / "annotations.csv"
    hechos = set()
    if csv_path.exists():
        with open(csv_path) as fh:
            hechos = {r["image"].split("/")[-1].split("_")[0] for r in csv.DictReader(fh)}
    nuevo = not csv_path.exists()
    caras = ojos = neg = sin_cara = 0
    with open(csv_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(COLS)
        for tar in sorted(args.tars.glob("*.tar")):
            with tarfile.open(tar) as t:
                for m in t:
                    if not m.isfile() or not m.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        continue
                    base = f"ffhq{Path(m.name).stem}"
                    if base in hechos:
                        continue
                    datos = np.frombuffer(t.extractfile(m).read(), np.uint8)
                    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                              data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                    if not res.face_landmarks:
                        sin_cara += 1
                        continue
                    h, wi = img.shape[:2]
                    pts = np.array([[p.x * wi, p.y * h] for p in res.face_landmarks[0]], np.float32)
                    filas = procesar(img, pts, rng, base, args.out)
                    w.writerows(filas)
                    caras += 1
                    ojos += sum(1 for f in filas if f[1] != "")
                    neg += sum(1 for f in filas if f[1] == "")
                    if caras % 250 == 0:
                        fh.flush()
                        print(f"{caras} caras → {ojos} ojos, {neg} negativos ({sin_cara} sin cara)", flush=True)
                    if args.max and caras >= args.max:
                        break
            if args.max and caras >= args.max:
                break
    print(f"LISTO: {caras} caras → {ojos} ojos, {neg} negativos ({sin_cara} sin cara)")


if __name__ == "__main__":
    main()
