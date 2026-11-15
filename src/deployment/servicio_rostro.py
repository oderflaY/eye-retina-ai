"""Servicio local de rasgos del rostro (MediaPipe). Corre en ``.venv-etiquetas`` en 127.0.0.1:8001.

MediaPipe choca con las versiones de TensorFlow de la API principal, así que vive aparte: este
servicio solo encuentra la cara y mide (asimetría, color del blanco del ojo, cajas de los ojos);
los modelos y el reporte los arma la API principal (``inference_rostro``).

    .venv-etiquetas/bin/python -m uvicorn src.deployment.servicio_rostro:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (FaceLandmarker, FaceLandmarkerOptions, HandLandmarker,
                                          HandLandmarkerOptions, RunningMode)

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src/data"))
from rasgos_color_rostro import CONTORNO, medir  # noqa: E402
from rasgos_rostro import rasgos  # noqa: E402

app = FastAPI(title="Rasgos del rostro (local)")
_det = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(RAIZ / "models/face_landmarker.task")), running_mode=RunningMode.IMAGE,
    num_faces=2, min_face_detection_confidence=0.5, output_face_blendshapes=True,
    output_facial_transformation_matrixes=True))
_candado = threading.Lock()  # el detector no es seguro entre hilos
_manos = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(RAIZ / "models/hand_landmarker.task")), running_mode=RunningMode.IMAGE,
    num_hands=1, min_hand_detection_confidence=0.4))
# Centro de la palma: muñeca, base del pulgar y nudillos de índice, medio, anular y meñique.
PALMA = [0, 1, 5, 9, 13, 17]


def _leer(datos: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "No se pudo leer la imagen")
    s = 1280 / max(bgr.shape[:2])
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


@app.post("/mano")
async def medir_mano(file: UploadFile = File(...)) -> dict:
    """Polígono del centro de la palma (normalizado 0..1) para recortar y medir el color."""
    rgb = _leer(await file.read())
    with _candado:
        res = _manos.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return {"mano": False}
    lm = res.hand_landmarks[0]
    pts = np.array([[lm[i].x, lm[i].y] for i in PALMA])
    centro = pts.mean(0)
    pts = centro + 0.75 * (pts - centro)  # se encoge hacia el centro: sin bordes ni dedos
    return {"mano": True, "confianza": float(res.handedness[0][0].score), "lado": res.handedness[0][0].category_name,
            "palma": pts.round(5).tolist()}


# Contorno interno de los labios (FaceMesh): la abertura de la boca, donde se ven dientes y encía.
BOCA_INTERNA = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


@app.get("/")
def salud() -> dict:
    return {"status": "ok", "servicio": "rostro"}


@app.post("/rasgos")
async def medir_rostro(file: UploadFile = File(...)) -> dict:
    datos = np.frombuffer(await file.read(), np.uint8)
    bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "No se pudo leer la imagen")
    s = 1280 / max(bgr.shape[:2])
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    with _candado:
        res = _det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return {"cara": False, "caras": 0}
    h, w = rgb.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
    ojos = {}
    for lado, nombre in (("d", "derecho"), ("i", "izquierdo")):
        c = pts[CONTORNO[lado]]
        x0, y0 = c.min(0)
        x1, y1 = c.max(0)
        ojos[nombre] = [float(x0 / w), float(y0 / h), float((x1 - x0) / w), float((y1 - y0) / h)]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return {
        "cara": True,
        "caras": len(res.face_landmarks),
        "asimetria": rasgos(res),
        "color": medir(rgb, res),
        "boca": (pts[BOCA_INTERNA] / [w, h]).round(5).tolist(),  # polígono normalizado de la abertura de la boca
        "ojos": ojos,  # cajas normalizadas (x, y, ancho, alto) de la abertura de cada ojo
        "caja_cara": [float(x0 / w), float(y0 / h), float((x1 - x0) / w), float((y1 - y0) / h)],
    }
