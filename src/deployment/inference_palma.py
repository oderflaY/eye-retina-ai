"""Anemia por la palidez de la palma de la mano (signo que usa la OMS / AIEPI).

El servicio local de MediaPipe (``servicio_rostro``, POST /mano) encuentra la mano y devuelve el
polígono del centro de la palma; aquí se recorta, se mide el color igual que en el entrenamiento
(``experimento_anemia_ghana.medir_imagen``) y se aplica el modelo de ``train_anemia_palma``.

La respuesta tiene la misma forma que ``/predict/anemia`` (ojo) para que la app use la misma pantalla.
"""

from __future__ import annotations

import json
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np

from src.training.experimento_anemia_ghana import medir_imagen

SERVICIO = os.environ.get("ROSTRO_SERVICIO", "http://127.0.0.1:8001")
LADO_MAX = 1280
AVISO = ("Tamizaje de investigación entrenado con fotos de niños de Ghana (6 meses a 5 años): en adultos y "
         "con otras cámaras puede fallar más. La anemia solo se confirma con un análisis de sangre (hemoglobina).")


def _mano(jpeg: bytes) -> dict:
    limite = uuid.uuid4().hex
    cuerpo = (f"--{limite}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"mano.jpg\"\r\n"
              f"Content-Type: image/jpeg\r\n\r\n").encode() + jpeg + f"\r\n--{limite}--\r\n".encode()
    req = urllib.request.Request(f"{SERVICIO}/mano", data=cuerpo, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={limite}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


class AnalizadorPalma:
    def __init__(self, ruta: Path) -> None:
        paq = joblib.load(ruta)
        self.modelo, self.cols = paq["modelo"], paq["caracteristicas"]
        self.zona = tuple(paq["zona_gris"])
        self.cv = paq["validacion"]

    def predict(self, rgb: np.ndarray) -> dict[str, Any]:
        s = LADO_MAX / max(rgb.shape[:2])
        if s < 1:  # misma escala que ve el servicio
            rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        base = {"model": "palma", "aviso": AVISO, "fraccion_conjuntiva": 0.0}
        sin = {**base, "valida": False, "resultado": "sin_palma", "titulo": "No encontré la palma de tu mano",
               "descripcion": "Abre la mano con la palma hacia la cámara, los dedos un poco separados, "
                              "dentro del recuadro y con luz de día."}
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        try:
            m = _mano(buf.tobytes())
        except Exception as exc:  # servicio caído
            return {**sin, "titulo": "Servicio de manos no disponible", "descripcion": str(exc)}
        if not m.get("mano"):
            return sin
        h, w = bgr.shape[:2]
        poli = (np.array(m["palma"]) * [w, h]).astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(poli)
        if min(bw, bh) < 40:
            return {**sin, "titulo": "La mano está muy lejos",
                    "descripcion": "Acerca la cámara hasta que la mano llene el recuadro."}
        alfa = np.zeros((h, w), np.uint8)
        cv2.fillPoly(alfa, [poli], 255)
        recorte = np.dstack([bgr, alfa])[y:y + bh, x:x + bw]
        med = medir_imagen(recorte)
        if med is None:
            return {**sin, "titulo": "La palma salió muy oscura o con reflejo",
                    "descripcion": "Repite con luz de día pareja, sin flash y sin sombra sobre la mano."}
        p = float(self.modelo.predict_proba(np.array([[med[c] for c in self.cols]]))[0, 1])
        bajo, alto = self.zona
        if p >= alto:
            resultado, titulo, desc = "palidez", "Palma pálida: posible anemia", (
                "La palma se ve más pálida de lo esperado. Conviene hacer una biometría hemática para medir "
                "la hemoglobina. Compárala también con el modo Anemia (párpado).")
        elif p < bajo:
            resultado, titulo, desc = "sin_palidez", "Color de palma normal", (
                "La palma tiene buena coloración; no se ven signos de anemia en ella.")
        else:
            resultado, titulo, desc = "no_concluyente", "Resultado no concluyente", (
                "El color quedó en un punto intermedio. Repite con luz de día, sin flash, con la mano relajada "
                "(sin apretar ni estirar de más). Si vuelve a salir así, usa también el modo Anemia del párpado.")
        res = {**base, "valida": True, "resultado": resultado, "probabilidad_anemia": round(p, 4),
               "zona_gris": [round(bajo, 4), round(alto, 4)], "titulo": titulo, "descripcion": desc,
               "medidas": {k: round(v, 4) for k, v in med.items() if k in self.cols}, "validacion": self.cv}
        if med["L"] > 85 or med["L"] < 25:
            res["advertencia_luz"] = ("La foto quedó muy clara u oscura; la luz cambia el color de la piel. "
                                      "Repite con luz de día pareja para un resultado más fiable.")
        return res
