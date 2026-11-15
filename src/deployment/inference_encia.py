"""Anemia por el color de la ENCÍA (experimental).

Cómo se encuentra la encía (no hay dataset público con la encía marcada):
    abertura de la boca (labios internos, MediaPipe en servicio_rostro :8001)
    − dientes (U-Net entrenada con Dentalai, models/dientes)
    − sombras y brillos  = encía
Los dientes sirven además de blanco de referencia para corregir la luz (como la esclerótica en la
conjuntiva).

Cómo se decide: el color corregido se compara con el de encías SANAS del dataset de gingivitis
(Mendeley 3253gj88rr, grado 0 según periodoncistas), en models/encia/referencia.json:
- más pálida que el 90 % de las encías sanas → palidez;
- más roja que las encías inflamadas típicas → inflamación (la gingivitis enrojece y puede tapar
  la palidez: el resultado no cuenta para anemia).

NO está validado con hemoglobina: no existe un dataset público de encía + hemoglobina. Por eso la
respuesta lleva ``experimental: true`` y en el pronóstico solo puede apoyar a la conjuntiva, no
decidir sola.
"""
from __future__ import annotations

import json
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.deployment.inference_anemia import medir_color

SERVICIO = os.environ.get("ROSTRO_SERVICIO", "http://127.0.0.1:8001")
LADO = 256
AVISO = ("EXPERIMENTAL: la encía se compara con encías sanas de fotos dentales, pero este método no se ha "
         "validado con análisis de sangre. Úsalo solo como apoyo; la anemia se confirma con hemoglobina.")


def _rasgos(jpeg: bytes) -> dict:
    limite = uuid.uuid4().hex
    cuerpo = (f"--{limite}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"boca.jpg\"\r\n"
              f"Content-Type: image/jpeg\r\n\r\n").encode() + jpeg + f"\r\n--{limite}--\r\n".encode()
    req = urllib.request.Request(f"{SERVICIO}/rasgos", data=cuerpo, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={limite}"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


class SegmentadorDientes:
    def __init__(self, ruta: Path) -> None:
        import tensorflow as tf

        import src.models.unet  # noqa: F401  registra capas propias

        self.tf = tf
        self.modelo = tf.keras.models.load_model(ruta, compile=False)

    def __call__(self, rgb01: np.ndarray) -> np.ndarray:
        """Máscara booleana de dientes del mismo tamaño que la imagen."""
        tf = self.tf
        h, w = rgb01.shape[:2]
        x = tf.constant(cv2.resize(rgb01, (LADO, LADO), interpolation=cv2.INTER_AREA)[None].astype(np.float32))
        p = tf.sigmoid(self.modelo(x, training=False)).numpy()[0, ..., 0]
        q = tf.sigmoid(self.modelo(x[:, :, ::-1], training=False)).numpy()[0, :, ::-1, 0]
        return cv2.resize((p + q) / 2, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5


MIN_BLANCO = 60


def blanco_dientes(rgb01: np.ndarray, dientes: np.ndarray) -> tuple[np.ndarray, int]:
    """Color de los dientes (los más claros y menos saturados) como blanco de referencia."""
    if dientes.sum() < MIN_BLANCO:
        return np.ones(3, np.float32), int(dientes.sum())
    hsv = cv2.cvtColor((rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    v, s = hsv[..., 2], hsv[..., 1]
    cand = dientes & (v > 0.25) & (v < 0.995)
    if cand.sum() < MIN_BLANCO:
        return np.ones(3, np.float32), int(cand.sum())
    claros = cand & (v >= np.percentile(v[cand], 60))
    sel = claros & (s <= np.percentile(s[claros], 60))
    return np.median(rgb01[sel], axis=0).astype(np.float32), int(sel.sum())


def solo_sobre_dientes(region: np.ndarray, dientes: np.ndarray) -> np.ndarray:
    """La encía de arriba está ENCIMA de los dientes superiores: por cada columna con diente, solo los
    píxeles de la boca por arriba del primer diente. Así no entran la lengua ni el interior de la boca."""
    h, w = region.shape
    cols = dientes.any(0)
    tope = np.where(cols, dientes.argmax(0), -1)  # primera fila con diente en cada columna
    filas = np.arange(h)[:, None]
    return region & cols[None, :] & (filas < tope[None, :])


def mascara_encia(rgb01: np.ndarray, region: np.ndarray, dientes: np.ndarray) -> np.ndarray:
    """Encía = región − dientes (con margen) − sombra − brillo, y solo tonos rosados/rojos."""
    hsv = cv2.cvtColor((rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.float32) * 2  # 0..360
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    sin_dientes = ~cv2.dilate(dientes.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    rosado = ((h < 30) | (h > 320)) & (s > 0.12)
    m = region & sin_dientes & rosado & (v > 0.22) & (v < 0.98)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    return m


def medir_encia(rgb01: np.ndarray, region: np.ndarray, seg: SegmentadorDientes) -> dict[str, Any]:
    dientes = seg(rgb01) & region
    encia = mascara_encia(rgb01, solo_sobre_dientes(region, dientes), dientes)
    blanco, n_blanco = blanco_dientes(rgb01, dientes)
    out = {"px_encia": int(encia.sum()), "px_dientes": int(dientes.sum()), "px_region": int(region.sum()),
           "px_blanco": n_blanco, "corregido": n_blanco >= MIN_BLANCO}
    if encia.sum() >= 80:
        out["color"] = medir_color(rgb01, encia, blanco)
    out["_encia"], out["_dientes"] = encia, dientes
    return out


class AnalizadorEncia:
    def __init__(self, carpeta_dientes: Path, referencia: Path) -> None:
        self.seg = SegmentadorDientes(carpeta_dientes / "best.keras")
        self.ref = json.loads(referencia.read_text())

    def predict(self, jpeg: bytes, rgb: np.ndarray) -> dict[str, Any]:
        base = {"model": "encia-experimental", "experimental": True, "aviso": AVISO, "validacion": None,
                "probabilidad_anemia": None, "umbral": None, "zona_gris": None}
        r = _rasgos(jpeg)
        if not r.get("cara") or not r.get("boca"):
            return {**base, "valida": False, "resultado": "sin_boca", "titulo": "No encontré tu boca",
                    "descripcion": "Toma la foto de frente, con la cara completa y buena luz."}
        # rasgos devuelve coordenadas normalizadas; se recorta la boca con margen
        H, W = rgb.shape[:2]
        poli = (np.array(r["boca"]) * [W, H]).astype(np.int32)
        x0, y0 = poli.min(0)
        x1, y1 = poli.max(0)
        mx, my = int((x1 - x0) * 0.15) + 4, int((y1 - y0) * 0.25) + 4
        x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)
        recorte = (rgb[y0:y1, x0:x1].astype(np.float32) / 255.0)
        region = np.zeros(recorte.shape[:2], np.uint8)
        cv2.fillPoly(region, [poli - [x0, y0]], 1)
        region = region.astype(bool)
        abierta = region.sum() / max(1, (y1 - y0) * (x1 - x0))
        if region.sum() < 400 or abierta < 0.08:
            return {**base, "valida": False, "resultado": "boca_cerrada", "titulo": "La boca está cerrada",
                    "descripcion": "Sonríe enseñando la encía de arriba: levanta el labio con los dientes juntos."}

        m = medir_encia(recorte, region, self.seg)
        if m["px_dientes"] < MIN_BLANCO:
            return {**base, "valida": False, "resultado": "sin_dientes", "titulo": "No se ven tus dientes",
                    "descripcion": "Sonríe con los dientes juntos: la encía se busca justo encima de los dientes de arriba."}
        if "color" not in m or m["px_encia"] < 0.03 * m["px_region"]:
            return {**base, "valida": False, "resultado": "sin_encia", "titulo": "No se ve la encía",
                    "descripcion": "Levanta el labio de arriba con un dedo para que se vea la parte rosada sobre los dientes.",
                    "px_encia": m["px_encia"]}

        c = m["color"]
        ref = self.ref
        eri = c["eritema"]
        # percentil del eritema entre encías sanas de referencia
        sanas = np.array(ref["sanas_eritema"])
        pct = float((sanas < eri).mean())
        inflam = eri >= ref["umbral_inflamacion"]
        if inflam:
            resultado, titulo = "no_concluyente", "Encía muy roja (posible inflamación)"
            desc = ("La encía se ve más roja que una encía sana; la gingivitis enrojece y puede ocultar la palidez, "
                    "así que no sirve para la anemia. Si sangra al cepillarte, consulta a tu dentista.")
        elif pct <= ref["pct_palidez"]:
            resultado, titulo = "palidez", "Encía pálida: posible anemia"
            desc = (f"Tu encía es más pálida que el {100 - pct * 100:.0f} % de las encías sanas de referencia. "
                    "Confírmalo con la conjuntiva o con un análisis de sangre.")
        elif pct <= ref["pct_gris"]:
            resultado, titulo = "no_concluyente", "Color de encía en el límite"
            desc = "La encía está algo pálida, pero dentro de lo que se ve en personas sanas."
        else:
            resultado, titulo = "sin_palidez", "Color de encía normal"
            desc = "La encía tiene el color rosado de una encía sana."
        res = {**base, "valida": True, "resultado": resultado, "titulo": titulo, "descripcion": desc,
               "percentil_sanas": round(pct, 3), "inflamacion": bool(inflam),
               "fraccion_conjuntiva": round(m["px_encia"] / max(1, m["px_region"]), 3),
               "eritema": round(eri, 4), "px_encia": m["px_encia"]}
        if not m["corregido"]:
            res["advertencia_luz"] = "No se vieron suficientes dientes para corregir la luz; el color depende más de la iluminación."
        return res
