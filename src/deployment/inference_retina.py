"""Inferencia de la U-Net de vasos sobre fondos de ojo de cualquier tamaño.

Ejemplo::

    from src.deployment.inference_retina import RetinaSegmenter
    seg = RetinaSegmenter("models/retina/best.keras")
    out = seg.predict(imagen_rgb)          # prob/máscara a la resolución original
    out["vessel_density"]                  # fracción del FOV ocupada por vasos

El preprocesado (tamaño, CLAHE) y el umbral se leen de ``preprocess.json`` junto
al modelo, para que coincidan con el entrenamiento.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import tensorflow as tf

import src.models.seg_losses  # noqa: F401  (registra las clases serializables)
import src.models.unet  # noqa: F401
from src.data.retina_preprocess import RetinaGeometry, preprocess_fundus, restore_to_original

DEFAULT_PREPROCESS = {"size": 1024, "clahe": True, "threshold": 0.5}


def load_preprocess(model_path: Path) -> dict[str, Any]:
    """Lee ``preprocess.json`` junto al modelo (o valores por defecto)."""
    p = Path(model_path).with_name("preprocess.json")
    cfg = dict(DEFAULT_PREPROCESS)
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


class RetinaSegmenter:
    """Envuelve el modelo: preprocesado → U-Net (+TTA) → mapa a la resolución original."""

    def __init__(self, model_path: str | Path, tta: bool = True, model: tf.keras.Model | None = None) -> None:
        self.model_path = Path(model_path)
        self.model = model or tf.keras.models.load_model(self.model_path, compile=False)
        self.cfg = load_preprocess(self.model_path)
        self.size = int(self.cfg["size"])
        self.threshold = float(self.cfg["threshold"])
        self.tta = tta

    def predict_processed(self, image: np.ndarray) -> np.ndarray:
        """Probabilidad de vaso para una imagen ya preprocesada (uint8 o float [0, 1])."""
        x = image.astype(np.float32)
        if image.dtype == np.uint8:
            x /= 255.0
        h, w = x.shape[:2]
        ph, pw = (-h) % 32, (-w) % 32
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)))[None]
        # una imagen por pasada: con 1024 px un lote de 2 no cabe junto a otras cargas en 6 GB
        p = self._forward(tf.constant(x)).numpy()[0, ..., 0]
        if self.tta:
            p = (p + self._forward(tf.constant(x[:, :, ::-1].copy())).numpy()[0, :, ::-1, 0]) / 2
        return p[:h, :w]

    @tf.function(reduce_retracing=True)
    def _forward(self, x: tf.Tensor) -> tf.Tensor:
        return tf.sigmoid(tf.cast(self.model(x, training=False), tf.float32))

    def predict(self, image: np.ndarray, fov: np.ndarray | None = None,
                threshold: float | None = None) -> dict[str, Any]:
        """Segmenta un fondo de ojo RGB uint8 de cualquier tamaño.

        Returns:
            ``prob`` y ``mask`` (H, W) en la resolución original, ``fov`` estimado,
            ``vessel_density`` dentro del FOV, ``threshold`` y ``geometry``.
        """
        t = self.threshold if threshold is None else threshold
        img, fov_s, _, g = preprocess_fundus(image, size=self.size, fov=fov, clahe=bool(self.cfg["clahe"]))
        prob_s = self.predict_processed(img) * fov_s
        prob = restore_to_original(prob_s, g)
        fov_o = restore_to_original(fov_s.astype(np.float32), g) >= 0.5
        mask = (prob >= t) & fov_o
        density = float(mask[fov_o].mean()) if fov_o.any() else 0.0
        return {"prob": prob, "mask": mask, "fov": fov_o, "vessel_density": density, "threshold": t,
                "geometry": g}

    @staticmethod
    def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 120),
                alpha: float = 0.6) -> np.ndarray:
        """Superpone la máscara de vasos sobre la imagen original."""
        out = image.copy()
        out[mask] = (alpha * np.array(color) + (1 - alpha) * out[mask]).astype(np.uint8)
        return out


__all__ = ["RetinaSegmenter", "RetinaGeometry", "load_preprocess"]
