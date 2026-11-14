"""Analizador de ojo externo: foto del ojo (teléfono) → probabilidad de cada hallazgo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

from src.data.prepare_ojo_externo import cuadrar

NOMBRES = {
    "normal": "Sin hallazgos visibles",
    "enrojecimiento": "Enrojecimiento / inflamación",
    "pterigion": "Pterigión (carnosidad)",
    "catarata": "Posible catarata",
}
DESCRIPCIONES = {
    "normal": "No se ven signos de las condiciones que el modelo conoce.",
    "enrojecimiento": "Ojo rojo o irritado, como en conjuntivitis o blefaritis. Si dura más de unos días, "
    "duele o hay secreción, consulta a un médico.",
    "pterigion": "Tejido que crece sobre la parte blanca del ojo hacia la córnea. Lo confirma un oftalmólogo.",
    "catarata": "Opacidad blanquecina o grisácea en la pupila. Lo confirma un oftalmólogo con lámpara de hendidura.",
}
AVISO = ("Herramienta de investigación entrenada con fotos públicas; no es un diagnóstico. "
         "Ante cualquier molestia consulta a un oftalmólogo.")


class AnalizadorOjoExterno:
    def __init__(self, model_path: str | Path) -> None:
        self.path = Path(model_path)
        self.model = tf.keras.models.load_model(self.path, compile=False)
        cfg = json.loads((self.path.parent / "config.json").read_text())
        self.clases: list[str] = cfg["clases"]
        self.tam: int = cfg["tam_entrada"]
        self.metricas_test: dict = cfg.get("test", {})
        self._f = tf.function(lambda x: self.model(x, training=False), reduce_retracing=True)

    def preparar(self, image: np.ndarray) -> np.ndarray:
        """``image`` RGB uint8 → cuadrado como en el entrenamiento (letterbox 288 → resize)."""
        img = cuadrar(image[..., ::-1])[..., ::-1]
        return cv2.resize(img, (self.tam, self.tam), interpolation=cv2.INTER_AREA).astype(np.float32)

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        x = self.preparar(image)
        lote = np.stack([x, x[:, ::-1]])  # TTA: espejo
        logits = self._f(tf.constant(lote)).numpy().mean(0)
        p = tf.nn.softmax(logits).numpy()
        orden = np.argsort(-p)
        principal = self.clases[int(orden[0])]
        auc = self.metricas_test.get("por_clase", {})
        return {
            "principal": principal,
            "titulo": NOMBRES[principal],
            "descripcion": DESCRIPCIONES[principal],
            "confianza": round(float(p[orden[0]]), 4),
            "hallazgos": [
                {
                    "clave": self.clases[int(i)],
                    "nombre": NOMBRES[self.clases[int(i)]],
                    "probabilidad": round(float(p[i]), 4),
                    "auc_test": auc.get(self.clases[int(i)], {}).get("auc"),
                }
                for i in orden
            ],
            "aviso": AVISO,
            "model": self.path.parent.name,
        }
