"""Inferencia del detector de ojos (``.h5``, ``.keras`` o SavedModel).

Ejemplo::

    from src.deployment.inference_eye import EyeDetector
    det = EyeDetector("models/eye_best.h5")
    det.predict(imagen_rgb)
    # {'bbox': [x, y, w, h], 'landmarks': [{'name': 'pupil', 'x': .., 'y': ..}, ...],
    #  'confidence': 0.98, 'is_eye': True}
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import tensorflow as tf

from src.data.eye_labels import LANDMARK_NAMES
from src.models.eye_detector import CUSTOM_OBJECTS


def load_keras_model(path: str | Path) -> tf.keras.Model:
    """Carga un modelo ``.h5``/``.keras`` sin compilar (solo inferencia)."""
    return tf.keras.models.load_model(str(path), custom_objects=CUSTOM_OBJECTS, compile=False)


class EyeDetector:
    """Envuelve el modelo y convierte sus salidas a coordenadas de la imagen original.

    Args:
        model_path: ruta a ``.h5``/``.keras`` o a una carpeta SavedModel.
        threshold: probabilidad mínima para considerar que hay un ojo.
    """

    def __init__(self, model_path: str | Path, threshold: float = 0.5) -> None:
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self._serve: Any = None
        if self.model_path.is_dir():  # SavedModel (exportado con model.export / tf.saved_model.save)
            loaded = tf.saved_model.load(str(self.model_path))
            self._serve = loaded.signatures.get("serving_default") if hasattr(loaded, "signatures") else None
            if self._serve is None:
                self._serve = loaded.serve
            spec = (
                self._serve.structured_input_signature[1]
                if hasattr(self._serve, "structured_input_signature")
                else {}
            )
            shape = next(iter(spec.values())).shape if spec else (None, 224, 224, 3)
            self.input_size = (int(shape[1] or 224), int(shape[2] or 224))
            self.model = None
        else:
            self.model = load_keras_model(self.model_path)
            self.input_size = (int(self.model.input_shape[1]), int(self.model.input_shape[2]))

    def preprocess(self, image: np.ndarray) -> tf.Tensor:
        """Convierte a RGB, redimensiona (bilineal), normaliza a [0, 1] y agrega el eje de lote.

        Args:
            image: imagen ``uint8`` (H, W), (H, W, 1), (H, W, 3) o (H, W, 4).

        Returns:
            Tensor float32 (1, alto, ancho, 3).
        """
        img = np.asarray(image)
        if img.ndim == 2:
            img = img[..., None]
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        x = tf.image.resize(tf.convert_to_tensor(img), self.input_size, method="bilinear")
        return tf.cast(x, tf.float32)[None] / 255.0

    def _run(self, batch: tf.Tensor) -> dict[str, np.ndarray]:
        """Ejecuta el modelo y devuelve las tres salidas como arreglos NumPy."""
        if self.model is not None:
            out = self.model(batch, training=False)
        else:
            out = self._serve(batch)
            if not isinstance(out, dict):  # firma con salida en lista
                out = dict(zip(("bbox", "eye_prob", "landmarks"), out))
            # model.export usa los nombres de las salidas; tf.saved_model.save puede prefijarlos
            out = {
                next(k for k in ("bbox", "landmarks", "eye_prob") if k in name): v for name, v in out.items()
            }
        return {k: np.asarray(v, np.float32) for k, v in out.items()}

    def _to_dict(self, out: dict[str, np.ndarray], i: int, w: int, h: int) -> dict[str, Any]:
        """Desnormaliza la predicción ``i`` al tamaño original (w, h)."""
        conf = float(out["eye_prob"][i, 0])
        bx, by, bw, bh = out["bbox"][i]
        x0, y0 = float(np.clip(bx * w, 0, w)), float(np.clip(by * h, 0, h))
        x1, y1 = float(np.clip((bx + bw) * w, 0, w)), float(np.clip((by + bh) * h, 0, h))
        pts = out["landmarks"][i].reshape(-1, 2) * [w, h]
        return {
            "bbox": [round(x0, 1), round(y0, 1), round(max(x1 - x0, 0.0), 1), round(max(y1 - y0, 0.0), 1)],
            "landmarks": [
                {"name": n, "x": round(float(px), 1), "y": round(float(py), 1)}
                for n, (px, py) in zip(LANDMARK_NAMES, pts)
            ],
            "confidence": round(conf, 4),
            "is_eye": conf >= self.threshold,
        }

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """Predice caja, landmarks y confianza para una imagen.

        Args:
            image: imagen RGB ``uint8`` de cualquier tamaño.

        Returns:
            ``{"bbox": [x, y, w, h], "landmarks": [...], "confidence": p, "is_eye": bool}``
            en píxeles de la imagen original.
        """
        h, w = np.asarray(image).shape[:2]
        return self._to_dict(self._run(self.preprocess(image)), 0, w, h)

    def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        """Predice para varias imágenes (de tamaños distintos) en un solo lote."""
        if not images:
            return []
        batch = tf.concat([self.preprocess(im) for im in images], axis=0)
        out = self._run(batch)
        return [self._to_dict(out, i, im.shape[1], im.shape[0]) for i, im in enumerate(images)]


if __name__ == "__main__":
    import argparse
    import json

    from src.data.io import read_image_rgb

    ap = argparse.ArgumentParser(description="Detecta un ojo en una imagen")
    ap.add_argument("image")
    ap.add_argument("--model", default="models/eye_best.h5")
    args = ap.parse_args()
    print(
        json.dumps(EyeDetector(args.model).predict(read_image_rgb(args.image)), indent=2, ensure_ascii=False)
    )
