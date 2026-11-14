"""Inferencia del detector de ojos (Keras ``.keras`` o TFLite ``.tflite``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

from src.data.encoding import Detection, GridEncoder
from src.data.eye_labels import LANDMARK_NAMES
from src.data.io import letterbox
from src.models.eye_grid_detector import build_grid_eye_detector

# Importar las clases registradas para que ``load_model`` las encuentre
from src.models import losses as _losses  # noqa: F401


def load_float32_model(model_path: str | Path) -> tuple[tf.keras.Model, dict[str, Any]]:
    """Carga un ``.keras`` y lo reconstruye en float32.

    Los modelos entrenados con ``mixed_float16`` guardan esa política por capa;
    para CPU y TFLite conviene un grafo float32 con los mismos pesos.
    """
    model_path = Path(model_path)
    meta_path = model_path.parent / "model_meta.json"
    trained = tf.keras.models.load_model(model_path, compile=False)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:  # deducir del propio modelo
        meta = {
            "image_size": int(trained.input_shape[1]),
            "backbone": "efficientnetb0" if "efficientnet" in trained.name else "mobilenetv2",
            "head_channels": int(trained.get_layer("lat_c5").filters),
        }
    prev = tf.keras.mixed_precision.global_policy().name
    tf.keras.mixed_precision.set_global_policy("float32")
    try:
        model = build_grid_eye_detector(meta["image_size"], meta["backbone"], None, meta["head_channels"])
        model.set_weights(trained.get_weights())
    finally:
        tf.keras.mixed_precision.set_global_policy(prev)
    return model, meta


class GridEyeDetector:
    """Detecta ojos y sus 6 landmarks en imágenes RGB de cualquier tamaño.

    Ejemplo::

        det = GridEyeDetector("models/eye_detection/eye_det_mobilenetv2_224/best.keras")
        resultados = det.predict(imagen_rgb)
    """

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        max_detections: int = 10,
    ) -> None:
        self.model_path = Path(model_path)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self._interp: Any = None
        if self.model_path.suffix == ".tflite":
            self._interp = tf.lite.Interpreter(model_path=str(self.model_path))
            self._interp.allocate_tensors()
            inp = self._interp.get_input_details()[0]
            self.image_size = int(inp["shape"][1])
            self.model = None
        else:
            self.model, meta = load_float32_model(self.model_path)
            self.image_size = int(meta["image_size"])
        self.encoder = GridEncoder(self.image_size)

    def raw_predict(self, batch: np.ndarray) -> np.ndarray:
        """Salida cruda (B, G, G, 17) para un lote ya en letterbox (float 0-255)."""
        batch = batch.astype(np.float32)
        if self._interp is None:
            return self.model(batch, training=False).numpy()
        inp = self._interp.get_input_details()[0]
        out = self._interp.get_output_details()[0]
        preds = []
        for img in batch:
            self._interp.set_tensor(inp["index"], img[None])
            self._interp.invoke()
            preds.append(self._interp.get_tensor(out["index"])[0])
        return np.stack(preds)

    def decode(self, pred: np.ndarray, score_threshold: float | None = None) -> list[Detection]:
        """Decodifica una salida (G, G, 17) a detecciones en espacio letterbox."""
        thr = self.score_threshold if score_threshold is None else score_threshold
        return self.encoder.decode(pred, thr, self.iou_threshold, self.max_detections)

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detecta ojos en una imagen RGB ``uint8`` (coordenadas originales).

        Returns:
            Lista de ``{"bbox": [x, y, w, h], "score": p, "landmarks": {nombre: [x, y]}}``.
        """
        lb, scale, px, py = letterbox(image, self.image_size)
        pred = self.raw_predict(lb[None])[0]
        h, w = image.shape[:2]
        out = []
        for d in self.decode(pred):
            x, y, bw, bh = d.box
            x0 = float(np.clip((x - px) / scale, 0, w))
            y0 = float(np.clip((y - py) / scale, 0, h))
            x1 = float(np.clip((x + bw - px) / scale, 0, w))
            y1 = float(np.clip((y + bh - py) / scale, 0, h))
            lms = (d.landmarks - [px, py]) / scale
            out.append(
                {
                    "bbox": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                    "score": round(d.score, 4),
                    "landmarks": {
                        n: [round(float(a), 1), round(float(b), 1)] for n, (a, b) in zip(LANDMARK_NAMES, lms)
                    },
                }
            )
        return out


def draw_detections(image: np.ndarray, dets: list[dict[str, Any]]) -> np.ndarray:
    """Dibuja cajas, puntaje y landmarks sobre una copia RGB de la imagen."""
    vis = image.copy()
    t = max(1, round(max(image.shape[:2]) / 400))
    colors = [(255, 80, 80), (255, 200, 0), (80, 200, 255), (80, 255, 120), (255, 0, 255), (255, 255, 255)]
    for d in dets:
        x, y, w, h = (int(round(v)) for v in d["bbox"])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), t)
        cv2.putText(
            vis, f"{d['score']:.2f}", (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, (0, 255, 0), t
        )
        for (px, py), c in zip(d["landmarks"].values(), colors):
            cv2.circle(vis, (int(px), int(py)), 2 * t, c, -1)
    return vis
