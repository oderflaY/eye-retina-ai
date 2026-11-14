"""Codificación de cajas y landmarks en una rejilla tipo YOLO-lite.

La imagen de ``S×S`` se divide en ``G×G`` celdas (``G = S / 16``). La celda que
contiene el centro de un ojo es responsable de predecirlo. Por celda:

======  =========  ==========================================================
canal   nombre     significado
======  =========  ==========================================================
0       obj        1 si la celda contiene el centro de un ojo
1–2     off_x/y    desplazamiento del centro dentro de la celda, en [0, 1)
3–4     √w, √h     raíz del ancho/alto normalizado por S (equilibra ojos chicos)
5–16    lm         6 landmarks: (l - centro) / (w, h) de la caja
17–22   lm_mask    1 si el landmark k está anotado (solo en el objetivo)
======  =========  ==========================================================

El modelo predice 17 canales (0–16) sin activación; la pérdida aplica sigmoid
a obj y a la caja.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.eye_labels import NUM_LANDMARKS

PRED_CHANNELS = 1 + 4 + 2 * NUM_LANDMARKS  # 17
TARGET_CHANNELS = PRED_CHANNELS + NUM_LANDMARKS  # 23
STRIDE = 16


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoide numéricamente estable."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass
class Detection:
    """Un ojo detectado en coordenadas de píxel."""

    box: np.ndarray  # (4,) x, y, w, h
    score: float
    landmarks: np.ndarray  # (6, 2)


class GridEncoder:
    """Convierte anotaciones <-> tensores de rejilla."""

    def __init__(self, image_size: int, stride: int = STRIDE) -> None:
        if image_size % 32:
            raise ValueError("image_size debe ser múltiplo de 32")
        self.size = image_size
        self.grid = image_size // stride
        self.cell = image_size / self.grid

    def encode(self, boxes: np.ndarray, landmarks: np.ndarray, lm_mask: np.ndarray) -> tuple[np.ndarray, int]:
        """Codifica las anotaciones de una imagen.

        Args:
            boxes: (N, 4) x, y, w, h en píxeles de la imagen ``S×S``.
            landmarks: (N, 6, 2) en píxeles.
            lm_mask: (N, 6) booleano, landmark anotado.

        Returns:
            (objetivo (G, G, 23) float32, número de colisiones). Si dos ojos
            caen en la misma celda se conserva el de mayor área.
        """
        g = self.grid
        target = np.zeros((g, g, TARGET_CHANNELS), np.float32)
        area = np.zeros((g, g), np.float32)
        collisions = 0
        for box, lm, m in zip(boxes, landmarks, lm_mask):
            x, y, w, h = (float(v) for v in box)
            if w < 1 or h < 1:
                continue
            cx, cy = x + w / 2, y + h / 2
            i = int(np.clip(cx // self.cell, 0, g - 1))  # columna
            j = int(np.clip(cy // self.cell, 0, g - 1))  # fila
            if target[j, i, 0] == 1:
                collisions += 1
                if w * h <= area[j, i]:
                    continue
            area[j, i] = w * h
            target[j, i, 0] = 1.0
            target[j, i, 1] = np.clip(cx / self.cell - i, 0.0, 0.999)
            target[j, i, 2] = np.clip(cy / self.cell - j, 0.0, 0.999)
            target[j, i, 3] = np.sqrt(w / self.size)
            target[j, i, 4] = np.sqrt(h / self.size)
            rel = (np.nan_to_num(lm) - [cx, cy]) / [w, h]
            m = np.asarray(m, bool)
            rel[~m] = 0.0
            target[j, i, 5:17] = rel.reshape(-1)
            target[j, i, 17:23] = m.astype(np.float32)
        return target, collisions

    def decode(
        self,
        pred: np.ndarray,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        max_detections: int = 10,
        is_target: bool = False,
    ) -> list[Detection]:
        """Convierte la salida (G, G, 17) en detecciones con NMS.

        Args:
            is_target: True si ``pred`` es un objetivo codificado (sin logits);
                útil para pruebas de ida y vuelta.
        """
        obj = pred[..., 0] if is_target else sigmoid(pred[..., 0])
        box = pred[..., 1:5] if is_target else sigmoid(pred[..., 1:5])
        js, is_ = np.where(obj >= score_threshold)
        dets: list[Detection] = []
        for j, i in zip(js, is_):
            ox, oy, sw, sh = box[j, i]
            cx, cy = (i + ox) * self.cell, (j + oy) * self.cell
            w, h = (sw**2) * self.size, (sh**2) * self.size
            lm = pred[j, i, 5:17].reshape(NUM_LANDMARKS, 2) * [w, h] + [cx, cy]
            dets.append(
                Detection(
                    np.array([cx - w / 2, cy - h / 2, w, h], np.float32),
                    float(obj[j, i]),
                    lm.astype(np.float32),
                )
            )
        return nms(dets, iou_threshold)[:max_detections]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre dos conjuntos de cajas x, y, w, h -> matriz (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    a = np.asarray(a, np.float32)[:, None, :]
    b = np.asarray(b, np.float32)[None, :, :]
    x0 = np.maximum(a[..., 0], b[..., 0])
    y0 = np.maximum(a[..., 1], b[..., 1])
    x1 = np.minimum(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2])
    y1 = np.minimum(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    union = a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter
    return inter / np.maximum(union, 1e-9)


def nms(dets: list[Detection], iou_threshold: float) -> list[Detection]:
    """Non-maximum suppression voraz por puntaje."""
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    keep: list[Detection] = []
    for d in dets:
        if not keep:
            keep.append(d)
            continue
        ious = iou_matrix(d.box[None], np.stack([k.box for k in keep]))[0]
        if (ious < iou_threshold).all():
            keep.append(d)
    return keep
