"""Pipeline ``tf.data`` para detección de ojos.

Flujo por muestra::

    índice ─► leer + letterbox (cache) ─► aumentar (solo train) ─► codificar rejilla
          ─► shuffle ─► batch ─► prefetch

La normalización (0-255 -> rango del backbone) vive *dentro* del modelo, así
la API y TFLite reciben la imagen tal cual.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.data.augment import build_eye_transform
from src.data.encoding import TARGET_CHANNELS, GridEncoder
from src.data.eye_labels import NUM_LANDMARKS, EyeSample
from src.data.io import letterbox, read_image_rgb

AUTOTUNE = tf.data.AUTOTUNE


def load_processed(processed_dir: str | Path) -> tuple[list[EyeSample], dict[str, Any]]:
    """Carga ``annotations.json`` y ``splits.json`` generados por ``prepare_eyes``."""
    processed_dir = Path(processed_dir)
    ann = json.loads((processed_dir / "annotations.json").read_text(encoding="utf-8"))
    splits = json.loads((processed_dir / "splits.json").read_text(encoding="utf-8"))
    return [EyeSample.from_dict(d) for d in ann["samples"]], splits


def load_letterboxed(
    sample: EyeSample, size: int, max_eyes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]:
    """Lee la imagen, aplica letterbox y lleva las anotaciones a ese espacio.

    Returns:
        imagen (S, S, 3) uint8, cajas (M, 4), landmarks (M, 6, 2),
        máscara de landmarks (M, 6) y número real de ojos; todo con relleno
        hasta ``max_eyes`` para tener formas fijas.
    """
    img = read_image_rgb(sample.image_path)
    out, scale, px, py = letterbox(img, size)
    n = min(sample.num_eyes, max_eyes)
    boxes = np.zeros((max_eyes, 4), np.float32)
    lms = np.zeros((max_eyes, NUM_LANDMARKS, 2), np.float32)
    mask = np.zeros((max_eyes, NUM_LANDMARKS), np.float32)
    if n:
        b = sample.boxes[:n].copy()
        boxes[:n] = b * [scale, scale, scale, scale] + [px, py, 0, 0]
        lm = sample.landmarks[:n]
        mask[:n] = (~np.isnan(lm).any(axis=-1)).astype(np.float32)
        lms[:n] = np.nan_to_num(lm) * scale + [px, py]
    return out, boxes, lms, mask, np.int32(n)


class EyeDatasetBuilder:
    """Construye los ``tf.data.Dataset`` de train/val/test."""

    def __init__(self, samples: list[EyeSample], cfg: dict[str, Any]) -> None:
        self.samples = samples
        self.cfg = cfg
        self.size = int(cfg["data"]["image_size"])
        self.max_eyes = int(cfg["data"].get("max_eyes", 8))
        self.encoder = GridEncoder(self.size)
        aug = cfg.get("augmentation", {})
        self.transform = build_eye_transform(aug) if aug.get("enabled", True) else None
        self.collisions = 0

    # -- funciones numpy (se ejecutan vía tf.numpy_function) ------------------
    def _np_load(self, idx: np.ndarray) -> tuple[np.ndarray, ...]:
        return load_letterboxed(self.samples[int(idx)], self.size, self.max_eyes)

    def _np_encode(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
        lms: np.ndarray,
        mask: np.ndarray,
        n: np.int32,
        augment: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(n)
        boxes, lms, mask = boxes[:n], lms[:n], mask[:n] > 0
        if augment and self.transform is not None:
            img, boxes, lms, mask = self._augment(img, boxes, lms, mask)
        target, coll = self.encoder.encode(boxes, lms, mask)
        self.collisions += coll
        return img.astype(np.float32), target

    def _augment(
        self, img: np.ndarray, boxes: np.ndarray, lms: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Aplica Albumentations y conserva la correspondencia caja ↔ landmarks."""
        n = len(boxes)
        kps = lms.reshape(-1, 2) if n else np.zeros((0, 2), np.float32)
        out = self.transform(
            image=img,
            bboxes=boxes.tolist(),
            box_ids=list(range(n)),
            keypoints=kps.tolist(),
            kp_ids=[k // NUM_LANDMARKS for k in range(len(kps))],
        )
        kept = [int(i) for i in out["box_ids"]]
        new_kps = np.asarray(out["keypoints"], np.float32).reshape(-1, NUM_LANDMARKS, 2)
        new_boxes = np.asarray(out["bboxes"], np.float32).reshape(-1, 4)
        new_lms = new_kps[kept] if kept else np.zeros((0, NUM_LANDMARKS, 2), np.float32)
        new_mask = mask[kept] if kept else np.zeros((0, NUM_LANDMARKS), bool)
        # Un landmark que salió de la imagen deja de contar en la pérdida
        inside = (
            (new_lms[..., 0] >= 0)
            & (new_lms[..., 0] < self.size)
            & (new_lms[..., 1] >= 0)
            & (new_lms[..., 1] < self.size)
        )
        return out["image"], new_boxes, new_lms, new_mask & inside

    # -- construcción ---------------------------------------------------------
    def build(self, indices: list[int], training: bool, batch_size: int | None = None) -> tf.data.Dataset:
        """Crea el dataset para un subconjunto de índices.

        Args:
            indices: índices de ``self.samples``.
            training: activa aumentación y shuffle.
            batch_size: por defecto ``train.batch_size`` del YAML.
        """
        bs = batch_size or int(self.cfg["train"]["batch_size"])
        s, m, k = self.size, self.max_eyes, NUM_LANDMARKS
        g = self.encoder.grid

        def load(idx: tf.Tensor) -> tuple[tf.Tensor, ...]:
            img, boxes, lms, mask, n = tf.numpy_function(
                self._np_load, [idx], [tf.uint8, tf.float32, tf.float32, tf.float32, tf.int32]
            )
            img.set_shape((s, s, 3))
            boxes.set_shape((m, 4))
            lms.set_shape((m, k, 2))
            mask.set_shape((m, k))
            n.set_shape(())
            return img, boxes, lms, mask, n

        def encode(img, boxes, lms, mask, n):  # type: ignore[no-untyped-def]
            x, y = tf.numpy_function(
                lambda *a: self._np_encode(*a, augment=training),
                [img, boxes, lms, mask, n],
                [tf.float32, tf.float32],
            )
            x.set_shape((s, s, 3))
            y.set_shape((g, g, TARGET_CHANNELS))
            return x, y

        ds = tf.data.Dataset.from_tensor_slices(np.asarray(indices, np.int64))
        ds = ds.map(load, num_parallel_calls=AUTOTUNE)
        if self.cfg["data"].get("cache", True):
            ds = ds.cache()  # imagen decodificada + letterbox (antes de aumentar)
        if training:
            ds = ds.shuffle(
                max(len(indices), 1), seed=int(self.cfg.get("seed", 42)), reshuffle_each_iteration=True
            )
        ds = ds.map(encode, num_parallel_calls=AUTOTUNE)
        return ds.batch(bs, drop_remainder=False).prefetch(AUTOTUNE)
