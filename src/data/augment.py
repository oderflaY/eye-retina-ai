"""Aumentación de datos con Albumentations para ojos y retinas.

* **Ojos**: rotación ±15°, flip horizontal, brillo/contraste, traslación y zoom.
  Las cajas y los landmarks se transforman junto con la imagen. Al hacer flip
  no se intercambian landmarks: cada punto es semántico *por ojo* (la esquina
  interna sigue siendo interna aunque el ojo quede del otro lado).
* **Retina**: recorte del disco, CLAHE, rotaciones libres (la retina no tiene
  orientación privilegiada), flips y ruido gaussiano.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
import numpy as np

NUM_LANDMARKS = 6
_TARGET_DIM = 4 + 3 * NUM_LANDMARKS + 2  # bbox, lms (12), máscaras (6), has_box, clase


def get_eye_augmentation_pipeline(
    rotate_limit: float = 15.0,
    hflip_p: float = 0.5,
    brightness_contrast: float = 0.2,
    translate: float = 0.05,
    zoom: float = 0.10,
    p: float = 0.8,
) -> A.Compose:
    """Pipeline de aumentación para ojos (imagen + puntos).

    Las cajas se tratan como 4 esquinas (keypoints): así una rotación produce
    la caja envolvente correcta y nunca se descarta un ojo.

    Args:
        rotate_limit: rotación máxima en grados (±).
        hflip_p: probabilidad de flip horizontal.
        brightness_contrast: límite de cambio de brillo y contraste.
        translate: traslación máxima como fracción del tamaño.
        zoom: zoom máximo (±).
        p: probabilidad de aplicar la transformación geométrica.

    Returns:
        ``A.Compose`` con ``keypoint_params`` en formato ``xy``.
    """
    return A.Compose(
        [
            A.Affine(
                rotate=(-rotate_limit, rotate_limit),
                translate_percent=(-translate, translate),
                scale=(1 - zoom, 1 + zoom),
                border_mode=cv2.BORDER_CONSTANT,
                p=p,
            ),
            A.HorizontalFlip(p=hflip_p),
            A.RandomBrightnessContrast(brightness_contrast, brightness_contrast, p=0.5),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_retina_augmentation_pipeline(size: int | None = None, train: bool = True) -> A.Compose:
    """Pipeline para fondos de ojo (imagen + máscara de vasos).

    Args:
        size: si se indica, redimensiona a ``size``×``size``.
        train: si es False solo aplica resize y CLAHE (validación/inferencia).

    Returns:
        ``A.Compose``; usar como ``pipeline(image=img, mask=mask)``.
    """
    ops: list[Any] = [A.Resize(size, size)] if size else []
    ops.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))
    if train:
        ops += [
            A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
            A.GaussNoise(p=0.3),
        ]
    return A.Compose(ops)


def _augment_one(img: np.ndarray, label: np.ndarray, pipeline: A.Compose) -> tuple[np.ndarray, np.ndarray]:
    """Aumenta una imagen y su vector de objetivos normalizados (24,)."""
    h, w = img.shape[:2]
    k = NUM_LANDMARKS
    label = label.copy()
    has_box = label[4 + 3 * k] > 0.5
    if not has_box:
        return pipeline(image=img, keypoints=[])["image"], label
    bx, by, bw, bh = label[:4] * [w, h, w, h]
    corners = [[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]]
    lms = (label[4 : 4 + 2 * k].reshape(k, 2) * [w, h]).tolist()
    out = pipeline(image=img, keypoints=corners + lms)
    pts = np.asarray(out["keypoints"], np.float32).reshape(-1, 2)
    c, lm = pts[:4], pts[4:]
    x0, y0 = np.clip(c.min(0), 0, [w, h])
    x1, y1 = np.clip(c.max(0), 0, [w, h])
    label[:4] = [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]
    label[4 : 4 + 2 * k] = (lm / [w, h]).reshape(-1)
    inside = (lm[:, 0] >= 0) & (lm[:, 0] < w) & (lm[:, 1] >= 0) & (lm[:, 1] < h)
    label[4 + 2 * k : 4 + 3 * k] *= inside.astype(np.float32)  # puntos fuera dejan de contar
    if x1 - x0 < 2 or y1 - y0 < 2:  # el ojo salió del cuadro: se vuelve negativo
        label[:] = 0.0
    return out["image"], label


def augment_batch(images: Any, labels: Any, pipeline: A.Compose) -> tuple[Any, Any]:
    """Aplica ``pipeline`` a un lote dentro de un grafo ``tf.data``.

    Args:
        images: tensor (B, H, W, 3) uint8.
        labels: tensor (B, 24) con bbox (4), landmarks (12), máscaras (6),
            has_box y clase, todo normalizado a [0, 1].
        pipeline: salida de :func:`get_eye_augmentation_pipeline`.

    Returns:
        (imágenes, etiquetas) aumentadas, con las mismas formas y tipos.
    """
    import tensorflow as tf

    def _np(imgs: np.ndarray, labs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out_i = np.empty_like(imgs)
        out_l = np.empty_like(labs)
        for i in range(len(imgs)):
            out_i[i], out_l[i] = _augment_one(imgs[i], labs[i], pipeline)
        return out_i, out_l.astype(np.float32)

    new_i, new_l = tf.numpy_function(_np, [images, labels], [tf.uint8, tf.float32])
    new_i.set_shape(images.shape)
    new_l.set_shape(labels.shape)
    return new_i, new_l


# --------------------------------------------------------------------------- #
# Utilidades usadas por el detector de rejilla y por el cargador de retina
# --------------------------------------------------------------------------- #
def build_eye_transform(aug_cfg: dict[str, Any]) -> A.Compose:
    """Transformación con ``bbox_params`` (detector de rejilla multi-ojo).

    Args:
        aug_cfg: sección ``augmentation`` del YAML.
    """
    zoom = float(aug_cfg.get("zoom", 0.1))
    translate = float(aug_cfg.get("translate", 0.05))
    bc = float(aug_cfg.get("brightness_contrast", 0.2))
    rot = float(aug_cfg.get("rotate_limit", 15))
    return A.Compose(
        [
            A.Affine(
                rotate=(-rot, rot),
                translate_percent=(-translate, translate),
                scale=(1 - zoom, 1 + zoom),
                border_mode=cv2.BORDER_CONSTANT,
                p=float(aug_cfg.get("p", 0.8)),
            ),
            A.HorizontalFlip(p=float(aug_cfg.get("hflip_p", 0.5))),
            A.RandomBrightnessContrast(brightness_limit=bc, contrast_limit=bc, p=0.5),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["box_ids"], min_visibility=0.4, clip=True),
        keypoint_params=A.KeypointParams(format="xy", label_fields=["kp_ids"], remove_invisible=False),
    )


def crop_retina_disc(img: np.ndarray, threshold: int = 10) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Recorta el fondo negro alrededor del disco de la retina.

    Returns:
        (imagen recortada, (y0, y1, x0, x1)) para aplicar el mismo recorte a la máscara.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > threshold)
    if ys.size == 0:
        return img, (0, img.shape[0], 0, img.shape[1])
    box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    return img[box[0] : box[1], box[2] : box[3]], box


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE (ecualización adaptativa del histograma) sobre el canal L de LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
