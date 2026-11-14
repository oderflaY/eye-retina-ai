"""Preprocesado de fondos de ojo, compartido por ``prepare_retina`` y la inferencia.

Pasos (``preprocess_fundus``):

1. Máscara FOV (campo de visión): se usa la del dataset si existe; si no, se
   estima umbralizando el canal rojo y quedándose con el componente más grande.
2. Recorte al rectángulo del FOV y relleno a cuadrado (centrado).
3. Redimensionado a ``size`` × ``size`` (INTER_AREA).
4. CLAHE opcional sobre el canal L de LAB, solo dentro del FOV.

Las máscaras de vasos se reducen con INTER_AREA y se umbralizan en 0.5: así
los vasos de 1 px no desaparecen como pasaría con vecino más cercano.

``RetinaGeometry`` guarda el recorte y la escala para devolver la predicción a
la resolución de la imagen original (``restore_to_original``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class RetinaGeometry:
    """Transformación imagen original → imagen cuadrada preprocesada."""

    orig_h: int
    orig_w: int
    x0: int  # recorte del FOV en la imagen original
    y0: int
    crop_w: int
    crop_h: int
    pad_x: int  # relleno añadido para hacer el recorte cuadrado
    pad_y: int
    side: int  # lado del cuadrado antes de redimensionar
    size: int  # lado final

    def to_dict(self) -> dict[str, int]:
        """Diccionario serializable."""
        return asdict(self)


def estimate_fov_mask(image: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Estima el campo de visión circular de un fondo de ojo.

    Args:
        image: RGB uint8 (H, W, 3).
        threshold: nivel mínimo del canal rojo considerado retina.

    Returns:
        Máscara uint8 {0, 1} (H, W).
    """
    red = image[..., 0]
    mask = (red > threshold).astype(np.uint8)
    k = max(3, (min(image.shape[:2]) // 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:  # imagen sin bordes negros: todo es FOV
        return np.ones(image.shape[:2], dtype=np.uint8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == biggest).astype(np.uint8)
    # rellena huecos internos (zonas muy oscuras dentro de la retina)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def binarize(mask: np.ndarray) -> np.ndarray:
    """Convierte una máscara (gris, RGB o {0,255}) a uint8 {0, 1}."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > (127 if mask.max() > 1 else 0)).astype(np.uint8)


def apply_clahe(image: np.ndarray, fov: np.ndarray, clip_limit: float = 2.0, tiles: int = 8) -> np.ndarray:
    """CLAHE en el canal L de LAB; fuera del FOV queda negro."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles))
    lab[..., 0] = clahe.apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return out * fov[..., None]


def _crop_square(arr: np.ndarray, g: RetinaGeometry) -> np.ndarray:
    crop = arr[g.y0 : g.y0 + g.crop_h, g.x0 : g.x0 + g.crop_w]
    pad = [(g.pad_y, g.side - g.crop_h - g.pad_y), (g.pad_x, g.side - g.crop_w - g.pad_x)]
    if arr.ndim == 3:
        pad.append((0, 0))
    return np.pad(crop, pad)


def preprocess_fundus(
    image: np.ndarray,
    size: int = 1024,
    fov: np.ndarray | None = None,
    vessels: np.ndarray | None = None,
    clahe: bool = True,
    margin: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, RetinaGeometry]:
    """Recorta al FOV, hace cuadrado, redimensiona y (opcional) aplica CLAHE.

    Args:
        image: RGB uint8 (H, W, 3).
        size: lado de salida.
        fov: máscara FOV del dataset; ``None`` = estimarla.
        vessels: máscara de vasos opcional (misma resolución que ``image``).
        clahe: aplica CLAHE.
        margin: margen relativo alrededor del FOV.

    Returns:
        ``(imagen, fov, vasos | None, geometría)``; imagen uint8 (size, size, 3),
        máscaras uint8 {0, 1} (size, size).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Se esperaba RGB (H, W, 3), llegó {image.shape}")
    h, w = image.shape[:2]
    fov = estimate_fov_mask(image) if fov is None else binarize(fov)
    ys, xs = np.nonzero(fov)
    if len(xs) == 0:
        fov = np.ones((h, w), np.uint8)
        ys, xs = np.nonzero(fov)
    m = int(round(margin * max(h, w)))
    x0, x1 = max(0, xs.min() - m), min(w, xs.max() + 1 + m)
    y0, y1 = max(0, ys.min() - m), min(h, ys.max() + 1 + m)
    cw, ch = x1 - x0, y1 - y0
    side = max(cw, ch)
    g = RetinaGeometry(h, w, int(x0), int(y0), int(cw), int(ch), (side - cw) // 2, (side - ch) // 2, side, size)

    img = cv2.resize(_crop_square(image, g), (size, size), interpolation=cv2.INTER_AREA)
    fov_s = cv2.resize(_crop_square(fov, g), (size, size), interpolation=cv2.INTER_NEAREST)
    img = img * fov_s[..., None]
    if clahe:
        img = apply_clahe(img, fov_s)
    ves_s = None
    if vessels is not None:
        v = _crop_square(binarize(vessels).astype(np.float32), g)
        ves_s = (cv2.resize(v, (size, size), interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)
        ves_s *= fov_s
    return img, fov_s, ves_s, g


def restore_to_original(prob: np.ndarray, g: RetinaGeometry) -> np.ndarray:
    """Devuelve un mapa (size, size) float a la resolución original (H, W)."""
    sq = cv2.resize(prob.astype(np.float32), (g.side, g.side), interpolation=cv2.INTER_LINEAR)
    crop = sq[g.pad_y : g.pad_y + g.crop_h, g.pad_x : g.pad_x + g.crop_w]
    out = np.zeros((g.orig_h, g.orig_w), np.float32)
    out[g.y0 : g.y0 + g.crop_h, g.x0 : g.x0 + g.crop_w] = crop
    return out
