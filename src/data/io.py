"""Lectura de imágenes desde disco (JPG, PNG, TIFF de alta resolución, GIF).

Todas las funciones devuelven imágenes RGB ``uint8`` con forma (H, W, 3).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".ppm"})


def list_images(root: str | Path) -> list[Path]:
    """Lista recursivamente las imágenes bajo ``root`` (orden estable)."""
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convierte imágenes de 16 bits o flotantes a ``uint8``."""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        # Los TIFF de retina suelen venir en 16 bits: 65535 -> 255
        return (img.astype(np.float32) / 257.0).round().astype(np.uint8)
    img = img.astype(np.float32)
    lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    if hi - lo < 1e-8:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - lo) / (hi - lo) * 255.0).round().astype(np.uint8)


def read_image_rgb(path: str | Path) -> np.ndarray:
    """Lee una imagen y la devuelve en RGB ``uint8``.

    Maneja escala de grises, canal alfa, TIFF de 16 bits y GIF (vía Pillow,
    porque OpenCV no siempre lo soporta).

    Raises:
        FileNotFoundError: si la imagen no existe o no se puede decodificar.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            from PIL import Image

            with Image.open(path) as pil:
                img = np.asarray(pil.convert("RGB"))
            return img
        except Exception as exc:  # noqa: BLE001 - se re-lanza con contexto
            raise FileNotFoundError(f"No se pudo leer la imagen: {path}") from exc
    img = _to_uint8(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decodifica bytes (p. ej. de una petición HTTP) a RGB ``uint8``."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Los bytes recibidos no son una imagen válida")
    img = _to_uint8(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_size(path: str | Path) -> tuple[int, int]:
    """Devuelve (ancho, alto) leyendo solo la cabecera cuando es posible."""
    try:
        from PIL import Image

        with Image.open(path) as pil:
            return int(pil.size[0]), int(pil.size[1])
    except Exception:  # noqa: BLE001 - recurre a lectura completa
        h, w = read_image_rgb(path).shape[:2]
        return w, h


def letterbox(img: np.ndarray, size: int, fill: int = 0) -> tuple[np.ndarray, float, int, int]:
    """Redimensiona conservando la proporción y rellena hasta ``size``x``size``.

    Returns:
        (imagen, escala, pad_x, pad_y). Un punto original (x, y) queda en
        ``(x * escala + pad_x, y * escala + pad_y)``.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.full((size, size, 3), fill, dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return out, scale, pad_x, pad_y
