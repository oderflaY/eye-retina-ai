"""Generador de rostros sintéticos con ojos y landmarks exactos.

**Solo para pruebas del pipeline** (tests unitarios y *smoke tests* de
entrenamiento). Las métricas obtenidas con estos datos NO dicen nada sobre el
desempeño con imágenes reales.

Uso::

    python -m src.data.synthetic --out data/synthetic/eyes --n 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.eye_labels import LANDMARK_NAMES


def _draw_eye(
    img: np.ndarray, cx: float, cy: float, a: float, b: float, inner_dir: int, rng: np.random.Generator
) -> tuple[list[float], list[list[float]]]:
    """Dibuja un ojo y devuelve (caja xywh, 6 landmarks)."""
    cv2.ellipse(img, (int(cx), int(cy)), (int(a), int(b)), 0, 0, 360, (245, 245, 240), -1, cv2.LINE_AA)
    r = b * rng.uniform(0.75, 0.95)
    px = cx + rng.uniform(-0.3, 0.3) * (a - r)
    iris_color = tuple(int(v) for v in rng.integers(30, 160, 3))
    cv2.circle(img, (int(px), int(cy)), int(r), iris_color, -1, cv2.LINE_AA)
    cv2.circle(img, (int(px), int(cy)), max(1, int(r * 0.45)), (10, 10, 10), -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(cx), int(cy)), (int(a), int(b)), 0, 0, 360, (60, 40, 30), 1, cv2.LINE_AA)
    box = [cx - a, cy - b, 2 * a, 2 * b]
    lms = [
        [cx + inner_dir * a, cy],  # esquina interna (hacia la nariz)
        [cx - inner_dir * a, cy],  # esquina externa
        [cx, cy - b],  # párpado superior
        [cx, cy + b],  # párpado inferior
        [px, cy],  # pupila
        [px - inner_dir * r, cy],  # borde lateral del iris
    ]
    return box, lms


def make_face(
    rng: np.random.Generator, w: int, h: int, with_face: bool = True
) -> tuple[np.ndarray, list[tuple[list[float], list[list[float]]]]]:
    """Genera una imagen RGB con 0 o 1 rostro (2 ojos)."""
    bg = rng.integers(0, 255, 3)
    img = np.full((h, w, 3), bg, np.uint8)
    img = cv2.add(img, rng.integers(0, 40, (h, w, 3), dtype=np.uint8))
    eyes: list[tuple[list[float], list[list[float]]]] = []
    if not with_face:
        for _ in range(rng.integers(2, 6)):  # distractores
            c = tuple(int(v) for v in rng.integers(0, 255, 3))
            cv2.circle(
                img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(5, 40)), c, -1
            )
        return img, eyes
    fw = rng.uniform(0.35, 0.7) * min(w, h)
    fh = fw * rng.uniform(1.2, 1.4)
    fx = rng.uniform(fw / 2, w - fw / 2)
    fy = rng.uniform(min(fh / 2, h / 2), max(h - fh / 2, h / 2))
    skin = (int(rng.integers(150, 235)), int(rng.integers(110, 190)), int(rng.integers(80, 160)))
    cv2.ellipse(img, (int(fx), int(fy)), (int(fw / 2), int(fh / 2)), 0, 0, 360, skin, -1, cv2.LINE_AA)
    a = fw * rng.uniform(0.11, 0.15)
    b = a * rng.uniform(0.45, 0.6)
    dx, ey = fw * 0.22, fy - fh * 0.1
    eyes.append(_draw_eye(img, fx - dx, ey, a, b, inner_dir=+1, rng=rng))  # ojo a la izquierda
    eyes.append(_draw_eye(img, fx + dx, ey, a, b, inner_dir=-1, rng=rng))  # ojo a la derecha
    cv2.ellipse(
        img,
        (int(fx), int(fy + fh * 0.25)),
        (int(fw * 0.15), int(fh * 0.04)),
        0,
        0,
        180,
        (120, 40, 40),
        2,
        cv2.LINE_AA,
    )  # boca
    return img, eyes


def generate(out_dir: str | Path, n: int = 200, seed: int = 42, negative_ratio: float = 0.1) -> Path:
    """Crea ``n`` imágenes y un ``annotations.csv`` en ``out_dir``.

    Returns:
        Ruta del CSV generado.
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        w, h = int(rng.integers(256, 480)), int(rng.integers(220, 400))
        img, eyes = make_face(rng, w, h, with_face=rng.random() >= negative_ratio)
        name = f"images/synth_{i:04d}.png"
        cv2.imwrite(str(out_dir / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not eyes:
            rows.append({"image": name})
        for box, lms in eyes:
            row = {"image": name, "class": "eye", "x": box[0], "y": box[1], "w": box[2], "h": box[3]}
            for lname, (lx, ly) in zip(LANDMARK_NAMES, lms):
                row[f"{lname}_x"], row[f"{lname}_y"] = lx, ly
            rows.append(row)
    csv = out_dir / "annotations.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic/eyes")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(generate(args.out, args.n, args.seed))


if __name__ == "__main__":
    main()
