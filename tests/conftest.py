"""Fixtures compartidas: datasets sintéticos pequeños (no se descarga nada)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_eyes(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Carpeta con 30 rostros sintéticos (2 ojos) y ~10 % negativos en formato CSV."""
    from src.data.synthetic import generate

    out = tmp_path_factory.mktemp("eyes")
    generate(out, n=30, seed=3)
    return out


@pytest.fixture(scope="session")
def folder_eyes(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Formato de carpetas por clase: ``eye/`` y ``no_eye/`` (una imagen en gris)."""
    import cv2
    import numpy as np

    out = tmp_path_factory.mktemp("folders")
    rng = np.random.default_rng(0)
    for cls in ("eye", "no_eye"):
        (out / cls).mkdir()
        for i in range(6):
            img = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)
            cv2.imwrite(str(out / cls / f"{i}.png"), img)
    cv2.imwrite(str(out / "no_eye" / "gris.png"), rng.integers(0, 255, (40, 60), dtype=np.uint8))
    return out
