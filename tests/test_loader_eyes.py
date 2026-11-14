"""Pruebas de ``EyeDataLoader``: metadatos, división sin fuga, shapes y tipos de tf.data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.data.augment import augment_batch, get_eye_augmentation_pipeline  # noqa: E402
from src.data.loader_eyes import EyeDataLoader  # noqa: E402


@pytest.fixture(scope="module")
def loader(synthetic_eyes: Path) -> EyeDataLoader:
    return EyeDataLoader(synthetic_eyes, (96, 96), batch_size=4, seed=42)


def test_metadata_rows(loader: EyeDataLoader) -> None:
    meta = loader.load_metadata()
    # cada rostro sintético tiene 2 ojos -> 2 filas recortadas; los negativos, 1 fila
    pos_imgs = meta[meta.n_eyes == 2].image_path.nunique()
    assert (meta.n_eyes == 2).sum() == 2 * pos_imgs
    assert set(meta.eye_prob.unique()) <= {0.0, 1.0}
    pos = meta[meta.has_box == 1]
    # el ojo queda completo dentro de su recorte (coordenadas normalizadas)
    assert (pos.bx >= -1e-6).all() and (pos.bx + pos.bw <= 1 + 1e-6).all()
    assert (pos[[f"lm{i}_m" for i in range(6)]] == 1).all().all()


def test_split_no_leakage(loader: EyeDataLoader) -> None:
    meta = loader.load_metadata()
    tr, va, te = loader.split(meta)
    sets = [set(d.image_path) for d in (tr, va, te)]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
    assert len(tr) + len(va) + len(te) == len(meta)


def test_tfdataset_shapes_and_types(loader: EyeDataLoader) -> None:
    meta = loader.load_metadata()
    ds = loader.build_tfdataset(meta, augment=True, shuffle=True)
    x, y, sw = next(iter(ds))
    assert x.shape == (4, 96, 96, 3) and x.dtype == tf.float32
    assert float(tf.reduce_min(x)) >= 0.0 and float(tf.reduce_max(x)) <= 1.0
    assert y["bbox"].shape == (4, 4) and y["landmarks"].shape == (4, 18) and y["eye_prob"].shape == (4, 1)
    assert set(sw) == {"bbox", "landmarks", "eye_prob"}


def test_augment_keeps_box_consistent() -> None:
    img = np.zeros((1, 64, 64, 3), np.uint8)
    lab = np.zeros((1, 24), np.float32)
    lab[0, :4] = [0.4, 0.4, 0.2, 0.1]
    lab[0, 4:16] = np.tile([0.5, 0.45], 6)
    lab[0, 16:22] = 1
    lab[0, 22:24] = 1
    out_i, out_l = augment_batch(tf.constant(img), tf.constant(lab), get_eye_augmentation_pipeline(p=1.0))
    b = out_l.numpy()[0]
    assert out_i.shape == img.shape and b[2] > 0 and b[3] > 0
    assert 0 <= b[0] <= 1 and b[0] + b[2] <= 1 + 1e-5


def test_folder_format_and_grayscale(folder_eyes: Path) -> None:
    ld = EyeDataLoader(folder_eyes, (32, 32), batch_size=20)
    meta = ld.load_metadata()
    assert meta.eye_prob.sum() == 6 and (meta.has_box == 0).all()
    x, y, sw = next(iter(ld.build_tfdataset(meta)))
    assert x.shape[-1] == 3  # la imagen en gris se convierte a RGB
    assert float(tf.reduce_sum(sw["bbox"])) == 0.0
