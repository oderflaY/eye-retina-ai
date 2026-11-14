"""``tf.data`` para la segmentación de vasos a partir de ``manifest.csv``.

* Entrenamiento: recorte aleatorio ``crop`` × ``crop`` de la imagen preprocesada,
  volteos, rotaciones de 90°, brillo/contraste/saturación/gamma.
* Validación y test: imagen completa (la U-Net es totalmente convolucional).

Las imágenes salen en float32 [0, 1] (H, W, 3); las máscaras en float32 {0, 1}
(H, W, 1). Las PNG se leen desde disco en cada época (no caben en 7 GB de RAM a
1024 px).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def read_manifest(root: str | Path, splits: str | list[str] | None = None) -> pd.DataFrame:
    """Lee el manifiesto y resuelve las rutas absolutas."""
    root = Path(root)
    df = pd.read_csv(root / "manifest.csv")
    for col in ("image", "vessel", "fov"):
        df[col] = df[col].map(lambda p: str(root / p))
    if splits is not None:
        splits = [splits] if isinstance(splits, str) else splits
        df = df[df["split"].isin(splits)].reset_index(drop=True)
    return df


def _load(image_path: tf.Tensor, vessel_path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    img = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
    ves = tf.io.decode_png(tf.io.read_file(vessel_path), channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    ves = tf.cast(ves > 127, tf.float32)
    return img, ves


def augment(img: tf.Tensor, ves: tf.Tensor, crop: int, color: float = 0.15) -> tuple[tf.Tensor, tf.Tensor]:
    """Recorte + geometría + color, igual para imagen y máscara."""
    both = tf.concat([img, ves], axis=-1)
    both = tf.image.random_crop(both, [crop, crop, 4])
    both = tf.image.random_flip_left_right(both)
    both = tf.image.random_flip_up_down(both)
    both = tf.image.rot90(both, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img, ves = both[..., :3], both[..., 3:]
    if color > 0:
        fov = tf.cast(tf.reduce_max(img, axis=-1, keepdims=True) > 0.02, tf.float32)
        img = tf.image.random_brightness(img, color)
        img = tf.image.random_contrast(img, 1 - color, 1 + color)
        img = tf.image.random_saturation(img, 1 - color, 1 + color)
        gamma = tf.random.uniform([], 1 - color, 1 + color)
        img = tf.clip_by_value(img, 0.0, 1.0) ** gamma
        img = img * fov  # el fondo sigue negro
    return img, ves


def make_dataset(
    df: pd.DataFrame,
    batch_size: int,
    training: bool,
    crop: int = 512,
    crops_per_image: int = 1,
    color: float = 0.15,
    seed: int = 42,
) -> tf.data.Dataset:
    """Construye el pipeline.

    Args:
        df: filas del manifiesto (``read_manifest``).
        batch_size: tamaño de lote.
        training: activa barajado, recortes y aumentación.
        crop: lado del recorte de entrenamiento (múltiplo de 32).
        crops_per_image: recortes por imagen y época (repite la lista).
        color: intensidad de la aumentación de color (0 = sin color).
        seed: semilla del barajado.
    """
    ds = tf.data.Dataset.from_tensor_slices((df["image"].tolist(), df["vessel"].tolist()))
    if training:
        if crops_per_image > 1:
            ds = ds.repeat(crops_per_image)
        ds = ds.shuffle(len(df) * crops_per_image, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda i, v: augment(i, v, crop, color), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
    else:
        ds = ds.batch(batch_size)
    return ds.prefetch(AUTOTUNE)
