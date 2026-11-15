"""Segmentador de dientes (U-Net ResNet50) para encontrar la ENCÍA por exclusión.

Encía = boca abierta (labios internos, MediaPipe) − dientes (este modelo) − sombra.
Datos: Teeth Segmentation Intraoral / Dentalai (MIT), https://huggingface.co/datasets/Frallex/Teeth_Segmentation_Intraoral
2,495 fotos con polígonos de diente/caries/cavidad/grieta (todo cuenta como diente), reducidas en el
VPS con src/data/reducir_encias_vps.py → ~/TemrariosSpace/datasets/Encias/reducido/dientes/{train,valid,test}.

    python -m src.training.train_dientes
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.models.unet import build_unet

DATOS = Path.home() / "TemrariosSpace/datasets/Encias/reducido/dientes"
SALIDA = Path("models/dientes")
LADO = 256
LOTE = 8
EPOCAS = 30


def lista(split: str) -> list[tuple[str, str]]:
    return [(str(j), str(j.with_suffix(".png"))) for j in sorted((DATOS / split).glob("*.jpg")) if j.with_suffix(".png").exists()]


def cargar(img_p, mask_p):
    img = tf.image.decode_jpeg(tf.io.read_file(img_p), channels=3)
    m = tf.image.decode_png(tf.io.read_file(mask_p), channels=1)
    img = tf.image.resize(tf.image.convert_image_dtype(img, tf.float32), (LADO, LADO))
    m = tf.cast(tf.image.resize(m, (LADO, LADO), method="nearest") > 127, tf.float32)
    return img, m


def aumentar(img, m):
    if tf.random.uniform(()) < 0.5:
        img, m = tf.image.flip_left_right(img), tf.image.flip_left_right(m)
    if tf.random.uniform(()) < 0.3:
        img, m = tf.image.flip_up_down(img), tf.image.flip_up_down(m)
    # luz y color de cámaras de teléfono distintas (sin tocar el tono: la encía no debe volverse diente)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    return tf.clip_by_value(img, 0, 1), m


def ds(split: str, entrenar: bool) -> tf.data.Dataset:
    pares = lista(split)
    d = tf.data.Dataset.from_tensor_slices(([a for a, _ in pares], [b for _, b in pares]))
    if entrenar:
        d = d.shuffle(len(pares), reshuffle_each_iteration=True)
    d = d.map(cargar, num_parallel_calls=2)
    if entrenar:
        d = d.map(aumentar, num_parallel_calls=2)
    return d.batch(LOTE).prefetch(2)


def perdida(y, logits):
    bce = tf.keras.losses.binary_crossentropy(y, logits, from_logits=True)
    p = tf.sigmoid(logits)
    inter = tf.reduce_sum(p * y, [1, 2, 3])
    dice = 1 - (2 * inter + 1) / (tf.reduce_sum(p, [1, 2, 3]) + tf.reduce_sum(y, [1, 2, 3]) + 1)
    return tf.reduce_mean(bce) + tf.reduce_mean(dice)


def dice_np(y, p):
    return float((2 * (y * p).sum() + 1) / (y.sum() + p.sum() + 1))


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(0)
    n = {s: len(lista(s)) for s in ("train", "valid", "test")}
    print("imágenes", n)
    modelo = build_unet("resnet50", "imagenet", (LADO, LADO, 3), decoder_filters=(256, 128, 64, 32, 16))
    modelo.compile(tf.keras.optimizers.Adam(3e-4), loss=perdida)
    modelo.fit(
        ds("train", True), validation_data=ds("valid", False), epochs=EPOCAS, verbose=2,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(SALIDA / "best.keras"), save_best_only=True, monitor="val_loss"),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
            tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True, monitor="val_loss"),
        ],
    )
    mejor = tf.keras.models.load_model(SALIDA / "best.keras", compile=False)
    dices = []
    for img, m in ds("test", False):
        p = (tf.sigmoid(mejor(img, training=False)).numpy() > 0.5).astype(np.float32)
        dices += [dice_np(m.numpy()[i], p[i]) for i in range(len(p))]
    res = {"n": n, "lado": LADO, "dice_test_media": round(float(np.mean(dices)), 4),
           "dice_test_mediana": round(float(np.median(dices)), 4),
           "datos": "Dentalai / Frallex Teeth_Segmentation_Intraoral (MIT)"}
    (SALIDA / "config.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
