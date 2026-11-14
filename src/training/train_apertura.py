"""Clasificador de apertura del ojo para la app (¿abierto o cerrado/entrecerrado?).

El detector de ojo de cerca ubica bien el ojo, pero sus puntos de párpado no siguen la apertura
real (predice ~0.29 en ojos abiertos y cerrados por igual: los cerrados son solo el 5 % y pesan
poco en la pérdida de los puntos). Este modelo aparte, chico, aprende directo:

* ``apertura``: párpado a párpado / esquina a esquina, medida por MediaPipe (regresión).
* ``cerrado``: apertura real < 0.18 (clasificación, con peso por clase).

Entrada: el recorte que da el detector (cuadrado alrededor del ojo), 128×128 RGB en [0, 1].
Datos: data/raw/eyes_cerca_v2 (FFHQ etiquetado con src/data/etiquetar_ffhq.py). División por
cara (los recortes de una misma cara caen juntos).

    python -m src.training.train_apertura
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LADO = 128
UMBRAL_CERRADO = 0.18


def cargar(datos: Path) -> pd.DataFrame:
    d = pd.read_csv(datos / "annotations.csv")
    d = d[d.x.notna()].copy()
    d["apertura"] = np.hypot(d.lower_lid_x - d.upper_lid_x, d.lower_lid_y - d.upper_lid_y) / np.hypot(
        d.outer_corner_x - d.inner_corner_x, d.outer_corner_y - d.inner_corner_y).clip(lower=1e-3)
    d["apertura"] = d.apertura.clip(0, 0.6)
    d["cerrado"] = (d.apertura < UMBRAL_CERRADO).astype(np.float32)
    d["ruta"] = [str(datos / p) for p in d.image]
    d["grupo"] = d.image.str.extract(r"images/(ffhq\d+)_")[0]
    return d.reset_index(drop=True)


def dividir(d: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    g1 = GroupShuffleSplit(1, test_size=0.15, random_state=seed)
    tr, te = next(g1.split(d, groups=d.grupo))
    resto, test = d.iloc[tr], d.iloc[te]
    g2 = GroupShuffleSplit(1, test_size=0.15, random_state=seed + 1)
    tr, va = next(g2.split(resto, groups=resto.grupo))
    return resto.iloc[tr], resto.iloc[va], test


def dataset(d: pd.DataFrame, lote: int, entrenar: bool, peso_cerrado: float) -> tf.data.Dataset:
    pesos = np.where(d.cerrado.values > 0, peso_cerrado, 1.0).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices(
        (d.ruta.values, d.apertura.values.astype(np.float32), d.cerrado.values, pesos))
    if entrenar:
        ds = ds.shuffle(len(d), reshuffle_each_iteration=True)

    def leer(ruta, ap, ce, w):
        img = tf.image.decode_jpeg(tf.io.read_file(ruta), channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        if entrenar:
            # zoom/traslación moderados: el recorte del detector no siempre queda centrado
            lado = tf.cast(tf.shape(img)[0], tf.float32)
            f = tf.random.uniform([], 0.8, 1.0)
            l = tf.cast(lado * f, tf.int32)
            img = tf.image.random_crop(img, tf.stack([l, l, 3]))
            img = tf.image.random_flip_left_right(img)  # la apertura no cambia con el espejo
            img = tf.image.random_brightness(img, 0.25)
            img = tf.image.random_contrast(img, 0.7, 1.3)
            img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.clip_by_value(tf.image.resize(img, (LADO, LADO)), 0, 1)
        return img, {"apertura": ap, "cerrado": ce}, {"apertura": w, "cerrado": w}

    return ds.map(leer, num_parallel_calls=4).batch(lote).prefetch(2)


def modelo(alpha: float = 0.5) -> tf.keras.Model:
    ent = tf.keras.Input((LADO, LADO, 3), name="imagen")
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(ent)  # MobileNetV2 espera [-1, 1]
    base = tf.keras.applications.MobileNetV2((LADO, LADO, 3), alpha=alpha, include_top=False, weights="imagenet")
    x = base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    ap = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    ap = tf.keras.layers.Rescaling(0.6, name="apertura")(ap)
    ce = tf.keras.layers.Dense(1, activation="sigmoid", name="cerrado")(x)
    return tf.keras.Model(ent, {"apertura": ap, "cerrado": ce})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=PROJECT_ROOT / "data/raw/eyes_cerca_v2")
    ap.add_argument("--salida", type=Path, default=PROJECT_ROOT / "models/apertura")
    ap.add_argument("--reporte", type=Path, default=PROJECT_ROOT / "reports/apertura")
    ap.add_argument("--epocas", type=int, default=25)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--paciencia", type=int, default=6)
    a = ap.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)
    a.reporte.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    d = cargar(a.datos)
    tr, va, te = dividir(d, 42)
    peso = float((tr.cerrado == 0).sum() / max((tr.cerrado == 1).sum(), 1)) ** 0.5
    print(f"train {len(tr)} ({int(tr.cerrado.sum())} cerrados) · val {len(va)} · test {len(te)} · peso cerrado {peso:.1f}")

    m = modelo(a.alpha)
    perdidas = {"apertura": tf.keras.losses.Huber(delta=0.05), "cerrado": "binary_crossentropy"}
    pesos_p = {"apertura": 20.0, "cerrado": 1.0}
    ds_tr = dataset(tr, a.lote, True, peso)
    ds_va = dataset(va, a.lote, False, peso)
    mejor = a.salida / "best.keras"

    # 1) cabezas con la base congelada; 2) ajuste fino de toda la red
    m.layers[2].trainable = False
    m.compile(tf.keras.optimizers.Adam(1e-3), perdidas, loss_weights=pesos_p,
              metrics={"cerrado": [tf.keras.metrics.AUC(name="auc")], "apertura": ["mae"]})
    m.fit(ds_tr, validation_data=ds_va, epochs=3, verbose=2)
    m.layers[2].trainable = True
    m.compile(tf.keras.optimizers.Adam(3e-4), perdidas, loss_weights=pesos_p,
              metrics={"cerrado": [tf.keras.metrics.AUC(name="auc")], "apertura": ["mae"]})
    m.fit(ds_tr, validation_data=ds_va, epochs=a.epocas, verbose=2, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(mejor), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.paciencia, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, a.paciencia // 3), min_lr=1e-6),
    ])

    # Evaluación en las caras de prueba
    p = m.predict(dataset(te, a.lote, False, 1.0), verbose=0)
    ap_pred, ce_pred = p["apertura"].ravel(), p["cerrado"].ravel()
    real_ce = te.cerrado.values
    res: dict = {
        "n_test": int(len(te)), "n_cerrados_test": int(real_ce.sum()),
        "auc_cerrado_prob": round(float(roc_auc_score(real_ce, ce_pred)), 4),
        "auc_cerrado_por_apertura": round(float(roc_auc_score(real_ce, -ap_pred)), 4),
        "mae_apertura": round(float(np.abs(ap_pred - te.apertura.values).mean()), 4),
        "corr_apertura": round(float(np.corrcoef(ap_pred, te.apertura.values)[0, 1]), 4),
        "umbral_cerrado": UMBRAL_CERRADO,
        "por_umbral_prob": {},
    }
    for u in (0.3, 0.5, 0.7):
        dice = ce_pred >= u
        res["por_umbral_prob"][str(u)] = {
            "cerrados_detectados": round(float(dice[real_ce == 1].mean()), 4),
            "abiertos_bien": round(float((~dice)[real_ce == 0].mean()), 4),
        }
    print(json.dumps(res, indent=2, ensure_ascii=False))
    (a.reporte / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    # TFLite para la app (float32, sin cuantizar: 1–2 MB)
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    datos = conv.convert()
    destino = a.salida / "ojo_apertura.tflite"
    destino.write_bytes(datos)
    print(f"TFLite {len(datos) / 1e6:.1f} MB → {destino}")


if __name__ == "__main__":
    main()
