"""Entrena el analizador de ojo externo (4 clases: normal, enrojecimiento, pterigión, catarata).

    python -m src.training.train_ojo_externo [--epocas 30] [--batch 32]

EfficientNetV2-B0 (ImageNet) → calentamiento con la base congelada → ajuste fino completo
(BatchNorm congelado). Pérdida con pesos por clase (hay pocas cataratas). Se guarda el mejor por
exactitud balanceada en validación y se evalúa en test (AUC por clase, sensibilidad,
especificidad, matriz de confusión) → ``models/ojo_externo/`` y ``reports/ojo_externo/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.config import PROJECT_ROOT
from src.data.prepare_ojo_externo import CLASES

DATA = PROJECT_ROOT / "data/processed/ojo_externo"
OUT = PROJECT_ROOT / "models/ojo_externo"
REP = PROJECT_ROOT / "reports/ojo_externo"
TAM = 256
BASES = {
    "b0": tf.keras.applications.EfficientNetV2B0,
    "b2": tf.keras.applications.EfficientNetV2B2,
    "s": tf.keras.applications.EfficientNetV2S,
}
AUTOTUNE = tf.data.AUTOTUNE


def cargar(path: tf.Tensor) -> tf.Tensor:
    return tf.io.decode_jpeg(tf.io.read_file(path), channels=3)


def aumentar(img: tf.Tensor) -> tf.Tensor:
    """Variaciones de una foto de teléfono: encuadre, giro, espejo, luz. El tono casi no se toca
    (el color rojo/blanco del ojo es justo lo que distingue las clases)."""
    img = tf.image.convert_image_dtype(img, tf.float32)
    lado = tf.cast(tf.shape(img)[0], tf.float32)
    frac = tf.random.uniform([], 0.70, 1.0)
    c = tf.cast(lado * frac, tf.int32)
    img = tf.image.random_crop(img, tf.stack([c, c, 3]))
    img = tf.image.resize(img, (TAM, TAM))
    img = tf.image.random_flip_left_right(img)
    k = tf.random.uniform([], 0, 4, tf.int32)
    img = tf.cond(tf.random.uniform([]) < 0.15, lambda: tf.image.rot90(img, k), lambda: img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.85, 1.15)
    img = tf.image.random_hue(img, 0.02)
    return tf.clip_by_value(img, 0.0, 1.0) * 255.0


def evaluar_img(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.resize(tf.image.convert_image_dtype(img, tf.float32), (TAM, TAM))
    return img * 255.0


def dataset(df: pd.DataFrame, batch: int, entrenar: bool) -> tf.data.Dataset:
    paths = [str(DATA / p) for p in df.image]
    y = df.clase.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, tf.one_hot(y, len(CLASES))))
    if entrenar:
        ds = ds.shuffle(len(df), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, t: (cargar(p), t), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda i, t: ((aumentar(i) if entrenar else evaluar_img(i)), t), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


def modelo(n: int, red: str = "b0") -> tuple[tf.keras.Model, tf.keras.Model]:
    base = BASES[red](include_top=False, weights="imagenet", input_shape=(None, None, 3))
    x_in = tf.keras.Input((None, None, 3), name="imagen")
    x = base(x_in, training=False)  # BatchNorm siempre en modo inferencia
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n, dtype="float32", name="logits")(x)
    return tf.keras.Model(x_in, out, name=f"ojo_externo_effv2{red}"), base


class ExactitudBalanceada(tf.keras.callbacks.Callback):
    def __init__(self, ds: tf.data.Dataset, y: np.ndarray) -> None:
        super().__init__()
        self.ds, self.y = ds, y

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        p = self.model.predict(self.ds, verbose=0).argmax(1)
        rec = [np.mean(p[self.y == k] == k) for k in range(len(CLASES)) if (self.y == k).any()]
        logs["val_bal_acc"] = float(np.mean(rec))
        print(f" — val_bal_acc {logs['val_bal_acc']:.4f} " + " ".join(f"{c[:5]}={r:.2f}" for c, r in zip(CLASES, rec)))


def probabilidades(m: tf.keras.Model, ds: tf.data.Dataset) -> np.ndarray:
    """Promedio con la imagen en espejo (TTA)."""
    a = m.predict(ds, verbose=0)
    b = m.predict(ds.map(lambda x, y: (tf.image.flip_left_right(x), y)), verbose=0)
    return tf.nn.softmax((a + b) / 2).numpy()


def metricas(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(len(CLASES)))
    por = {}
    for k, c in enumerate(CLASES):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        por[c] = {
            "n": int(cm[k].sum()),
            "auc": round(float(roc_auc_score(y == k, p[:, k])), 4),
            "sensibilidad": round(float(tp / max(tp + fn, 1)), 4),
            "especificidad": round(float(tn / max(tn + fp, 1)), 4),
            "precision": round(float(tp / max(tp + fp, 1)), 4),
        }
    return {
        "exactitud": round(float((pred == y).mean()), 4),
        "exactitud_balanceada": round(float(np.mean([por[c]["sensibilidad"] for c in CLASES])), 4),
        "auc_macro": round(float(np.mean([por[c]["auc"] for c in CLASES])), 4),
        "por_clase": por,
        "matriz_confusion": {"filas=real, columnas=predicho": list(CLASES), "valores": cm.tolist()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--calentamiento", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_fino", type=float, default=1e-4)
    ap.add_argument("--red", choices=list(BASES), default="b0")
    ap.add_argument("--tam", type=int, default=256)
    ap.add_argument("--suavizado", type=float, default=0.0, help="label smoothing")
    ap.add_argument("--xla", action="store_true", help="compilar con XLA (más rápido en GPU)")
    ap.add_argument("--salida", default="models/ojo_externo")
    ap.add_argument("--reporte", default="reports/ojo_externo")
    a = ap.parse_args()
    global TAM, OUT, REP
    TAM, OUT, REP = a.tam, PROJECT_ROOT / a.salida, PROJECT_ROOT / a.reporte

    tf.keras.utils.set_random_seed(42)
    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "manifest.csv")
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    y_va = va.clase.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
    y_te = te.clase.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
    cuenta = tr.clase.value_counts()
    pesos = {i: float(len(tr) / (len(CLASES) * cuenta[c])) for i, c in enumerate(CLASES)}
    print("pesos por clase", {CLASES[i]: round(w, 2) for i, w in pesos.items()})

    ds_tr = dataset(tr, a.batch, True)
    ds_va = dataset(va, a.batch, False)
    ds_te = dataset(te, a.batch, False)
    m, base = modelo(len(CLASES), a.red)
    perdida = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=a.suavizado)
    ck = OUT / "best.keras"
    cbs = lambda: [  # noqa: E731
        ExactitudBalanceada(ds_va, y_va),
        tf.keras.callbacks.ModelCheckpoint(ck, monitor="val_bal_acc", mode="max", save_best_only=True),
        tf.keras.callbacks.CSVLogger(REP / "historial.csv", append=True),
    ]
    t0 = time.time()
    base.trainable = False
    m.compile(tf.keras.optimizers.Adam(a.lr), perdida, metrics=["accuracy"], jit_compile=a.xla)
    m.fit(ds_tr, validation_data=ds_va, epochs=a.calentamiento, class_weight=pesos, callbacks=cbs(), verbose=2)

    base.trainable = True
    pasos = a.epocas * len(ds_tr)
    lr = tf.keras.optimizers.schedules.CosineDecay(a.lr_fino, pasos, alpha=0.02)
    m.compile(tf.keras.optimizers.AdamW(lr, weight_decay=1e-4), perdida, metrics=["accuracy"], jit_compile=a.xla)
    para = tf.keras.callbacks.EarlyStopping(monitor="val_bal_acc", mode="max", patience=8)
    m.fit(ds_tr, validation_data=ds_va, initial_epoch=a.calentamiento, epochs=a.calentamiento + a.epocas,
          class_weight=pesos, callbacks=cbs() + [para], verbose=2)

    mejor = tf.keras.models.load_model(ck)
    res = {
        "modelo": f"EfficientNetV2-{a.red.upper()} (ImageNet) ajustado, {a.tam} px, suavizado {a.suavizado}",
        "clases": list(CLASES),
        "tam_entrada": TAM,
        "minutos": round((time.time() - t0) / 60, 1),
        "val": metricas(y_va, probabilidades(mejor, ds_va)),
        "test": metricas(y_te, probabilidades(mejor, ds_te)),
        "datos": json.loads((DATA / "resumen.json").read_text()),
    }
    (REP / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    (OUT / "config.json").write_text(json.dumps(
        {"clases": list(CLASES), "tam_entrada": TAM, "test": res["test"]}, indent=2, ensure_ascii=False))
    print(json.dumps(res["test"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
