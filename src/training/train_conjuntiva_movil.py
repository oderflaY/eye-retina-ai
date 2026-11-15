"""Segmentador de conjuntiva palpebral CHICO para correr en el teléfono (anemia sin internet).

    python -m src.training.train_conjuntiva_movil pseudo     # máscaras del U-Net grande en Eyes-Defy
    python -m src.training.train_conjuntiva_movil entrenar   # MobileNetV2-UNet 256 px
    python -m src.training.train_conjuntiva_movil anemia     # AUC de anemia con el chico + coeficientes para Kotlin

Datos de segmentación: data/processed/conjuntiva (411 máscaras a mano, Roboflow/HF anemia-seg) + 217 fotos de
Eyes-Defy con la máscara del U-Net ResNet50 (650 MB, el del servidor) como etiqueta (destilación).
Entrada igual que en el servidor: la foto completa en letterbox cuadrado, aquí 256 px.
Salida: models/conjuntiva_movil/{best.keras, conjuntiva_movil.tflite, anemia_movil.json, vectores_prueba.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

LADO = 256
OUT = PROJECT_ROOT / "models/conjuntiva_movil"
PSEUDO = PROJECT_ROOT / "data/processed/conjuntiva_pseudo"
PROC = PROJECT_ROOT / "data/processed/conjuntiva"
RASGOS = PROJECT_ROOT / "reports/anemia_hb/rasgos.parquet"


def letterbox(img: np.ndarray, lado: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    s = lado / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    t, l = (lado - nh) // 2, (lado - nw) // 2
    out = np.zeros((lado, lado) + img.shape[2:], img.dtype)
    out[t:t + nh, l:l + nw] = img
    util = np.zeros((lado, lado), bool)
    util[t:t + nh, l:l + nw] = True
    return out, util


def pseudo() -> None:
    from src.deployment.inference_anemia import SegmentadorConjuntiva, letterbox as lb512, mayor_componente

    r = pd.read_parquet(RASGOS)
    seg = SegmentadorConjuntiva(PROJECT_ROOT / "models/conjuntiva/best.keras")
    (PSEUDO / "images").mkdir(parents=True, exist_ok=True)
    (PSEUDO / "masks").mkdir(parents=True, exist_ok=True)
    filas = []
    for _, x in r.iterrows():
        if not isinstance(x.foto, str):
            continue
        bgr = cv2.imread(x.foto)
        img, util = lb512(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        conj = mayor_componente((seg(img.astype(np.float32) / 255) * util) >= seg.umbral)
        if conj.sum() < 300:
            continue
        cv2.imwrite(str(PSEUDO / "images" / f"{x.persona}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(PSEUDO / "masks" / f"{x.persona}.png"), conj.astype(np.uint8) * 255)
        filas.append({"id": x.persona, "pais": x.pais})
    pd.DataFrame(filas).to_csv(PSEUDO / "manifest.csv", index=False)
    print("pseudo", len(filas))


def cargar() -> tuple[list, list, list]:
    """(imagen RGB uint8 256, máscara bool 256, grupo) para entrenar y validar."""
    xs, ms, gs = [], [], []
    man = pd.read_csv(PROC / "manifest.csv")
    for _, r in man.iterrows():
        im = cv2.imread(str(PROC / r.image))
        mk = cv2.imread(str(PROC / r.vessel), cv2.IMREAD_GRAYSCALE)
        if im is None or mk is None:
            continue
        a, _ = letterbox(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), LADO)
        b, _ = letterbox(mk, LADO)
        xs.append(a); ms.append(b > 127); gs.append(f"seg_{r.split}")
    pm = pd.read_csv(PSEUDO / "manifest.csv")
    for _, r in pm.iterrows():
        im = cv2.imread(str(PSEUDO / "images" / f"{r.id}.png"))
        mk = cv2.imread(str(PSEUDO / "masks" / f"{r.id}.png"), cv2.IMREAD_GRAYSCALE)
        a, _ = letterbox(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), LADO)
        b = cv2.resize(mk, (LADO, LADO), interpolation=cv2.INTER_NEAREST)
        xs.append(a); ms.append(b > 127); gs.append(f"eyesdefy_{r.id}")
    return xs, ms, gs


def modelo():
    import tensorflow as tf

    L = tf.keras.layers
    base = tf.keras.applications.MobileNetV2((LADO, LADO, 3), alpha=0.75, include_top=False, weights="imagenet")
    nombres = ["block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu", "block_16_project"]
    enc = tf.keras.Model(base.input, [base.get_layer(n).output for n in nombres])
    x_in = L.Input((LADO, LADO, 3))
    x = L.Rescaling(1 / 127.5, offset=-1)(x_in)
    s1, s2, s3, s4, x = enc(x)
    for skip, f in ((s4, 128), (s3, 64), (s2, 48), (s1, 32)):
        x = L.UpSampling2D(interpolation="bilinear")(x)
        x = L.Concatenate()([x, skip])
        x = L.SeparableConv2D(f, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
    x = L.UpSampling2D(interpolation="bilinear")(x)
    x = L.Conv2D(16, 3, padding="same", activation="relu")(x)
    out = L.Conv2D(1, 1, activation="sigmoid", dtype="float32")(x)
    return tf.keras.Model(x_in, out), base


def entrenar() -> None:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    xs, ms, gs = cargar()
    X = np.stack(xs).astype(np.uint8)
    M = np.stack(ms)[..., None].astype(np.float32)
    gs = np.array(gs)
    # validación: el split de prueba de seg + 20 % de Eyes-Defy (por persona)
    rng = np.random.default_rng(0)
    ed = np.array([g.startswith("eyesdefy") for g in gs])
    val = (gs == "seg_test") | (ed & (rng.random(len(gs)) < 0.2))
    print("train", (~val).sum(), "val", val.sum(), flush=True)

    def aug(x, m):
        if tf.random.uniform(()) < 0.5:
            x, m = tf.image.flip_left_right(x), tf.image.flip_left_right(m)
        k = tf.random.uniform((), 0, 4, tf.int32)
        x, m = tf.image.rot90(x, k), tf.image.rot90(m, k)
        x = tf.cast(x, tf.float32)
        x = tf.image.random_brightness(x, 25.0)
        x = tf.image.random_contrast(x, 0.8, 1.2)
        x = tf.image.random_saturation(x, 0.8, 1.2)
        return tf.clip_by_value(x, 0, 255), m

    tr = tf.data.Dataset.from_tensor_slices((X[~val], M[~val])).shuffle(800).map(aug, num_parallel_calls=4).batch(16).prefetch(2)
    va = tf.data.Dataset.from_tensor_slices((X[val].astype(np.float32), M[val])).batch(16)

    def dice(y, p):
        i = tf.reduce_sum(y * p, [1, 2, 3])
        return tf.reduce_mean((2 * i + 1) / (tf.reduce_sum(y, [1, 2, 3]) + tf.reduce_sum(p, [1, 2, 3]) + 1))

    def perdida(y, p):
        return tf.keras.losses.binary_crossentropy(y, p) + (1 - dice(y, p))

    m, base = modelo()
    OUT.mkdir(parents=True, exist_ok=True)
    cb = [tf.keras.callbacks.ModelCheckpoint(str(OUT / "best.keras"), monitor="val_dice", mode="max", save_best_only=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_dice", mode="max", factor=0.5, patience=4, min_lr=1e-5),
          tf.keras.callbacks.EarlyStopping(monitor="val_dice", mode="max", patience=12, restore_best_weights=True),
          tf.keras.callbacks.CSVLogger(str(OUT / "historia.csv"), append=True)]
    base.trainable = False
    m.compile(tf.keras.optimizers.Adam(1e-3), perdida, metrics=[dice])
    m.fit(tr, validation_data=va, epochs=6, callbacks=cb, verbose=2)
    base.trainable = True
    m.compile(tf.keras.optimizers.Adam(3e-4), perdida, metrics=[dice])
    m.fit(tr, validation_data=va, epochs=70, callbacks=cb, verbose=2)
    m = tf.keras.models.load_model(OUT / "best.keras", compile=False)
    p = m.predict(X[val].astype(np.float32), batch_size=16, verbose=0)
    d = [(2 * ((pi > 0.5) & (mi > 0.5)).sum() + 1) / ((pi > 0.5).sum() + (mi > 0.5).sum() + 1) for pi, mi in zip(p, M[val])]
    print("Dice val (0.5):", round(float(np.mean(d)), 3), "eyes-defy:", round(float(np.mean(np.array(d)[ed[val]])), 3))
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    (OUT / "conjuntiva_movil.tflite").write_bytes(conv.convert())
    print("tflite MB", round((OUT / "conjuntiva_movil.tflite").stat().st_size / 1e6, 2))


# ── medidas de color (idénticas a medir_color, en float64) para portarlas a Kotlin ──
def mayor_componente(m: np.ndarray) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return m
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def lab_exacto(px: np.ndarray) -> np.ndarray:
    """sRGB→Lab con fórmulas exactas (OpenCV usa tablas interpoladas, ±0.3): idéntico a ColorConjuntiva.lab en Kotlin."""
    p = px.astype(np.float64)
    lin = np.where(p <= 0.04045, p / 12.92, ((p + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
    X = lin @ M.T
    x, y, z = X[:, 0] / 0.950456, X[:, 1], X[:, 2] / 1.088754
    f = lambda t: np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)  # noqa: E731
    L = np.where(y > 0.008856, 116 * np.cbrt(y) - 16, 903.3 * y)
    return np.c_[L, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))]


def medir_exacto(px: np.ndarray) -> dict[str, float]:
    """medir_color sin balance de blancos, en float64 y con Lab exacto (lo mismo que hace el teléfono)."""
    px = px.astype(np.float64)
    v = px.max(1)
    sel = px[(v > 0.08) & (v < 0.98)]
    if len(sel) < 50:
        sel = px
    eps = 1e-3
    r, g, b = sel[:, 0] + eps, sel[:, 1] + eps, sel[:, 2] + eps
    er = np.log(r) - np.log(g)
    lab = lab_exacto(sel)
    mx, mn = sel.max(1), sel.min(1)
    s = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1), 0)
    d = np.where(mx == mn, 1, mx - mn)
    R, G, B = sel[:, 0], sel[:, 1], sel[:, 2]
    h = np.where(mx == mn, 0, np.where(mx == R, 60 * (G - B) / d, np.where(mx == G, 120 + 60 * (B - R) / d, 240 + 60 * (R - G) / d)))
    h = np.where(h < 0, h + 360, h)
    rojo = ((h < 20) | (h > 340)) & (s > 0.35)
    suma = r + g + b
    return {"eritema": float(er.mean()), "eritema_p75": float(np.percentile(er, 75)), "a_media": float(lab[:, 1].mean()),
            "a_p25": float(np.percentile(lab[:, 1], 25)), "a_p75": float(np.percentile(lab[:, 1], 75)),
            "l_media": float(lab[:, 0].mean()), "b_media": float(lab[:, 2].mean()), "crom_r": float((r / suma).mean()),
            "crom_g": float((g / suma).mean()), "rojo_alto": float(rojo.mean()), "saturacion": float(s.mean()),
            "r_sobre_g": float(np.median(r / g))}


def anemia() -> None:
    """Rasgos con el segmentador chico (como en el teléfono) → AUC dentro del país y coeficientes."""
    import tensorflow as tf
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.deployment.inference_anemia import CARACTERISTICAS, medir_color

    interp = tf.lite.Interpreter(str(OUT / "conjuntiva_movil.tflite"))
    interp.allocate_tensors()
    i_in, i_out = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]

    def segmentar(rgb256: np.ndarray) -> np.ndarray:
        interp.set_tensor(i_in, rgb256[None].astype(np.float32)); interp.invoke()
        a = interp.get_tensor(i_out)[0, ..., 0]
        interp.set_tensor(i_in, rgb256[None, :, ::-1].astype(np.float32)); interp.invoke()
        b = interp.get_tensor(i_out)[0, :, ::-1, 0]
        return (a + b) / 2

    r = pd.read_parquet(RASGOS)
    filas, vect = [], []
    for _, x in r.iterrows():
        if not isinstance(x.foto, str):
            continue
        rgb = cv2.cvtColor(cv2.imread(x.foto), cv2.COLOR_BGR2RGB)
        img, util = letterbox(rgb, LADO)
        prob = segmentar(img) * util
        conj = mayor_componente(prob >= 0.5)
        if conj.sum() < 80:
            filas.append({"persona": x.persona}); continue
        px = (img.astype(np.float32) / 255)[conj]
        med = medir_exacto(px)
        filas.append({"persona": x.persona, **{f"m_{k}": v for k, v in med.items()}, "frac": float(conj.mean())})
        if len(vect) < 5:
            sub = px[np.random.default_rng(len(vect)).choice(len(px), min(400, len(px)), replace=False)]
            sub = np.round(sub.astype(np.float64), 6).astype(np.float32)  # como quedan en el JSON
            vect.append({"persona": x.persona, "pixeles": [round(float(t), 6) for t in sub.reshape(-1)], "medidas": medir_exacto(sub)})
    f = r[["persona", "pais", "anemia", "hb"]].merge(pd.DataFrame(filas), on="persona")
    cols = [f"m_{k}" for k in CARACTERISTICAS]
    ok = f[cols].notna().all(1)
    print("válidas", ok.sum(), "de", len(f))
    f = f[ok].reset_index(drop=True)
    X, y = f[cols].to_numpy(float), f.anemia.to_numpy()
    ind = (f.pais == "India").to_numpy()
    fab = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"))  # noqa: E731
    ai, at = [], []
    for rep in range(20):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, ind * 2 + y):
            p[te] = fab().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        ai.append(roc_auc_score(y[ind], p[ind])); at.append(roc_auc_score(y[~ind], p[~ind]))
    dentro = (np.mean(ai) * ind.sum() + np.mean(at) * (~ind).sum()) / len(y)
    print(f"AUC chico: India {np.mean(ai):.3f} Italia {np.mean(at):.3f} dentro {dentro:.3f}")
    m = fab().fit(X, y)
    sc, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
    paq = {"caracteristicas": list(CARACTERISTICAS), "media": sc.mean_.tolist(), "escala": sc.scale_.tolist(),
           "coef": lr.coef_[0].tolist(), "intercepto": float(lr.intercept_[0]), "zona_gris": [0.35, 0.65],
           "umbral_mascara": 0.5, "lado": LADO,
           "validacion": {"auc_cv": round(float(dentro), 4), "auc_india": round(float(np.mean(ai)), 4),
                          "auc_italia": round(float(np.mean(at)), 4), "n": int(len(y)), "n_anemia": int(y.sum()),
                          "datos": "Eyes-Defy-Anemia con Hb (217); segmentador MobileNetV2 en el teléfono"}}
    for v in vect:
        z = (np.array([v["medidas"][k] for k in CARACTERISTICAS]) - sc.mean_) / sc.scale_
        v["probabilidad"] = float(1 / (1 + np.exp(-(z @ lr.coef_[0] + lr.intercept_[0]))))
    json.dump(paq, open(OUT / "anemia_movil.json", "w"), indent=1)
    json.dump(vect, open(OUT / "vectores_prueba.json", "w"))
    dst = PROJECT_ROOT / "mobile/shared/src/androidHostTest/resources/anemia_movil_vectores.json"
    json.dump(vect, open(dst, "w"))
    for a in ("anemia_movil.json", "conjuntiva_movil.tflite"):
        (PROJECT_ROOT / "mobile/androidApp/src/main/assets" / a).write_bytes((OUT / a).read_bytes())
    print(json.dumps(paq["validacion"]))


if __name__ == "__main__":
    {"pseudo": pseudo, "entrenar": entrenar, "anemia": anemia}[sys.argv[1]]()
