"""Convierte el detector de ojo de cerca a TFLite para la app y lo verifica.

    python -m src.deployment.export_tflite_ojo [--modelo models/eye_cerca/best.h5]

1. Reconstruye el modelo en float32 (el entrenamiento usa mixed_float16) y lo exporta como
   SavedModel → TFLite con pesos en float16 (≈ la mitad de tamaño, misma precisión en CPU).
2. Compara TFLite contra Keras en fotos de prueba (deben dar lo mismo).
3. Mide en el subconjunto de test: exactitud ojo / no-ojo, IoU de la caja y error de los puntos
   (en % del ancho del ojo), y la apertura medida por párpados en ojos abiertos vs cerrados.
4. Copia el .tflite a ``mobile/androidApp/src/main/assets/ojo_detector.tflite``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.config import PROJECT_ROOT
from src.data.loader_eyes import EyeDataLoader
from src.models.eye_detector import build_eye_detector

ASSET = PROJECT_ROOT / "mobile/androidApp/src/main/assets/ojo_detector.tflite"


def a_float32(path: Path, lado: int) -> tf.keras.Model:
    tf.keras.mixed_precision.set_global_policy("float32")
    entrenado = tf.keras.models.load_model(path, compile=False)
    limpio = build_eye_detector((lado, lado, 3), 6, weights=None)
    limpio.set_weights(entrenado.get_weights())
    return limpio


def convertir(modelo: tf.keras.Model, salida: Path) -> bytes:
    sm = salida.with_suffix("")
    if sm.exists():
        shutil.rmtree(sm)
    modelo.export(str(sm))
    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    datos = conv.convert()
    salida.write_bytes(datos)
    shutil.rmtree(sm)
    return datos


class Tflite:
    def __init__(self, datos: bytes) -> None:
        self.it = tf.lite.Interpreter(model_content=datos)
        self.it.allocate_tensors()
        self.ent = self.it.get_input_details()[0]["index"]
        self.sal = {d["shape"][-1]: d["index"] for d in self.it.get_output_details()}

    def __call__(self, x: np.ndarray) -> dict[str, np.ndarray]:
        self.it.set_tensor(self.ent, x[None].astype(np.float32))
        self.it.invoke()
        g = lambda n: self.it.get_tensor(self.sal[n])[0]  # noqa: E731
        return {"bbox": g(4), "landmarks": g(12), "eye_prob": g(1)}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    i = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    u = a[2] * a[3] + b[2] * b[3] - i
    return float(i / u) if u > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eye_cerca.yaml")
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--max_test", type=int, default=1500)
    a = ap.parse_args()
    cfg = yaml.safe_load((PROJECT_ROOT / a.config).read_text())
    d = cfg["data"]
    lado = int(d["img_size"][0])
    ruta = PROJECT_ROOT / (a.modelo or cfg["output"]["checkpoint"])
    destino = ruta.with_name("ojo_detector.tflite")

    keras_m = a_float32(ruta, lado)
    datos = convertir(keras_m, destino)
    lite = Tflite(datos)
    print(f"TFLite: {len(datos) / 1e6:.1f} MB")

    loader = EyeDataLoader(PROJECT_ROOT / d["data_dir"], (lado, lado), 1, label_format=d.get("label_format", "auto"),
                           seed=int(cfg.get("seed", 42)), crop_scale=float(d.get("crop_scale", 1.0)),
                           group_prefix=bool(d.get("group_prefix", False)))
    _, _, test = loader.split(loader.load_metadata(), d.get("test_size", 0.1), d.get("val_size", 0.1))
    test = test.sample(min(len(test), a.max_test), random_state=0).reset_index(drop=True)

    filas, dif = [], []
    for i, r in test.iterrows():
        img = tf.image.decode_image(tf.io.read_file(r.image_path), channels=3, expand_animations=False)
        img = tf.image.crop_to_bounding_box(img, int(r.crop_y), int(r.crop_x), int(r.crop_h), int(r.crop_w))
        x = tf.image.resize(tf.image.convert_image_dtype(img, tf.float32), (lado, lado)).numpy()
        o = lite(x)
        if i < 40:
            k = keras_m(x[None], training=False)
            dif.append(max(float(np.abs(k[n].numpy()[0] - o[n]).max()) for n in ("bbox", "landmarks", "eye_prob")))
        f = {"real": int(r.eye_prob), "prob": float(o["eye_prob"][0]), "fuente": Path(r.image_path).stem}
        if r.eye_prob == 1:
            caja = np.array([r.bx, r.by, r.bw, r.bh])
            f["iou"] = iou(caja, o["bbox"])
            ancho = max(float(r.bw), 1e-3)
            pred = o["landmarks"].reshape(6, 2)
            for j, n in enumerate(("interna", "externa", "parp_sup", "parp_inf", "pupila", "iris")):
                if r[f"lm{j}_m"] > 0:
                    f[f"err_{n}"] = float(np.hypot(*(pred[j] - [r[f"lm{j}_x"], r[f"lm{j}_y"]]))) / ancho
            # apertura medida igual que en la app: (párpado inf − sup) / ancho
            f["apertura_pred"] = float((pred[3, 1] - pred[2, 1]) / max(np.hypot(*(pred[1] - pred[0])), 1e-3))
            f["abierto"] = int(r["lm4_m"] > 0)
        filas.append(f)
    df = pd.DataFrame(filas)
    ojos = df[df.real == 1]
    pred_ojo = df.prob >= 0.6
    res = {
        "tflite_mb": round(len(datos) / 1e6, 2),
        "max_dif_tflite_vs_keras": round(max(dif), 5),
        "n_test": int(len(df)),
        "exactitud_ojo_no_ojo": round(float((pred_ojo == (df.real == 1)).mean()), 4),
        "sensibilidad_ojo": round(float(pred_ojo[df.real == 1].mean()), 4),
        "especificidad_ojo": round(float((~pred_ojo[df.real == 0]).mean()), 4),
        "iou_mediana": round(float(ojos.iou.median()), 4),
        "iou_mayor_0_5": round(float((ojos.iou > 0.5).mean()), 4),
        "error_puntos_mediana_pct_ancho_ojo": {
            c[4:]: round(float(ojos[c].median() * 100), 2) for c in ojos.columns if c.startswith("err_")},
        "apertura_mediana_abiertos": round(float(ojos[ojos.abierto == 1].apertura_pred.median()), 3),
        "apertura_mediana_cerrados": round(float(ojos[ojos.abierto == 0].apertura_pred.median()), 3),
    }
    rep = PROJECT_ROOT / "reports/eye_cerca"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "tflite.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(destino, ASSET)
    print("copiado a", ASSET)


if __name__ == "__main__":
    main()
