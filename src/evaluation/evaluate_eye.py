"""Evaluación e interpretabilidad del detector de ojos de regresión directa.

Uso::

    python src/evaluation/evaluate_eye.py --config configs/eye.yaml --model models/eye_best.h5

Métricas:

* Clasificación ojo/no-ojo: accuracy, precision, recall, F1, AUC-ROC, matriz de
  confusión normalizada y curva Precision-Recall.
* Caja (solo filas con ojo): IoU medio, % con IoU ≥ 0.5 y MSE.
* Landmarks: NME por punto (error / diagonal de la caja real) y MSE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import NUM_LANDMARKS  # noqa: E402
from src.data.eye_labels import LANDMARK_NAMES  # noqa: E402
from src.models.eye_detector import GRADCAM_LAYER  # noqa: E402


# --------------------------------------------------------------------------- #
# Predicción y métricas
# --------------------------------------------------------------------------- #
def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict[str, np.ndarray]:
    """Recorre el dataset y junta imágenes, verdad y predicciones.

    Returns:
        Diccionario con ``images`` (N,H,W,3 en [0,1]), ``bbox_true``, ``bbox_pred``,
        ``lm_true``, ``lm_mask``, ``lm_pred``, ``cls_true``, ``cls_score``.
    """
    out: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "images",
            "bbox_true",
            "bbox_pred",
            "lm_true",
            "lm_mask",
            "lm_pred",
            "cls_true",
            "cls_score",
        )
    }
    k2 = 2 * NUM_LANDMARKS
    for batch in dataset:
        x, y = batch[0], batch[1]
        p = model(x, training=False)
        out["images"].append(x.numpy())
        out["bbox_true"].append(y["bbox"].numpy())
        out["bbox_pred"].append(np.asarray(p["bbox"], np.float32))
        out["lm_true"].append(y["landmarks"].numpy()[:, :k2])
        out["lm_mask"].append(y["landmarks"].numpy()[:, k2:])
        out["lm_pred"].append(np.asarray(p["landmarks"], np.float32))
        out["cls_true"].append(y["eye_prob"].numpy()[:, 0])
        out["cls_score"].append(np.asarray(p["eye_prob"], np.float32)[:, 0])
    return {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in out.items()}


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU fila a fila entre cajas (x, y, w, h)."""
    x0 = np.maximum(a[:, 0], b[:, 0])
    y0 = np.maximum(a[:, 1], b[:, 1])
    x1 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y1 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    union = (
        np.clip(a[:, 2], 0, None) * np.clip(a[:, 3], 0, None)
        + np.clip(b[:, 2], 0, None) * np.clip(b[:, 3], 0, None)
        - inter
    )
    return inter / np.maximum(union, 1e-9)


def compute_metrics(preds: dict[str, np.ndarray], threshold: float = 0.5) -> dict[str, Any]:
    """Calcula las métricas a partir de :func:`collect_predictions`.

    Args:
        preds: predicciones y verdad.
        threshold: umbral de ``eye_prob`` para decidir ojo/no-ojo.
    """
    y, s = preds["cls_true"].astype(int), preds["cls_score"]
    yhat = (s >= threshold).astype(int)
    two = len(np.unique(y)) == 2
    m: dict[str, Any] = {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "classification": {
            "accuracy": float(accuracy_score(y, yhat)),
            "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat, zero_division=0)),
            "f1": float(f1_score(y, yhat, zero_division=0)),
            "auc_roc": float(roc_auc_score(y, s)) if two else None,
            "average_precision": float(average_precision_score(y, s)) if two else None,
            "threshold": threshold,
        },
    }
    pos = y == 1
    if pos.any():
        bt, bp = preds["bbox_true"][pos], preds["bbox_pred"][pos]
        iou = box_iou(bt, bp)
        m["bbox"] = {
            "mean_iou": float(iou.mean()),
            "iou_ge_0.5": float((iou >= 0.5).mean()),
            "mse": float(np.mean((bt - bp) ** 2)),
        }
        lt = preds["lm_true"][pos].reshape(-1, NUM_LANDMARKS, 2)
        lp = preds["lm_pred"][pos].reshape(-1, NUM_LANDMARKS, 2)
        mk = preds["lm_mask"][pos] > 0.5
        diag = np.hypot(bt[:, 2], bt[:, 3])[:, None]
        err = np.linalg.norm(lt - lp, axis=-1) / np.maximum(diag, 1e-6)
        nme = {
            n: (float(err[mk[:, i], i].mean()) if mk[:, i].any() else None)
            for i, n in enumerate(LANDMARK_NAMES)
        }
        valid = [v for v in nme.values() if v is not None]
        m["landmarks"] = {
            "nme": nme,
            "nme_mean": float(np.mean(valid)) if valid else None,
            "mse": float(np.sum(((lt - lp) ** 2).sum(-1) * mk) / max(mk.sum(), 1)),
        }
    return m


def evaluate_model(
    model: tf.keras.Model, dataset: tf.data.Dataset, threshold: float = 0.5
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Predice sobre ``dataset`` y devuelve (métricas, predicciones)."""
    preds = collect_predictions(model, dataset)
    return compute_metrics(preds, threshold), preds


# --------------------------------------------------------------------------- #
# Gráficas
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], save_path: Path
) -> None:
    """Matriz de confusión normalizada por fila.

    Args:
        y_true: etiquetas reales (enteros).
        y_pred: etiquetas predichas (enteros).
        classes: nombres en orden de índice.
        save_path: PNG de salida.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
        xticklabels=classes,
        yticklabels=classes,
    )
    ax.set(xlabel="Predicción", ylabel="Real", title="Matriz de confusión (normalizada)")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: Path) -> None:
    """Curva Precision-Recall con su average precision.

    Args:
        y_true: etiquetas binarias.
        y_scores: probabilidades de la clase positiva.
        save_path: PNG de salida.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(np.unique(y_true)) == 2:
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        ax.plot(rec, prec, lw=1.5, label=f"AP = {ap:.3f}")
        ax.legend(loc="lower left")
    else:
        ax.text(0.5, 0.5, "Se necesitan ambas clases", ha="center")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Curva Precision-Recall")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def compute_gradcam(
    model: tf.keras.Model, image: np.ndarray, layer_name: str = GRADCAM_LAYER, output: str = "eye_prob"
) -> np.ndarray:
    """Mapa Grad-CAM para una imagen.

    Args:
        model: detector (salidas en diccionario).
        image: (H, W, 3) en [0, 1].
        layer_name: capa convolucional a explicar (``out_relu`` en MobileNetV2).
        output: salida a explicar; ``eye_prob`` explica la decisión ojo/no-ojo.

    Returns:
        Mapa (H, W) en [0, 1].
    """
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    x = tf.convert_to_tensor(image[None].astype(np.float32))
    with tf.GradientTape() as tape:
        feats, preds = grad_model(x, training=False)
        target = tf.reduce_sum(tf.cast(preds[output], tf.float32))
    # El gradiente se toma respecto a la salida real de la capa; el cast a
    # float32 va después (con mixed precision un cast previo cortaría el grafo).
    grads = tf.cast(tape.gradient(target, feats), tf.float32)
    feats = tf.cast(feats, tf.float32)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * feats, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    return cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def _draw(
    ax: Any,
    img: np.ndarray,
    bbox: np.ndarray | None,
    lms: np.ndarray | None,
    color: str,
    mask: np.ndarray | None = None,
) -> None:
    """Dibuja caja y landmarks normalizados sobre un eje de matplotlib."""
    h, w = img.shape[:2]
    if bbox is not None:
        x, y, bw, bh = bbox * [w, h, w, h]
        ax.add_patch(plt.Rectangle((x, y), bw, bh, fill=False, ec=color, lw=1.5))
    if lms is not None:
        pts = lms.reshape(-1, 2) * [w, h]
        ok = np.ones(len(pts), bool) if mask is None else mask > 0.5
        ax.scatter(pts[ok, 0], pts[ok, 1], s=9, c=color)


def visualize_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    num_samples: int = 9,
    save_path: Path = Path("reports/eye/predicciones.png"),
    preds: dict[str, np.ndarray] | None = None,
) -> None:
    """Rejilla de imágenes: verdad en verde, predicción en rojo.

    Args:
        model: detector.
        dataset: dataset de evaluación (sin aumentación).
        num_samples: imágenes a mostrar.
        save_path: PNG de salida.
        preds: predicciones ya calculadas (evita recalcular).
    """
    preds = preds or collect_predictions(model, dataset.take(max(1, num_samples)))
    n = min(num_samples, len(preds["images"]))
    cols = int(np.ceil(np.sqrt(n))) or 1
    rows = int(np.ceil(n / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i in range(n):
        ax = axes.ravel()[i]
        img = preds["images"][i]
        ax.imshow(img)
        if preds["cls_true"][i] > 0.5:
            _draw(ax, img, preds["bbox_true"][i], preds["lm_true"][i], "lime", preds["lm_mask"][i])
        if preds["cls_score"][i] >= 0.5:
            _draw(ax, img, preds["bbox_pred"][i], preds["lm_pred"][i], "red")
        ax.set_title(f"real={int(preds['cls_true'][i])} p={preds['cls_score'][i]:.2f}", fontsize=8)
    fig.suptitle("Verde = real · Rojo = predicción", fontsize=10)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def error_analysis(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    df: pd.DataFrame | None = None,
    top: int = 20,
    preds: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Tabla con las peores predicciones (mayor error).

    El error combina el MSE de caja y landmarks (filas con ojo) y el error de
    clasificación ``|real − p|``.

    Args:
        model: detector.
        dataset: dataset de evaluación **sin barajar** (para alinear con ``df``).
        df: metadatos del mismo subconjunto (añade ``image_path``).
        top: filas a devolver.
        preds: predicciones ya calculadas.

    Returns:
        DataFrame ordenado de mayor a menor error.
    """
    preds = preds or collect_predictions(model, dataset)
    pos = preds["cls_true"] > 0.5
    bbox_mse = np.where(pos, ((preds["bbox_true"] - preds["bbox_pred"]) ** 2).mean(1), 0.0)
    mk = np.repeat(preds["lm_mask"], 2, axis=1)
    lm_mse = np.where(
        pos, (((preds["lm_true"] - preds["lm_pred"]) ** 2) * mk).sum(1) / np.maximum(mk.sum(1), 1), 0.0
    )
    cls_err = np.abs(preds["cls_true"] - preds["cls_score"])
    out = pd.DataFrame(
        {
            "idx": np.arange(len(cls_err)),
            "eye_real": preds["cls_true"].astype(int),
            "eye_prob": preds["cls_score"].round(4),
            "bbox_mse": bbox_mse,
            "landmarks_mse": lm_mse,
            "cls_error": cls_err,
        }
    )
    out["error_total"] = out["bbox_mse"] + 0.5 * out["landmarks_mse"] + out["cls_error"]
    out["tipo"] = np.select(
        [(out.eye_real == 1) & (out.eye_prob < 0.5), (out.eye_real == 0) & (out.eye_prob >= 0.5)],
        ["falso negativo", "falso positivo"],
        "localización",
    )
    if df is not None and len(df) == len(out):
        out.insert(1, "image_path", df["image_path"].to_numpy())
    return out.sort_values("error_total", ascending=False).head(top).reset_index(drop=True)


def save_evaluation_figures(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    preds: dict[str, np.ndarray],
    df: pd.DataFrame | None,
    out_dir: Path,
    n_gradcam: int = 6,
) -> None:
    """Guarda matriz de confusión, curva PR, predicciones, Grad-CAM y errores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y, s = preds["cls_true"].astype(int), preds["cls_score"]
    plot_confusion_matrix(y, (s >= 0.5).astype(int), ["no ojo", "ojo"], out_dir / "confusion_matrix.png")
    plot_pr_curve(y, s, out_dir / "pr_curve.png")
    visualize_predictions(model, dataset, 9, out_dir / "predicciones.png", preds)
    errs = error_analysis(model, dataset, df, top=20, preds=preds)
    errs.to_csv(out_dir / "peores_predicciones.csv", index=False)
    # Grad-CAM: original | mapa | predicción
    n = min(n_gradcam, len(preds["images"]))
    if n:
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
        for i in range(n):
            img = preds["images"][i]
            cam = compute_gradcam(model, img)
            axes[i, 0].imshow(img)
            axes[i, 1].imshow(img)
            axes[i, 1].imshow(cam, cmap="jet", alpha=0.45)
            axes[i, 2].imshow(img)
            if s[i] >= 0.5:
                _draw(axes[i, 2], img, preds["bbox_pred"][i], preds["lm_pred"][i], "red")
            for ax, t in zip(axes[i], ("Original", "Grad-CAM (eye_prob)", f"Predicción p={s[i]:.2f}")):
                ax.set_title(t, fontsize=8)
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "gradcam.png", dpi=110)
        plt.close(fig)


def main() -> None:
    """CLI: evalúa un modelo guardado sobre el split de test."""
    from src.config import PROJECT_ROOT
    from src.data.loader_eyes import EyeDataLoader
    from src.deployment.inference_eye import load_keras_model
    from src.utils.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eye.yaml")
    ap.add_argument("--model", default=None, help="ruta .h5/.keras (por defecto output.checkpoint)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)
    d = cfg["data"]
    loader = EyeDataLoader(
        PROJECT_ROOT / d["data_dir"],
        tuple(d["img_size"]),
        int(cfg["train"]["batch_size"]),
        d.get("label_format", "auto"),
        int(cfg.get("seed", 42)),
        float(d.get("crop_scale", 2.5)),
    )
    parts = dict(
        zip(("train", "val", "test"), loader.split(loader.load_metadata(), d["test_size"], d["val_size"]))
    )
    df = parts[args.split]
    model = load_keras_model(PROJECT_ROOT / (args.model or cfg["output"]["checkpoint"]))
    ds = loader.build_tfdataset(df, cache=False)
    metrics, preds = evaluate_model(model, ds)
    out_dir = PROJECT_ROOT / cfg["output"]["figures_dir"] / args.split
    save_evaluation_figures(model, ds, preds, df, out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
