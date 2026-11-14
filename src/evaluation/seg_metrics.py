"""Métricas de segmentación de vasos (NumPy), calculadas dentro del FOV.

Dice (= F1), IoU, sensibilidad, especificidad, exactitud, AUC-ROC y HD95.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def confusion(pred: np.ndarray, gt: np.ndarray, fov: np.ndarray | None = None) -> tuple[int, int, int, int]:
    """TP, FP, FN, TN de máscaras binarias (solo píxeles del FOV)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    m = np.ones_like(gt) if fov is None else fov.astype(bool)
    tp = int(np.sum(pred & gt & m))
    fp = int(np.sum(pred & ~gt & m))
    fn = int(np.sum(~pred & gt & m))
    tn = int(np.sum(~pred & ~gt & m))
    return tp, fp, fn, tn


def from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Métricas a partir de la matriz de confusión."""
    eps = 1e-9
    return {
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
        "precision": tp / (tp + fp + eps),
    }


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Distancia de Hausdorff al percentil 95 entre los bordes (píxeles).

    Devuelve ``nan`` si alguna máscara está vacía.
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    bp = pred ^ ndimage.binary_erosion(pred)
    bg = gt ^ ndimage.binary_erosion(gt)
    dt_g = ndimage.distance_transform_edt(~bg)
    dt_p = ndimage.distance_transform_edt(~bp)
    d = np.concatenate([dt_g[bp], dt_p[bg]])
    return float(np.percentile(d, 95))


def auc_in_fov(prob: np.ndarray, gt: np.ndarray, fov: np.ndarray | None = None, max_pixels: int = 200_000,
               seed: int = 0) -> float:
    """AUC-ROC píxel a píxel dentro del FOV (submuestreo para que sea rápido)."""
    m = np.ones(gt.shape, bool) if fov is None else fov.astype(bool)
    p, g = prob[m].ravel(), gt[m].ravel().astype(np.uint8)
    if g.min() == g.max():
        return float("nan")
    if p.size > max_pixels:
        idx = np.random.default_rng(seed).choice(p.size, max_pixels, replace=False)
        p, g = p[idx], g[idx]
    return float(roc_auc_score(g, p))


def image_metrics(prob: np.ndarray, gt: np.ndarray, fov: np.ndarray | None, threshold: float = 0.5,
                  with_hd95: bool = True) -> dict[str, float]:
    """Todas las métricas de una imagen."""
    pred = prob >= threshold
    if fov is not None:
        pred = pred & fov.astype(bool)
    tp, fp, fn, tn = confusion(pred, gt, fov)
    out = from_confusion(tp, fp, fn, tn)
    out["auc"] = auc_in_fov(prob, gt, fov)
    if with_hd95:
        out["hd95"] = hd95(pred, gt.astype(bool))
    out.update(tp=tp, fp=fp, fn=fn, tn=tn)
    return out


def best_threshold(probs: list[np.ndarray], gts: list[np.ndarray], fovs: list[np.ndarray],
                   grid: np.ndarray | None = None) -> tuple[float, float]:
    """Umbral que maximiza el Dice global (en validación). Devuelve (umbral, dice)."""
    grid = np.arange(0.2, 0.81, 0.05) if grid is None else grid
    best = (0.5, -1.0)
    for t in grid:
        tot = np.zeros(4, np.int64)
        for p, g, f in zip(probs, gts, fovs):
            tot += confusion(p >= t, g, f)
        d = from_confusion(*tot)["dice"]
        if d > best[1]:
            best = (float(round(t, 3)), float(d))
    return best
