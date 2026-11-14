"""Métricas de detección de ojos.

* Por caja (IoU ≥ 0.5): Precision, Recall, F1, AP@0.5 y curva Precision-Recall.
* Por celda de la rejilla: AUC-ROC de la objetividad.
* Por imagen: *accuracy* de conteo (¿se detectó el número correcto de ojos?)
  y matriz de confusión normalizada del conteo (0, 1, 2, 3+).
* Landmarks: NME (error normalizado por la diagonal de la caja real) por punto.
* Localización: IoU medio de las detecciones correctas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.data.encoding import Detection, iou_matrix
from src.data.eye_labels import LANDMARK_NAMES, NUM_LANDMARKS

COUNT_LABELS = ["0", "1", "2", "3+"]


@dataclass
class ImageResult:
    """Detecciones y verdad de una imagen (mismo espacio de coordenadas)."""

    gt_boxes: np.ndarray
    gt_landmarks: np.ndarray  # (N, 6, 2) con NaN si falta
    dets: list[Detection]
    image_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def match(
    dets: list[Detection], gt: np.ndarray, iou_thr: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empareja detecciones (ordenadas por puntaje) con cajas reales.

    Returns:
        (tp por detección, índice gt emparejado o -1, IoU del emparejamiento).
    """
    order = np.argsort([-d.score for d in dets])
    tp = np.zeros(len(dets), bool)
    gt_idx = np.full(len(dets), -1, int)
    ious_out = np.zeros(len(dets), np.float32)
    if len(dets) == 0 or len(gt) == 0:
        return tp, gt_idx, ious_out
    ious = iou_matrix(np.stack([d.box for d in dets]), gt)
    used = np.zeros(len(gt), bool)
    for k in order:
        cand = np.where(~used)[0]
        if cand.size == 0:
            break
        j = cand[np.argmax(ious[k, cand])]
        if ious[k, j] >= iou_thr:
            tp[k], gt_idx[k], ious_out[k] = True, j, ious[k, j]
            used[j] = True
    return tp, gt_idx, ious_out


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> tuple[float, np.ndarray, np.ndarray]:
    """AP estilo VOC (todos los puntos) y la curva precision-recall."""
    if n_gt == 0:
        return float("nan"), np.array([]), np.array([])
    order = np.argsort(-scores)
    tp = tp[order].astype(np.float64)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, precision, recall


def evaluate_detections(
    results: list[ImageResult],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    cell_scores: np.ndarray | None = None,
    cell_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calcula todas las métricas del detector.

    Args:
        results: una entrada por imagen; ``dets`` debe incluir detecciones con
            puntaje bajo (p. ej. ≥ 0.05) para trazar la curva PR completa.
        score_threshold: umbral operativo para precision/recall/F1 y conteo.
        cell_scores, cell_labels: probabilidades y etiquetas por celda para AUC-ROC.
    """
    all_scores, all_tp = [], []
    n_gt = 0
    tp_op = fp_op = fn_op = 0
    ious_ok: list[float] = []
    lm_err: list[list[float]] = [[] for _ in range(NUM_LANDMARKS)]
    count_true, count_pred = [], []

    for r in results:
        n_gt += len(r.gt_boxes)
        tp, gt_idx, ious = match(r.dets, r.gt_boxes, iou_threshold)
        all_scores += [d.score for d in r.dets]
        all_tp += tp.tolist()
        op = np.array([d.score >= score_threshold for d in r.dets], bool)
        tp_op += int((tp & op).sum())
        fp_op += int((~tp & op).sum())
        fn_op += len(r.gt_boxes) - int((tp & op).sum())
        ious_ok += ious[tp & op].tolist()
        for k in np.where(tp & op)[0]:
            g = gt_idx[k]
            diag = float(np.hypot(*r.gt_boxes[g][2:]))
            for p in range(NUM_LANDMARKS):
                gt_pt = r.gt_landmarks[g, p]
                if not np.isnan(gt_pt).any():
                    lm_err[p].append(float(np.linalg.norm(r.dets[k].landmarks[p] - gt_pt)) / max(diag, 1e-6))
        count_true.append(min(len(r.gt_boxes), 3))
        count_pred.append(min(int(op.sum()), 3))

    ap, prec_curve, rec_curve = average_precision(np.asarray(all_scores), np.asarray(all_tp, bool), n_gt)
    precision = tp_op / max(tp_op + fp_op, 1)
    recall = tp_op / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    cm = confusion_matrix(count_true, count_pred, labels=[0, 1, 2, 3])
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    auc = None
    if cell_scores is not None and cell_labels is not None and len(np.unique(cell_labels)) == 2:
        auc = float(roc_auc_score(cell_labels, cell_scores))

    nme = {name: (float(np.mean(e)) if e else None) for name, e in zip(LANDMARK_NAMES, lm_err)}
    valid = [v for v in nme.values() if v is not None]
    return {
        "num_images": len(results),
        "num_gt_eyes": n_gt,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50": ap,
        "tp": tp_op,
        "fp": fp_op,
        "fn": fn_op,
        "mean_iou_tp": float(np.mean(ious_ok)) if ious_ok else None,
        "count_accuracy": (
            float(np.mean(np.asarray(count_true) == np.asarray(count_pred))) if results else None
        ),
        "cell_auc_roc": auc,
        "landmark_nme": nme,
        "landmark_nme_mean": float(np.mean(valid)) if valid else None,
        "count_confusion": cm.tolist(),
        "count_confusion_norm": cm_norm.round(4).tolist(),
        "_pr_curve": {"precision": prec_curve.tolist(), "recall": rec_curve.tolist()},
    }
