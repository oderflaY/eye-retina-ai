"""Pruebas de codificación de rejilla, splits y métricas (sin TensorFlow)."""

from __future__ import annotations

import numpy as np

from src.data.encoding import GridEncoder, iou_matrix
from src.data.splits import kfold_splits, stratified_split
from src.evaluation.eye_metrics import ImageResult, average_precision, evaluate_detections


def _example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.array([[40, 50, 30, 14], [120, 52, 32, 16]], np.float32)
    lms = np.stack(
        [
            np.array(
                [
                    [b[0] + b[2], b[1] + b[3] / 2],
                    [b[0], b[1] + b[3] / 2],
                    [b[0] + b[2] / 2, b[1]],
                    [b[0] + b[2] / 2, b[1] + b[3]],
                    [b[0] + 12, b[1] + 7],
                    [b[0] + 6, b[1] + 7],
                ]
            )
            for b in boxes
        ]
    ).astype(np.float32)
    return boxes, lms, np.ones((2, 6), bool)


def test_encode_decode_roundtrip() -> None:
    enc = GridEncoder(224)
    boxes, lms, mask = _example()
    target, coll = enc.encode(boxes, lms, mask)
    assert target.shape == (14, 14, 23) and coll == 0 and target[..., 0].sum() == 2
    dets = enc.decode(target[..., :17], score_threshold=0.5, is_target=True)
    assert len(dets) == 2
    got = np.stack(sorted([d.box for d in dets], key=lambda b: b[0]))
    np.testing.assert_allclose(got, boxes, atol=1e-3)
    got_lm = sorted(dets, key=lambda d: d.box[0])[0].landmarks
    np.testing.assert_allclose(got_lm, lms[0], atol=1e-3)


def test_collision_keeps_larger() -> None:
    enc = GridEncoder(224)
    boxes = np.array([[100, 100, 10, 6], [101, 101, 20, 8]], np.float32)
    lms = np.zeros((2, 6, 2), np.float32)
    target, coll = enc.encode(boxes, lms, np.zeros((2, 6), bool))
    assert coll == 1 and target[..., 0].sum() == 1
    j, i = np.argwhere(target[..., 0] == 1)[0]
    assert np.isclose(target[j, i, 3] ** 2 * 224, 20, atol=1e-3)


def test_iou() -> None:
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [20, 20, 5, 5]], np.float32)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0], atol=1e-6)


def test_metrics_perfect() -> None:
    enc = GridEncoder(224)
    boxes, lms, mask = _example()
    target, _ = enc.encode(boxes, lms, mask)
    dets = enc.decode(target[..., :17], is_target=True)
    m = evaluate_detections([ImageResult(boxes, lms, dets)])
    assert m["precision"] == 1 and m["recall"] == 1 and abs(m["ap50"] - 1) < 1e-9
    assert m["landmark_nme_mean"] < 1e-4 and m["count_accuracy"] == 1


def test_average_precision_half() -> None:
    ap, _, _ = average_precision(np.array([0.9, 0.8]), np.array([True, False]), n_gt=2)
    assert abs(ap - 0.5) < 1e-9


def test_splits_are_disjoint_and_complete() -> None:
    labels = ["2"] * 60 + ["0"] * 20 + ["1"] * 20
    sp = stratified_split(labels, seed=42)
    allidx = sp["train"] + sp["val"] + sp["test"]
    assert sorted(allidx) == list(range(100)) and len(set(allidx)) == 100
    assert abs(len(sp["train"]) - 70) <= 1 and abs(len(sp["test"]) - 15) <= 1
    folds = kfold_splits(labels, k=5)
    assert len(folds) == 5 and all(set(f["train"]).isdisjoint(f["val"]) for f in folds)
