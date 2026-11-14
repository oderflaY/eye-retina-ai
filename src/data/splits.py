"""División estratificada 70/15/15 y K-Fold reproducibles."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def _strata(labels: Sequence[int | str]) -> np.ndarray | None:
    """Devuelve las etiquetas de estrato si cada clase tiene al menos 2 muestras."""
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return None
    return np.asarray(labels)


def stratified_split(
    labels: Sequence[int | str],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Divide índices en train/val/test estratificando por ``labels``.

    Si algún estrato tiene menos de 2 elementos se hace una división aleatoria
    simple (sklearn no puede estratificar en ese caso).

    Args:
        labels: etiqueta de estrato por muestra (p. ej. número de ojos).
        train, val, test: proporciones; deben sumar 1.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` con índices ordenados.
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("Las proporciones train/val/test deben sumar 1")
    idx = np.arange(len(labels))
    labels = list(labels)
    if len(idx) < 3:
        return {"train": idx.tolist(), "val": [], "test": []}
    hold = val + test
    tr, rest = train_test_split(idx, test_size=hold, random_state=seed, stratify=_strata(labels))
    rest_labels = [labels[i] for i in rest]
    if test == 0 or len(rest) < 2:
        va, te = rest, np.array([], dtype=int)
    else:
        va, te = train_test_split(
            rest,
            test_size=test / hold,
            random_state=seed,
            stratify=_strata(rest_labels),
        )
    return {"train": sorted(tr.tolist()), "val": sorted(va.tolist()), "test": sorted(te.tolist())}


def kfold_splits(labels: Sequence[int | str], k: int = 5, seed: int = 42) -> list[dict[str, list[int]]]:
    """Genera ``k`` particiones estratificadas train/val (para datasets pequeños)."""
    y = np.asarray(list(labels))
    strat = _strata(list(labels))
    n_min = min(Counter(labels).values()) if strat is not None else 0
    if strat is None or n_min < k:
        y = np.zeros(len(labels), dtype=int)  # sin estratificar si no alcanza
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [
        {"train": sorted(tr.tolist()), "val": sorted(va.tolist())}
        for tr, va in skf.split(np.zeros(len(y)), y)
    ]
