"""Escanea ``data/raw/eyes`` y organiza ``data/processed/eyes`` (+ vista previa aumentada).

Uso::

    python -m src.data.prepare_eyes --config configs/eye_grid.yaml
    python -m src.data.prepare_eyes --config configs/eye_grid.yaml --preview 16

Genera:

* ``annotations.json`` – muestras en formato unificado (píxeles originales).
* ``splits.json``      – índices train/val/test (y folds si ``cv_folds > 1``).
* ``stats.json``       – estadísticas para el EDA.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data.eye_labels import LANDMARK_NAMES, EyeSample, load_eye_samples
from src.data.splits import kfold_splits, stratified_split
from src.utils.config import load_config, resolve_path


def compute_stats(samples: list[EyeSample]) -> dict[str, Any]:
    """Estadísticas descriptivas del dataset de ojos."""
    eyes_per_img = Counter(min(s.num_eyes, 3) for s in samples)
    rel_w = [b[2] / s.width for s in samples for b in s.boxes]
    lm_cov = np.zeros(len(LANDMARK_NAMES))
    n_eyes = 0
    for s in samples:
        if s.num_eyes:
            lm_cov += (~np.isnan(s.landmarks).any(-1)).sum(0)
            n_eyes += s.num_eyes
    return {
        "num_images": len(samples),
        "num_eyes": n_eyes,
        "images_by_eye_count": {("3+" if k == 3 else str(k)): v for k, v in sorted(eyes_per_img.items())},
        "box_width_rel": {
            "mean": float(np.mean(rel_w)) if rel_w else None,
            "p05": float(np.percentile(rel_w, 5)) if rel_w else None,
            "p95": float(np.percentile(rel_w, 95)) if rel_w else None,
        },
        "landmark_coverage": {
            n: (float(c / n_eyes) if n_eyes else 0.0) for n, c in zip(LANDMARK_NAMES, lm_cov)
        },
        "image_sizes": dict(Counter(f"{s.width}x{s.height}" for s in samples).most_common(5)),
    }


def strata_labels(samples: list[EyeSample], raw_dir: Path) -> list[str]:
    """Estrato = número de ojos (0, 1, 2, 3+) × subcarpeta de origen."""
    out = []
    for s in samples:
        rel = Path(s.image_path).resolve()
        try:
            top = rel.relative_to(raw_dir.resolve()).parts[0]
        except (ValueError, IndexError):
            top = ""
        top = top if not Path(top).suffix else ""
        out.append(f"{min(s.num_eyes, 3)}|{top}")
    return out


def save_previews(samples: list[EyeSample], cfg: dict[str, Any], n: int, out_dir: Path) -> None:
    """Guarda ``n`` ejemplos aumentados con cajas y landmarks dibujados."""
    from src.data.eye_dataset import EyeDatasetBuilder  # requiere TensorFlow

    out_dir.mkdir(parents=True, exist_ok=True)
    builder = EyeDatasetBuilder(samples, cfg)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    for k, idx in enumerate(rng.choice(len(samples), size=min(n, len(samples)), replace=False)):
        img, boxes, lms, mask, cnt = builder._np_load(np.int64(idx))
        img, boxes, lms, mask = builder._augment(img, boxes[:cnt], lms[:cnt], mask[:cnt] > 0)
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for b, lm, m in zip(boxes, lms, mask):
            x, y, w, h = (int(round(v)) for v in b)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
            for (px, py), ok in zip(lm, m):
                if ok:
                    cv2.circle(vis, (int(px), int(py)), 2, (0, 0, 255), -1)
        cv2.imwrite(str(out_dir / f"aug_{k:03d}.jpg"), vis)


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eye_grid.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="clave=valor")
    ap.add_argument("--preview", type=int, default=0, help="ejemplos aumentados a guardar")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config), args.override)
    dcfg = cfg["data"]
    raw_dir = resolve_path(dcfg["raw_dir"])
    out_dir = resolve_path(dcfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_eye_samples(raw_dir, dcfg.get("label_format", "auto"))
    labels = strata_labels(samples, raw_dir)
    sp = dcfg["split"]
    splits: dict[str, Any] = stratified_split(labels, sp["train"], sp["val"], sp["test"], int(cfg["seed"]))
    k = int(dcfg.get("cv_folds", 0) or 0)
    if k > 1:
        # K-Fold sobre train+val; el test se mantiene intacto
        pool = splits["train"] + splits["val"]
        folds = kfold_splits([labels[i] for i in pool], k, int(cfg["seed"]))
        splits["folds"] = [{key: [pool[i] for i in f[key]] for key in f} for f in folds]

    stats = compute_stats(samples)
    stats["split_sizes"] = {key: len(v) for key, v in splits.items() if key != "folds"}
    (out_dir / "annotations.json").write_text(
        json.dumps({"samples": [s.to_dict() for s in samples]}, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nGuardado en {out_dir}")

    if args.preview:
        save_previews(samples, cfg, args.preview, resolve_path(dcfg["augmented_dir"]))
        print(f"Vista previa aumentada en {resolve_path(dcfg['augmented_dir'])}")


if __name__ == "__main__":
    main()
