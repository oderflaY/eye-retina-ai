"""Evalúa la U-Net de vasos: umbral en validación, test (FIVES) y externa (HRF).

Uso::

    python src/evaluation/evaluate_retina.py --config configs/retina_unet.yaml
    python src/evaluation/evaluate_retina.py --config configs/retina_unet.yaml --model models/retina/best.keras

Salidas en ``output.report_dir``:

* ``retina_results.json``: umbral elegido y métricas por división, dataset y
  categoría (promedio por imagen y global agregando píxeles).
* ``per_image.csv``: métricas de cada imagen.
* ``figures/``: imagen | anotación | probabilidad | errores (blanco = acierto,
  rojo = falso positivo, verde = vaso no detectado), los mejores y peores casos.

Las métricas se calculan a la resolución preprocesada (1024 px) y dentro del FOV.
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
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.data.loader_retina import read_manifest
from src.deployment.inference_retina import RetinaSegmenter
from src.evaluation.seg_metrics import best_threshold, from_confusion, image_metrics
from src.utils.config import load_config

METRICS = ("dice", "iou", "sensitivity", "specificity", "accuracy", "auc", "hd95")


def _p(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _read(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = cv2.cvtColor(cv2.imread(row["image"]), cv2.COLOR_BGR2RGB)
    gt = (cv2.imread(row["vessel"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    fov = (cv2.imread(row["fov"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    return img, gt, fov


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RGB: TP blanco, FP rojo, FN verde."""
    out = np.zeros(gt.shape + (3,), np.uint8)
    out[pred & gt] = (255, 255, 255)
    out[pred & ~gt] = (230, 40, 40)
    out[~pred & gt] = (40, 220, 60)
    return out


def save_figure(path: Path, img: np.ndarray, gt: np.ndarray, prob: np.ndarray, thr: float, title: str) -> None:
    """Panel de 4 columnas a 512 px."""
    pred = prob >= thr
    s = 512
    cols = [
        img,
        np.repeat(gt[..., None] * 255, 3, -1).astype(np.uint8),
        cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1],
        error_map(pred, gt.astype(bool)),
    ]
    panel = np.hstack([cv2.resize(c, (s, s), interpolation=cv2.INTER_AREA) for c in cols])
    bar = np.full((36, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack([bar, panel]), cv2.COLOR_RGB2BGR))


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Promedio por imagen (± desv.) y métricas globales agregando píxeles."""
    out: dict[str, Any] = {"n": int(len(df))}
    for m in METRICS:
        if m in df:
            out[m] = {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=0))}
    pooled = from_confusion(*(int(df[k].sum()) for k in ("tp", "fp", "fn", "tn")))
    out["pooled"] = {k: float(v) for k, v in pooled.items()}
    return out


def evaluate(config: dict[str, Any], model_path: Path | None = None, segmenter: RetinaSegmenter | None = None,
             splits: tuple[str, ...] = ("test", "external")) -> dict[str, Any]:
    """Elige el umbral en ``val`` y evalúa ``splits``. Devuelve el resumen."""
    ocfg = config["output"]
    model_path = model_path or _p(ocfg["dir"]) / "best.keras"
    seg = segmenter or RetinaSegmenter(model_path, tta=bool(config.get("evaluation", {}).get("tta", True)))
    report_dir = _p(ocfg["report_dir"])
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(_p(config["data"]["processed_dir"]))

    # 1. Umbral en validación
    val = manifest[manifest.split == "val"]
    probs, gts, fovs = [], [], []
    for _, row in val.iterrows():
        img, gt, fov = _read(row)
        probs.append(seg.predict_processed(img) * fov)
        gts.append(gt)
        fovs.append(fov)
    thr, val_dice = best_threshold(probs, gts, fovs) if len(val) else (0.5, float("nan"))
    del probs, gts, fovs
    print(f"Umbral elegido en validación: {thr} (Dice val {val_dice:.4f})")

    # 2. Test y externa
    rows = []
    keep: dict[str, tuple] = {}
    for split in splits:
        part = manifest[manifest.split == split]
        for i, (_, row) in enumerate(part.iterrows()):
            img, gt, fov = _read(row)
            prob = seg.predict_processed(img) * fov
            m = image_metrics(prob, gt, fov, thr)
            rows.append({"id": row["id"], "split": split, "dataset": row["dataset"], "category": row["category"],
                         **m})
            keep[row["id"]] = (img, gt, prob)
            if len(keep) > 40:  # limita la RAM: conserva solo lo necesario para figuras
                keep.pop(next(iter(keep)))
            if (i + 1) % 25 == 0:
                print(f"  {split}: {i + 1}/{len(part)}")
    per_image = pd.DataFrame(rows)
    per_image.to_csv(report_dir / "per_image.csv", index=False)

    results: dict[str, Any] = {"model": str(model_path), "threshold": thr, "val_dice_at_threshold": val_dice,
                               "tta": seg.tta, "resolution": seg.size, "splits": {}}
    for split, g in per_image.groupby("split"):
        results["splits"][split] = {
            "overall": summarize(g),
            "by_category": {c: summarize(gc) for c, gc in g.groupby("category")},
            "dataset": sorted(g["dataset"].unique().tolist()),
        }

    # 3. Figuras: los mejores y peores casos de cada división
    max_fig = int(config.get("evaluation", {}).get("max_figures", 12))
    for split, g in per_image.groupby("split"):
        g = g.sort_values("dice")
        chosen = pd.concat([g.head(max_fig // 4), g.tail(max_fig // 4)])
        for _, r in chosen.iterrows():
            if r["id"] not in keep:
                img, gt, fov = _read(manifest[manifest.id == r["id"]].iloc[0])
                keep[r["id"]] = (img, gt, seg.predict_processed(img) * fov)
            img, gt, prob = keep[r["id"]]
            save_figure(fig_dir / f"{split}_{r['id']}.png", img, gt, prob, thr,
                        f"{r['id']}  {r['category']}  Dice {r['dice']:.3f}  Se {r['sensitivity']:.3f}")

    (report_dir / "retina_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # El umbral queda con el modelo para la inferencia / API
    pp = Path(seg.model_path).with_name("preprocess.json")
    cfg = json.loads(pp.read_text()) if pp.exists() else {}
    cfg["threshold"] = thr
    pp.write_text(json.dumps(cfg, indent=2))
    seg.threshold = thr

    print_table(results)
    return results


def print_table(results: dict[str, Any]) -> None:
    """Tabla resumen en consola."""
    print(f"\n| División | Grupo | n | Dice | IoU | Se | Sp | AUC | HD95 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for split, s in results["splits"].items():
        for name, block in [("todas", s["overall"]), *s["by_category"].items()]:
            f = lambda k: f"{block[k]['mean']:.4f}" if k in block else "-"  # noqa: E731
            print(f"| {split} | {name} | {block['n']} | {f('dice')} | {f('iou')} | {f('sensitivity')} | "
                  f"{f('specificity')} | {f('auc')} | {block['hd95']['mean']:.1f} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/retina_unet.yaml")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(_p(a.config), a.override)
    from src.utils.seed import configure_gpu

    configure_gpu(False)
    evaluate(cfg, a.model)


if __name__ == "__main__":
    main()
