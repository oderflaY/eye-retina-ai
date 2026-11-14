"""Evalúa el detector sobre el split de test (o val) y genera el reporte.

Uso::

    python -m src.evaluation.evaluate_eye_grid --config configs/eye_grid.yaml
    python -m src.evaluation.evaluate_eye_grid --model models/eye_grid/<exp>/best.keras --split val

Salida en ``reports/eye_grid/<experimento>/``:

* ``metrics.json`` / ``metrics.md``  – tabla de resultados.
* ``pr_curve.png``, ``count_confusion.png``  – gráficas.
* ``explain/``  – original | Grad-CAM | Grad-CAM++ | predicción.
* ``errors/``   – peores falsos negativos y falsos positivos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data.encoding import Detection, sigmoid  # noqa: E402
from src.data.eye_dataset import load_letterboxed, load_processed  # noqa: E402
from src.deployment.predictor_grid import GridEyeDetector  # noqa: E402
from src.evaluation.eye_metrics import COUNT_LABELS, ImageResult, evaluate_detections, match  # noqa: E402
from src.evaluation.explain import grad_cam, grad_cam_pp, overlay  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.seed import configure_gpu, set_seed  # noqa: E402


def _draw(img: np.ndarray, gt: np.ndarray, dets: list[Detection], thr: float) -> np.ndarray:
    """Verde = verdad, rojo = predicción ≥ umbral, con landmarks."""
    vis = img.copy()
    for b in gt:
        x, y, w, h = (int(round(v)) for v in b)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 1)
    for d in dets:
        if d.score < thr:
            continue
        x, y, w, h = (int(round(v)) for v in d.box)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (230, 30, 30), 1)
        for px, py in d.landmarks:
            cv2.circle(vis, (int(px), int(py)), 1, (255, 220, 0), -1)
    return vis


def run_evaluation(
    cfg: dict[str, Any],
    model_path: Path,
    split: str = "test",
    out_dir: Path | None = None,
    n_explain: int = 8,
) -> dict[str, Any]:
    """Ejecuta la evaluación completa y escribe el reporte.

    Returns:
        Diccionario de métricas (el mismo que ``metrics.json``).
    """
    set_seed(int(cfg["seed"]))
    configure_gpu(False)
    icfg = cfg["inference"]
    det = GridEyeDetector(model_path, icfg["score_threshold"], icfg["iou_threshold"], icfg["max_detections"])
    samples, splits = load_processed(resolve_path(cfg["data"]["processed_dir"]))
    idx = splits.get(split) or []
    if not idx:
        raise ValueError(f"El split '{split}' está vacío")
    out_dir = out_dir or resolve_path(cfg["output"]["reports_dir"]) / cfg["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)

    size = det.image_size
    results: list[ImageResult] = []
    images: list[np.ndarray] = []
    cell_scores, cell_labels = [], []
    bs = 32
    for start in range(0, len(idx), bs):
        chunk = idx[start : start + bs]
        loaded = [load_letterboxed(samples[i], size, max_eyes=max(samples[i].num_eyes, 1)) for i in chunk]
        batch = np.stack([lb[0] for lb in loaded])
        preds = det.raw_predict(batch)
        for i, (img, boxes, lms, mask, n), pred in zip(chunk, loaded, preds):
            n = int(n)
            gt_lm = lms[:n].copy()
            gt_lm[mask[:n] == 0] = np.nan
            dets = det.decode(pred, score_threshold=0.05)  # umbral bajo para la curva PR
            results.append(ImageResult(boxes[:n], gt_lm, dets, Path(samples[i].image_path).name))
            images.append(img)
            target, _ = det.encoder.encode(boxes[:n], lms[:n], mask[:n] > 0)
            cell_scores.append(sigmoid(pred[..., 0]).ravel())
            cell_labels.append(target[..., 0].ravel())

    metrics = evaluate_detections(
        results, icfg["score_threshold"], 0.5, np.concatenate(cell_scores), np.concatenate(cell_labels)
    )
    metrics.update({"split": split, "model": str(model_path), "experiment": cfg["experiment"]})
    pr = metrics.pop("_pr_curve")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "metrics.md").write_text(metrics_markdown(metrics), encoding="utf-8")
    _plot_pr(pr, metrics, out_dir / "pr_curve.png")
    _plot_confusion(np.asarray(metrics["count_confusion_norm"]), out_dir / "count_confusion.png")
    _explain(det, results, images, out_dir / "explain", n_explain, icfg["score_threshold"])
    _error_analysis(results, images, out_dir / "errors", icfg["score_threshold"])
    print(metrics_markdown(metrics))
    print(f"Reporte en {out_dir}")
    return metrics


def metrics_markdown(m: dict[str, Any]) -> str:
    """Tabla Markdown con las métricas principales."""

    def f(v: Any) -> str:
        return (
            "—"
            if v is None or (isinstance(v, float) and np.isnan(v))
            else f"{v:.4f}" if isinstance(v, float) else str(v)
        )

    rows = [
        ("Imágenes", m["num_images"]),
        ("Ojos reales", m["num_gt_eyes"]),
        ("Precision", m["precision"]),
        ("Recall", m["recall"]),
        ("F1", m["f1"]),
        ("AP@0.5", m["ap50"]),
        ("IoU medio (TP)", m["mean_iou_tp"]),
        ("Accuracy de conteo", m["count_accuracy"]),
        ("AUC-ROC (celdas)", m["cell_auc_roc"]),
        ("NME landmarks (media)", m["landmark_nme_mean"]),
    ]
    lines = [
        f"## Detección de ojos — {m.get('experiment', '')} ({m.get('split', '')})",
        "",
        f"Umbral de puntaje {m['score_threshold']}, IoU {m['iou_threshold']}.",
        "",
        "| Métrica | Valor |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {f(v)} |" for k, v in rows]
    lines += ["", "| Landmark | NME |", "| --- | --- |"]
    lines += [f"| {k} | {f(v)} |" for k, v in m["landmark_nme"].items()]
    return "\n".join(lines) + "\n"


def _plot_pr(pr: dict[str, list[float]], m: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    if pr["recall"]:
        ax.plot(pr["recall"], pr["precision"], lw=1.5)
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title=f"Curva PR (AP@0.5 = {m['ap50']:.3f})" if m["ap50"] == m["ap50"] else "Curva PR",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
        xticklabels=COUNT_LABELS,
        yticklabels=COUNT_LABELS,
    )
    ax.set(xlabel="Ojos detectados", ylabel="Ojos reales", title="Conteo por imagen (normalizado)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _explain(
    det: GridEyeDetector, results: list[ImageResult], images: list[np.ndarray], out: Path, n: int, thr: float
) -> None:
    """Original | Grad-CAM | Grad-CAM++ | predicción, para ``n`` imágenes."""
    if det.model is None or not images:
        return
    out.mkdir(parents=True, exist_ok=True)
    sel = list(range(min(n, len(images))))
    batch = np.stack([images[i] for i in sel]).astype(np.float32)
    cams, campp = grad_cam(det.model, batch), grad_cam_pp(det.model, batch)
    for k, i in enumerate(sel):
        img = images[i]
        row = np.concatenate(
            [
                img,
                overlay(img, cams[k]),
                overlay(img, campp[k]),
                _draw(img, results[i].gt_boxes, results[i].dets, thr),
            ],
            axis=1,
        )
        cv2.imwrite(str(out / f"{k:02d}_{results[i].image_id}.jpg"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))


def _error_analysis(
    results: list[ImageResult], images: list[np.ndarray], out: Path, thr: float, top: int = 8
) -> None:
    """Guarda las imágenes con más falsos negativos y falsos positivos."""
    out.mkdir(parents=True, exist_ok=True)
    fn_rank, fp_rank = [], []
    for i, r in enumerate(results):
        ops = [d for d in r.dets if d.score >= thr]
        tp, _, _ = match(ops, r.gt_boxes)
        fn_rank.append((len(r.gt_boxes) - int(tp.sum()), i))
        fp_rank.append((sum(d.score for d, ok in zip(ops, tp) if not ok), i))
    summary = {"falsos_negativos": [], "falsos_positivos": []}
    for name, rank in (("falsos_negativos", fn_rank), ("falsos_positivos", fp_rank)):
        for val, i in sorted(rank, reverse=True)[:top]:
            if val <= 0:
                break
            cv2.imwrite(
                str(out / f"{name}_{results[i].image_id}.jpg"),
                cv2.cvtColor(_draw(images[i], results[i].gt_boxes, results[i].dets, thr), cv2.COLOR_RGB2BGR),
            )
            summary[name].append({"imagen": results[i].image_id, "valor": round(float(val), 3)})
    (out / "resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eye_grid.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--model", default=None, help="ruta al .keras o .tflite (por defecto best.keras)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.override)
    model = (
        Path(args.model)
        if args.model
        else (resolve_path(cfg["output"]["models_dir"]) / cfg["experiment"] / "best.keras")
    )
    run_evaluation(cfg, model, args.split)


if __name__ == "__main__":
    main()
