"""Entrenamiento del detector de ojos con fine-tuning progresivo.

Uso::

    python -m src.training.train_eye_grid --config configs/eye_grid.yaml
    python -m src.training.train_eye_grid --config configs/eye_grid.yaml --fold 0
    python -m src.training.train_eye_grid --override train.epochs=5 train.batch_size=8

Etapas:

1. **Calentamiento** (``warmup_epochs``): backbone congelado, solo se entrena la
   cabeza con ``lr``.
2. **Fine-tuning**: se descongelan las últimas ``unfreeze_last`` capas del
   backbone con ``finetune_lr`` y CosineDecay (o ReduceLROnPlateau).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml

from src.data.eye_dataset import EyeDatasetBuilder, load_processed
from src.models.eye_grid_detector import build_grid_eye_detector, model_meta, set_backbone_trainable
from src.models.losses import EyeDetectionLoss, make_metrics
from src.utils.config import load_config, resolve_path
from src.utils.seed import configure_gpu, set_seed


def make_optimizer(
    name: str, lr: float | tf.keras.optimizers.schedules.LearningRateSchedule, weight_decay: float
) -> tf.keras.optimizers.Optimizer:
    """Adam o AdamW (con clipnorm para estabilidad en mixed precision)."""
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay, clipnorm=10.0)
    return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=10.0)


def make_callbacks(
    cfg: dict[str, Any],
    out_dir: Path,
    log_dir: Path,
    stage: str,
    use_plateau: bool,
    best_so_far: float | None = None,
) -> list[tf.keras.callbacks.Callback]:
    """ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau y W&B.

    Args:
        best_so_far: mejor ``val_loss`` de la etapa anterior; el checkpoint de
            esta etapa solo sobrescribe ``best.keras`` si lo mejora.
    """
    tcfg = cfg["train"]
    cbs: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            initial_value_threshold=best_so_far,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(tcfg["early_stopping_patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir / stage)),
        tf.keras.callbacks.CSVLogger(str(out_dir / f"history_{stage}.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    if use_plateau:
        cbs.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
            )
        )
    if tcfg.get("wandb"):
        try:
            import wandb
            from wandb.integration.keras import WandbMetricsLogger

            if wandb.run is None:
                wandb.init(
                    project=tcfg.get("wandb_project", "eye-retina-ai"), name=cfg["experiment"], config=cfg
                )
            cbs.append(WandbMetricsLogger())
        except ImportError:
            print("[aviso] wandb no está instalado; se omite WandbMetricsLogger")
    return cbs


def compile_model(model: tf.keras.Model, cfg: dict[str, Any], lr: Any) -> None:
    """Compila con la pérdida combinada y métricas por término."""
    lcfg = cfg["loss"]
    loss = EyeDetectionLoss(
        obj_loss=lcfg.get("obj_loss", "bce"),
        pos_weight=float(lcfg.get("pos_weight", 10.0)),
        obj_weight=float(lcfg.get("obj_weight", 1.0)),
        box_weight=float(lcfg.get("box_weight", 5.0)),
        landmark_weight=float(lcfg.get("landmark_weight", 2.0)),
        huber_delta=float(lcfg.get("huber_delta", 0.1)),
    )
    tcfg = cfg["train"]
    model.compile(
        optimizer=make_optimizer(tcfg.get("optimizer", "adamw"), lr, float(tcfg.get("weight_decay", 1e-4))),
        loss=loss,
        metrics=make_metrics(lcfg),
    )


def train(cfg: dict[str, Any], fold: int | None = None) -> dict[str, Any]:
    """Entrena y devuelve un resumen (rutas y mejores pérdidas).

    Args:
        cfg: configuración completa.
        fold: índice de fold si se usa K-Fold (``data.cv_folds > 1``).
    """
    set_seed(int(cfg["seed"]), bool(cfg.get("deterministic", False)))
    mixed = configure_gpu(bool(cfg["train"].get("mixed_precision", True)))
    tcfg, ocfg = cfg["train"], cfg["output"]

    samples, splits = load_processed(resolve_path(cfg["data"]["processed_dir"]))
    if fold is not None:
        split = splits["folds"][fold]
        tr_idx, va_idx = split["train"], split["val"]
    else:
        tr_idx, va_idx = splits["train"], splits["val"]
    if not va_idx:
        raise ValueError("El split de validación está vacío; revisa data.split")

    suffix = f"_fold{fold}" if fold is not None else ""
    out_dir = resolve_path(ocfg["models_dir"]) / f"{cfg['experiment']}{suffix}"
    log_dir = resolve_path(ocfg["logs_dir"]) / f"{cfg['experiment']}{suffix}" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (out_dir / "model_meta.json").write_text(json.dumps(model_meta(cfg)), encoding="utf-8")

    builder = EyeDatasetBuilder(samples, cfg)
    train_ds = builder.build(tr_idx, training=True)
    val_ds = builder.build(va_idx, training=False)

    mcfg = cfg["model"]
    model = build_grid_eye_detector(
        image_size=int(cfg["data"]["image_size"]),
        backbone=mcfg["backbone"],
        weights=mcfg.get("weights"),
        head_channels=int(mcfg.get("head_channels", 128)),
    )
    print(f"Modelo: {model.name} | parámetros: {model.count_params():,} | mixed_float16: {mixed}")
    print(f"Imágenes -> train: {len(tr_idx)}  val: {len(va_idx)}")

    epochs = int(tcfg["epochs"])
    warmup = min(int(tcfg.get("warmup_epochs", 10)), epochs)
    use_plateau = tcfg.get("scheduler", "cosine") == "plateau"
    histories: dict[str, Any] = {}

    # ---- Etapa 1: backbone congelado ---------------------------------------
    if warmup > 0:
        set_backbone_trainable(model, 0)
        compile_model(model, cfg, float(tcfg["lr"]))
        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup,
            callbacks=make_callbacks(cfg, out_dir, log_dir, "warmup", use_plateau),
            verbose=2,
        )
        histories["warmup"] = h1.history

    # ---- Etapa 2: fine-tuning de las últimas N capas -----------------------
    if epochs > warmup:
        n = set_backbone_trainable(model, int(tcfg.get("unfreeze_last", 50)))
        print(f"Fine-tuning: {n} capas del backbone entrenables")
        ft_lr: Any = float(tcfg["finetune_lr"])
        if not use_plateau:
            steps = max(1, math.ceil(len(tr_idx) / int(tcfg["batch_size"]))) * (epochs - warmup)
            ft_lr = tf.keras.optimizers.schedules.CosineDecay(ft_lr, decay_steps=steps, alpha=0.01)
        compile_model(model, cfg, ft_lr)
        warm_losses = histories.get("warmup", {}).get("val_loss", [])
        best_warmup = float(min(warm_losses)) if warm_losses else None
        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            initial_epoch=warmup,
            callbacks=make_callbacks(cfg, out_dir, log_dir, "finetune", use_plateau, best_warmup),
            verbose=2,
        )
        histories["finetune"] = h2.history

    # EarlyStopping restaura los mejores pesos de la última etapa; se guardan
    # también como ``final.keras`` para reproducibilidad.
    model.save(out_dir / "final.keras")
    if not (out_dir / "best.keras").exists():
        shutil.copy(out_dir / "final.keras", out_dir / "best.keras")

    def _best(h: dict[str, list[float]]) -> float | None:
        return float(min(h["val_loss"])) if h.get("val_loss") else None

    summary = {
        "experiment": cfg["experiment"],
        "fold": fold,
        "model_dir": str(out_dir),
        "best_val_loss": {k: _best(v) for k, v in histories.items()},
        "epochs_run": {k: len(v.get("loss", [])) for k, v in histories.items()},
        "grid_collisions_seen": builder.collisions,
        "mixed_precision": mixed,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(histories, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eye_grid.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="clave=valor (p. ej. train.epochs=5)")
    ap.add_argument("--fold", type=int, default=None, help="fold de K-Fold a entrenar")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.override)
    train(cfg, args.fold)


if __name__ == "__main__":
    main()
