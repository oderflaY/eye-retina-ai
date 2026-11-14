"""Entrena la U-Net de vasos de retina y la evalúa.

Uso::

    python -m src.data.prepare_retina                       # una vez
    python src/training/train_retina.py --config configs/retina_unet.yaml
    python src/training/train_retina.py --override model.encoder=plain model.weights=null train.epochs=5

Flujo:

1. ``manifest.csv`` → ``tf.data`` (recortes de 512 en train, imagen completa en val).
2. Calentamiento (``warmup_epochs``) con el encoder congelado.
3. Fine-tuning de todo el modelo (BatchNorm del encoder congeladas) con
   ``finetune_lr``; checkpoint por mejor ``val_dice``.
4. ``evaluate_retina``: umbral en val, test FIVES, externa HRF, figuras.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tensorflow as tf
import yaml

from src.config import PROJECT_ROOT
from src.data.loader_retina import make_dataset, read_manifest
from src.models.seg_losses import DiceAtThreshold, DiceBCELoss
from src.models.unet import build_unet, set_encoder_trainable
from src.utils.config import load_config
from src.utils.seed import configure_gpu, set_seed


def _p(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def compile_unet(model: tf.keras.Model, lr: float, loss_cfg: dict[str, float]) -> None:
    """Adam + Dice/BCE + Dice duro a 0.5."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=DiceBCELoss(**loss_cfg),
        metrics=[DiceAtThreshold(0.5, name="dice")],
        # XLA falla con el gradiente de UpSampling2D bilineal en mixed_float16
        # ("Seen floating point types of different precisions ... ReluGrad").
        jit_compile=False,
    )


def callbacks(config: dict[str, Any], stage: str, run: str, best: float | None) -> list[tf.keras.callbacks.Callback]:
    """Checkpoint por ``val_dice``, early stopping, ReduceLR, TensorBoard y CSV."""
    t, o = config["train"], config["output"]
    out = _p(o["dir"])
    rl = t.get("reduce_lr", {})
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(str(out / "best.keras"), monitor="val_dice", mode="max",
                                           save_best_only=True, initial_value_threshold=best, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=str(_p(o["log_dir"]) / run / stage)),
        tf.keras.callbacks.CSVLogger(str(out / f"history_{stage}.csv"), append=bool(t.get("resume_from"))),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    if stage == "finetune":
        cbs += [
            tf.keras.callbacks.EarlyStopping(monitor="val_dice", mode="max", restore_best_weights=True,
                                             patience=int(t.get("early_stopping_patience", 8)), verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_dice", mode="max", factor=float(rl.get("factor", 0.5)),
                                                 patience=int(rl.get("patience", 3)),
                                                 min_lr=float(rl.get("min_lr", 1e-6)), verbose=1),
        ]
    return cbs


def train(config: dict[str, Any]) -> dict[str, Any]:
    """Entrena y evalúa. Devuelve el resumen de la evaluación."""
    set_seed(int(config.get("seed", 42)))
    t, d, m, o = config["train"], config["data"], config["model"], config["output"]
    mixed = configure_gpu(bool(t.get("mixed_precision", True)))
    out = _p(o["dir"])
    out.mkdir(parents=True, exist_ok=True)
    run = time.strftime("%Y%m%d-%H%M%S")

    root = _p(d["processed_dir"])
    tr_df, va_df = read_manifest(root, "train"), read_manifest(root, "val")
    if tr_df.empty or va_df.empty:
        raise SystemExit(f"Sin datos en {root}. Corre antes: python -m src.data.prepare_retina")
    train_ds = make_dataset(tr_df, int(t["batch_size"]), True, int(d["crop"]), int(d.get("crops_per_image", 1)),
                            float(d.get("color_aug", 0.15)), int(config.get("seed", 42)))
    val_ds = make_dataset(va_df, int(t.get("val_batch_size", 1)), False)
    steps = t.get("steps_per_epoch")
    if steps:
        train_ds = train_ds.repeat()
    print(f"train {len(tr_df)} imágenes × {d.get('crops_per_image', 1)} recortes · val {len(va_df)} · "
          f"mixed precision: {mixed}")

    model = build_unet(m["encoder"], m.get("weights"), dropout=float(m.get("dropout", 0.1)))
    print(f"{model.name}: {model.count_params() / 1e6:.1f} M parámetros")

    # preprocess.json: lo que la inferencia necesita para reproducir la entrada
    prep = json.loads((root / "prepare_config.json").read_text())
    (out / "preprocess.json").write_text(json.dumps(
        {"size": prep["size"], "clahe": prep["clahe"], "threshold": 0.5, "encoder": m["encoder"],
         "experiment": config.get("experiment")}, indent=2))
    (out / "config_used.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    histories: dict[str, Any] = {}
    best = None
    warm = int(t.get("warmup_epochs", 0)) if m["encoder"] != "plain" and m.get("weights") else 0
    loss_cfg = {k: float(v) for k, v in t.get("loss", {}).items()}
    start = warm
    finetune_lr = float(t["finetune_lr"] if warm else t["lr"])
    resume = t.get("resume_from")
    if resume:
        # Reanudar: pesos del checkpoint, sin calentamiento, directo al fine-tuning
        model.load_weights(str(_p(resume)))
        start = int(t.get("resume_epoch", 0))
        best = t.get("resume_best")
        best = float(best) if best is not None else None
        finetune_lr = float(t["finetune_lr"])
        warm = 0
        print(f"Reanudando desde {resume} en la época {start + 1} (mejor val_dice previo: {best})")
    if warm:
        set_encoder_trainable(model, False)
        compile_unet(model, float(t["lr"]), loss_cfg)
        h = model.fit(train_ds, validation_data=val_ds, epochs=warm, steps_per_epoch=steps,
                      callbacks=callbacks(config, "warmup", run, None), verbose=2)
        histories["warmup"] = h.history
        best = max(h.history["val_dice"])

    n = set_encoder_trainable(model, True)
    print(f"Fine-tuning: {n} capas del encoder entrenables (BatchNorm congeladas)")
    compile_unet(model, finetune_lr, loss_cfg)
    h = model.fit(train_ds, validation_data=val_ds, epochs=int(t["epochs"]), initial_epoch=start,
                  steps_per_epoch=steps, callbacks=callbacks(config, "finetune", run, best), verbose=2)
    histories["finetune"] = h.history
    (out / "history.json").write_text(json.dumps(
        {k: {m_: [float(x) for x in v] for m_, v in hist.items()} for k, hist in histories.items()}, indent=2))

    report = _p(o["report_dir"])
    report.mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "config_used.yaml", report / "config_used.yaml")

    # La evaluación corre en otro proceso: TF no libera la memoria de la GPU del
    # entrenamiento y la inferencia a 1024 px no cabe al lado (6 GB).
    cfg_path = out / "config_used.yaml"
    subprocess.run([sys.executable, str(PROJECT_ROOT / "src/evaluation/evaluate_retina.py"),
                    "--config", str(cfg_path), "--model", str(out / "best.keras")], check=True)
    return json.loads((report / "retina_results.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/retina_unet.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    train(load_config(_p(a.config), a.override))


if __name__ == "__main__":
    main()
