"""Entrena el detector de ojos de regresión directa.

Uso::

    python src/training/train_eye.py --config configs/eye.yaml
    python src/training/train_eye.py --config configs/eye.yaml --override train.epochs=5 train.batch_size=16

Flujo:

1. ``EyeDataLoader``: metadatos → división 70/15/15 por imagen → ``tf.data``.
2. Calentamiento (``warmup_epochs``) con el backbone congelado.
3. Fine-tuning de las últimas ``unfreeze_last`` capas con ``finetune_lr``.
4. Evaluación en test → ``reports/eye_results.json`` + figuras.
5. Exportación a SavedModel (para TF Serving / ``EyeDetector``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tensorflow as tf
import yaml

from src.config import PROJECT_ROOT, enable_mixed_precision_if_gpu, set_global_seed
from src.data.augment import get_eye_augmentation_pipeline
from src.data.loader_eyes import EyeDataLoader
from src.models.eye_detector import build_eye_detector, compile_eye_detector, set_trainable_backbone
from src.utils.config import load_config


def _p(path: str | Path) -> Path:
    """Ruta absoluta respecto a la raíz del proyecto."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def make_callbacks(
    config: dict[str, Any], stage: str, best_so_far: float | None = None
) -> list[tf.keras.callbacks.Callback]:
    """ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau y CSVLogger.

    Args:
        config: configuración completa.
        stage: nombre de la etapa (subcarpeta de TensorBoard).
        best_so_far: mejor ``val_loss`` de la etapa anterior; el checkpoint solo
            se sobrescribe si lo mejora.
    """
    tcfg, ocfg = config["train"], config["output"]
    ckpt = _p(ocfg["checkpoint"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rl = tcfg.get("reduce_lr", {})
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_loss", save_best_only=True, initial_value_threshold=best_so_far, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(tcfg.get("early_stopping_patience", 10)),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(_p(ocfg["log_dir"]) / time.strftime("%Y%m%d-%H%M%S") / stage)
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(rl.get("factor", 0.5)),
            patience=int(rl.get("patience", 5)),
            min_lr=float(rl.get("min_lr", 1e-6)),
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(ckpt.with_name(f"eye_history_{stage}.csv"))),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def train_eye_detector(config: dict[str, Any]) -> dict[str, Any]:
    """Entrena, evalúa en test y exporta el detector.

    Args:
        config: diccionario cargado de ``configs/eye.yaml``.

    Returns:
        Métricas finales en test (las mismas que ``reports/eye_results.json``).
    """
    from src.evaluation.evaluate_eye import evaluate_model, save_evaluation_figures

    set_global_seed(int(config.get("seed", 42)))
    mixed = bool(config["train"].get("mixed_precision", True)) and enable_mixed_precision_if_gpu()
    if not mixed:
        tf.keras.mixed_precision.set_global_policy("float32")
    dcfg, tcfg, mcfg, ocfg = config["data"], config["train"], config["model"], config["output"]
    img_size = tuple(dcfg.get("img_size", [224, 224]))

    # ---- Datos ----------------------------------------------------------------
    acfg = config.get("augmentation", {})
    pipeline = get_eye_augmentation_pipeline(
        rotate_limit=acfg.get("rotate_limit", 15),
        hflip_p=acfg.get("hflip_p", 0.5),
        brightness_contrast=acfg.get("brightness_contrast", 0.2),
        translate=acfg.get("translate", 0.05),
        zoom=acfg.get("zoom", 0.1),
    )
    loader = EyeDataLoader(
        _p(dcfg["data_dir"]),
        img_size,
        int(tcfg["batch_size"]),
        label_format=dcfg.get("label_format", "auto"),
        seed=int(config.get("seed", 42)),
        crop_scale=float(dcfg.get("crop_scale", 2.5)),
        augment_pipeline=pipeline,
        group_prefix=bool(dcfg.get("group_prefix", False)),
    )
    meta = loader.load_metadata()
    train_df, val_df, test_df = loader.split(meta, dcfg.get("test_size", 0.15), dcfg.get("val_size", 0.15))
    if val_df.empty or train_df.empty:
        raise ValueError(f"Datos insuficientes: train={len(train_df)} val={len(val_df)} filas")
    cache = bool(dcfg.get("cache", True))
    train_ds = loader.build_tfdataset(
        train_df, augment=bool(acfg.get("enabled", True)), shuffle=True, cache=cache
    )
    val_ds = loader.build_tfdataset(val_df, cache=cache)
    print(f"Filas -> train {len(train_df)} · val {len(val_df)} · test {len(test_df)} | mixed_float16={mixed}")

    # ---- Modelo ---------------------------------------------------------------
    model = build_eye_detector(
        (*img_size, 3), int(mcfg.get("num_landmarks", 6)), mcfg.get("weights", "imagenet")
    )
    epochs = int(tcfg.get("epochs", 50))
    warmup = min(int(tcfg.get("warmup_epochs", 0)), epochs)
    lw = tcfg.get("loss_weights")
    history: dict[str, Any] = {}

    if warmup:
        set_trainable_backbone(model, 0)
        compile_eye_detector(model, float(tcfg.get("lr", 1e-3)), lw, bool(tcfg.get("jit_compile", False)))
        h = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup,
            callbacks=make_callbacks(config, "warmup"),
            verbose=2,
        )
        history["warmup"] = h.history
    if epochs > warmup:
        n = set_trainable_backbone(model, int(tcfg.get("unfreeze_last", 50)) if warmup else 10_000)
        print(f"Fine-tuning con {n} capas del backbone entrenables")
        compile_eye_detector(model, float(tcfg.get("finetune_lr", 1e-4)), lw, bool(tcfg.get("jit_compile", False)))
        best = min(history["warmup"]["val_loss"]) if warmup else None
        h = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            initial_epoch=warmup,
            callbacks=make_callbacks(config, "finetune", best),
            verbose=2,
        )
        history["finetune"] = h.history

    # EarlyStopping deja en memoria los mejores pesos de la última etapa; el
    # checkpoint guarda el mejor global. Se evalúa con el checkpoint.
    ckpt = _p(ocfg["checkpoint"])
    if ckpt.exists():
        from src.deployment.inference_eye import load_keras_model

        model = load_keras_model(ckpt)

    # ---- Evaluación en test ----------------------------------------------------
    results: dict[str, Any] = {
        "experiment": config.get("experiment"),
        "mixed_precision": mixed,
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "epochs_run": {k: len(v["loss"]) for k, v in history.items()},
        "best_val_loss": {k: float(min(v["val_loss"])) for k, v in history.items()},
    }
    if not test_df.empty:
        test_ds = loader.build_tfdataset(test_df, cache=False)
        metrics, preds = evaluate_model(model, test_ds)
        results["test"] = metrics
        save_evaluation_figures(model, test_ds, preds, test_df, _p(ocfg["figures_dir"]))
    else:
        results["test"] = "sin datos de test"

    # ---- Exportación -----------------------------------------------------------
    sm_dir = _p(ocfg["savedmodel_dir"])
    try:
        export_float32_savedmodel(model, sm_dir, img_size, int(mcfg.get("num_landmarks", 6)))
        results["savedmodel"] = str(sm_dir)
    except Exception as exc:  # noqa: BLE001 - el entrenamiento ya terminó; se reporta y sigue
        results["savedmodel_error"] = repr(exc)

    report = _p(ocfg["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (report.with_name("eye_history.json")).write_text(json.dumps(history, default=float), encoding="utf-8")
    (report.with_name("eye_config_used.yaml")).write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def export_float32_savedmodel(
    model: tf.keras.Model, out_dir: Path, img_size: tuple[int, int], num_landmarks: int
) -> Path:
    """Exporta un SavedModel en float32 (sirve en CPU y para TFLite).

    Args:
        model: modelo entrenado (puede tener política ``mixed_float16``).
        out_dir: carpeta de salida (se sobrescribe).
        img_size: (alto, ancho) de entrada.
        num_landmarks: landmarks por ojo.
    """
    import shutil

    prev = tf.keras.mixed_precision.global_policy().name
    tf.keras.mixed_precision.set_global_policy("float32")
    try:
        clean = build_eye_detector((*img_size, 3), num_landmarks, weights=None)
        clean.set_weights(model.get_weights())
    finally:
        tf.keras.mixed_precision.set_global_policy(prev)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if hasattr(clean, "export"):
        clean.export(str(out_dir))  # Keras 3
    else:
        tf.saved_model.save(clean, str(out_dir))  # Keras 2
    return out_dir


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eye.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="clave=valor (p. ej. train.epochs=5)")
    args = ap.parse_args()
    train_eye_detector(load_config(_p(args.config), args.override))


if __name__ == "__main__":
    main()
