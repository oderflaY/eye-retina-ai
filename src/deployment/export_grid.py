"""Exporta el detector a SavedModel (TF Serving) y TFLite (móvil/edge).

Uso::

    python -m src.deployment.export_grid --model models/eye_grid/<exp>/best.keras
    python -m src.deployment.export_grid --model ... --fp16      # TFLite cuantizado a float16

Verifica que la salida TFLite coincida con la de Keras sobre una imagen aleatoria.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.deployment.predictor_grid import load_float32_model


def export_model(model_path: Path, out_dir: Path | None = None, fp16: bool = False) -> dict[str, str | float]:
    """Exporta y devuelve rutas + diferencia máxima Keras vs TFLite."""
    model, meta = load_float32_model(model_path)
    out_dir = out_dir or model_path.parent / "export"
    saved_dir = out_dir / "saved_model" / "1"  # carpeta versionada para TF Serving
    if saved_dir.exists():
        shutil.rmtree(saved_dir)
    saved_dir.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "export"):  # Keras 3
        model.export(str(saved_dir))
    else:  # Keras 2 (TF 2.15)
        tf.saved_model.save(model, str(saved_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    if fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()
    tflite_path = out_dir / ("eye_grid_fp16.tflite" if fp16 else "eye_grid.tflite")
    tflite_path.write_bytes(tflite_bytes)
    (out_dir / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # Verificación numérica
    size = int(meta["image_size"])
    x = np.random.default_rng(0).uniform(0, 255, (1, size, size, 3)).astype(np.float32)
    ref = model(x, training=False).numpy()
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    got = interp.get_tensor(interp.get_output_details()[0]["index"])
    diff = float(np.max(np.abs(ref - got)))
    info = {
        "saved_model": str(saved_dir),
        "tflite": str(tflite_path),
        "tflite_mb": round(len(tflite_bytes) / 1e6, 2),
        "max_abs_diff_keras_vs_tflite": diff,
    }
    print(json.dumps(info, indent=2))
    return info


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="ruta al .keras entrenado")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    export_model(Path(args.model), Path(args.out) if args.out else None, args.fp16)


if __name__ == "__main__":
    main()
