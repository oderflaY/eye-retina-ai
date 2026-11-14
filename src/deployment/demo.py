"""Demo interactiva con Gradio para el detector de ojos.

Uso::

    python src/deployment/demo.py                       # usa models/eye_best.h5
    EYE_MODEL_PATH=models/eye_savedmodel python src/deployment/demo.py

Requiere ``pip install gradio``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from src.config import PROJECT_ROOT

COLORS = [(255, 80, 80), (255, 200, 0), (80, 200, 255), (80, 255, 120), (255, 0, 255), (255, 255, 255)]


def draw_prediction(image: np.ndarray, pred: dict[str, Any], threshold: float) -> np.ndarray:
    """Dibuja caja, landmarks y confianza sobre una copia de la imagen RGB.

    Args:
        image: imagen RGB ``uint8``.
        pred: salida de ``EyeDetector.predict``.
        threshold: si la confianza es menor, solo se escribe "sin ojo".

    Returns:
        Imagen anotada.
    """
    vis = image.copy()
    t = max(1, round(max(image.shape[:2]) / 300))
    if pred["confidence"] < threshold:
        cv2.putText(
            vis,
            f"sin ojo (p={pred['confidence']:.2f})",
            (8, 24 * t),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * t,
            (255, 60, 60),
            t,
            cv2.LINE_AA,
        )
        return vis
    x, y, w, h = (int(round(v)) for v in pred["bbox"])
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), t)
    cv2.putText(
        vis,
        f"ojo {pred['confidence']:.2f}",
        (x, max(12, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5 * t,
        (0, 255, 0),
        t,
        cv2.LINE_AA,
    )
    for lm, c in zip(pred["landmarks"], COLORS):
        cv2.circle(vis, (int(lm["x"]), int(lm["y"])), 2 * t, c, -1)
    return vis


def build_demo(model_path: Path) -> Any:
    """Construye la interfaz de Gradio (sin lanzarla)."""
    import gradio as gr

    from src.deployment.inference_eye import EyeDetector

    det = EyeDetector(model_path)

    def run(image: np.ndarray | None, threshold: float) -> tuple[Any, dict[str, Any], str | None]:
        if image is None:
            return None, {"error": "Sube una imagen"}, None
        pred = det.predict(image)
        pred["is_eye"] = pred["confidence"] >= threshold
        vis = draw_prediction(image, pred, threshold)
        tmp = Path(tempfile.mkdtemp()) / "prediccion_ojo.png"
        cv2.imwrite(str(tmp), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        return vis, pred, str(tmp)

    with gr.Blocks(title="Eye-Retina AI · Detección de ojos") as demo:
        gr.Markdown("## Detección de ojos\nHerramienta de investigación; no es un dispositivo médico.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="Imagen")
                thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Umbral de confianza")
                btn = gr.Button("Detectar", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Resultado")
                out_json = gr.JSON(label="Predicción")
                out_file = gr.File(label="Descargar resultado")
        btn.click(run, [inp, thr], [out_img, out_json, out_file])
        thr.release(run, [inp, thr], [out_img, out_json, out_file])
    return demo


def main() -> None:
    """Lanza la demo en http://127.0.0.1:7860."""
    path = Path(os.environ.get("EYE_MODEL_PATH", "models/eye_best.h5"))
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if not path.exists():
        sys.exit(f"No existe el modelo {path}. Entrena primero: python src/training/train_eye.py")
    build_demo(path).launch()


if __name__ == "__main__":
    main()
