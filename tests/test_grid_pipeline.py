"""Prueba de extremo a extremo con TensorFlow sobre datos sintéticos.

prepare -> tf.data -> modelo -> entrenamiento corto -> evaluación ->
exportación SavedModel/TFLite -> predictor.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

tf = pytest.importorskip("tensorflow")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Dataset sintético + config apuntando a carpetas temporales."""
    from src.data.synthetic import generate

    base = tmp_path_factory.mktemp("e2e")
    generate(base / "raw", n=40, seed=7)
    cfg = yaml.safe_load((ROOT / "configs" / "eye_grid.yaml").read_text())
    cfg["experiment"] = "test_e2e"
    cfg["data"].update(
        raw_dir=str(base / "raw"),
        processed_dir=str(base / "processed"),
        augmented_dir=str(base / "aug"),
        image_size=128,
    )
    cfg["model"]["weights"] = None  # sin descargar pesos en las pruebas
    cfg["train"].update(epochs=2, warmup_epochs=1, batch_size=8, mixed_precision=False)
    cfg["output"] = {k: str(base / k) for k in ("models_dir", "logs_dir", "reports_dir")}
    cfg_path = base / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    subprocess.run(
        [sys.executable, "-m", "src.data.prepare_eyes", "--config", str(cfg_path), "--preview", "3"],
        cwd=ROOT,
        check=True,
    )
    return base, cfg_path


def test_model_shapes() -> None:
    from src.models.eye_grid_detector import build_grid_eye_detector, set_backbone_trainable

    m = build_grid_eye_detector(128, "mobilenetv2", None, 32)
    assert m.output_shape == (None, 8, 8, 17)
    assert set_backbone_trainable(m, 0) == 0 and set_backbone_trainable(m, 20) > 0


def test_loss_finite_and_masks() -> None:
    from src.models.losses import EyeDetectionLoss

    y_true = np.zeros((2, 8, 8, 23), np.float32)
    y_true[0, 3, 3, :5] = [1, 0.5, 0.5, 0.3, 0.2]
    y_true[0, 3, 3, 17:19] = 1  # solo 2 landmarks anotados
    y_pred = np.random.default_rng(0).normal(size=(2, 8, 8, 17)).astype(np.float32)
    loss = EyeDetectionLoss()(y_true, y_pred)
    assert np.isfinite(float(loss))
    # Cambiar landmarks NO anotados no debe cambiar la pérdida
    y_pred2 = y_pred.copy()
    y_pred2[0, 3, 3, 9:17] += 5
    assert np.isclose(float(EyeDetectionLoss()(y_true, y_pred2)), float(loss))


def test_end_to_end(workdir: tuple[Path, Path]) -> None:
    base, cfg_path = workdir
    assert (base / "processed" / "splits.json").exists()
    assert len(list((base / "aug").glob("*.jpg"))) == 3

    from src.training.train_eye_grid import train
    from src.utils.config import load_config

    cfg = load_config(cfg_path)
    summary = train(cfg)
    model_path = Path(summary["model_dir"]) / "best.keras"
    assert model_path.exists()

    from src.evaluation.evaluate_eye_grid import run_evaluation

    metrics = run_evaluation(cfg, model_path, "test", base / "report", n_explain=2)
    for key in ("precision", "recall", "f1", "ap50", "count_accuracy"):
        assert key in metrics
    assert (base / "report" / "pr_curve.png").exists()
    assert len(list((base / "report" / "explain").glob("*.jpg"))) == 2

    from src.deployment.export_grid import export_model

    info = export_model(model_path, base / "export")
    assert info["max_abs_diff_keras_vs_tflite"] < 1e-3

    from src.deployment.predictor_grid import GridEyeDetector

    img = cv2.cvtColor(cv2.imread(str(next((base / "raw" / "images").glob("*.png")))), cv2.COLOR_BGR2RGB)
    for path in (model_path, Path(info["tflite"])):
        out = GridEyeDetector(path, score_threshold=0.0, max_detections=3).predict(img)
        assert len(out) <= 3 and all(len(d["landmarks"]) == 6 for d in out)
