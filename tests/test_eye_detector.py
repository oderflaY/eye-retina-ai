"""Modelo, entrenamiento corto, evaluación, inferencia y API del detector de un ojo."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import yaml

tf = pytest.importorskip("tensorflow")

ROOT = Path(__file__).resolve().parents[1]


def test_output_shapes() -> None:
    from src.models.eye_detector import build_eye_detector, set_trainable_backbone

    m = build_eye_detector((96, 96, 3), 6, weights=None)
    out = m(np.zeros((2, 96, 96, 3), np.float32))
    assert (
        out["bbox"].shape == (2, 4) and out["landmarks"].shape == (2, 12) and out["eye_prob"].shape == (2, 1)
    )
    assert set_trainable_backbone(m, 0) == 0 and set_trainable_backbone(m, 20) > 0


def test_masked_huber_ignores_missing() -> None:
    from src.models.eye_detector import MaskedHuber

    y_true = np.zeros((1, 18), np.float32)
    y_true[0, 12:14] = 1  # solo los 2 primeros landmarks anotados
    p1 = np.zeros((1, 12), np.float32)
    p2 = p1.copy()
    p2[0, 4:] = 3.0  # cambia solo landmarks NO anotados
    loss = MaskedHuber()
    assert float(loss(y_true, p1)) == pytest.approx(float(loss(y_true, p2)))


@pytest.fixture(scope="module")
def trained(synthetic_eyes: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    """Entrena 2 épocas sobre datos sintéticos (pesos aleatorios, 96 px)."""
    from src.training.train_eye import train_eye_detector

    out = tmp_path_factory.mktemp("train")
    cfg = yaml.safe_load((ROOT / "configs" / "eye.yaml").read_text())
    cfg["data"].update(data_dir=str(synthetic_eyes), img_size=[96, 96])
    cfg["model"]["weights"] = None
    cfg["train"].update(epochs=2, warmup_epochs=1, batch_size=8, mixed_precision=False)
    cfg["output"] = {
        "checkpoint": str(out / "eye_best.h5"),
        "savedmodel_dir": str(out / "sm"),
        "log_dir": str(out / "logs"),
        "report": str(out / "eye_results.json"),
        "figures_dir": str(out / "fig"),
    }
    return train_eye_detector(cfg), out


def test_training_report(trained: tuple[dict, Path]) -> None:
    results, out = trained
    assert (out / "eye_best.h5").exists() and (out / "eye_results.json").exists()
    assert "classification" in results["test"]
    for f in (
        "confusion_matrix.png",
        "pr_curve.png",
        "predicciones.png",
        "gradcam.png",
        "peores_predicciones.csv",
    ):
        assert (out / "fig" / f).exists(), f
    assert "savedmodel" in results, results.get("savedmodel_error")


def test_inference_h5_and_savedmodel(trained: tuple[dict, Path]) -> None:
    from src.deployment.inference_eye import EyeDetector

    _, out = trained
    img = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    a = EyeDetector(out / "eye_best.h5").predict(img)
    b = EyeDetector(out / "sm").predict(img)
    assert set(a) == {"bbox", "landmarks", "confidence", "is_eye"} and len(a["landmarks"]) == 6
    assert a["confidence"] == pytest.approx(b["confidence"], abs=1e-3)
    batch = EyeDetector(out / "eye_best.h5").predict_batch([img, img[:50, :80]])
    assert len(batch) == 2


def test_api(trained: tuple[dict, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2
    from fastapi.testclient import TestClient

    import src.deployment.api as api

    _, out = trained
    monkeypatch.setenv("EYE_MODEL_PATH", str(out / "eye_best.h5"))
    api.get_eye_detector.cache_clear()
    client = TestClient(api.app)
    assert client.get("/").json()["status"] == "ok"
    img = np.zeros((50, 70, 3), np.uint8)
    buf = cv2.imencode(".png", img)[1].tobytes()
    r = client.post("/predict/eye", files={"file": ("a.png", io.BytesIO(buf), "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["image_size"] == [70, 50]
    bad = client.post("/predict/eye", files={"file": ("a.png", io.BytesIO(b"no soy imagen"), "image/png")})
    assert bad.status_code == 400 and "error" in bad.json()
    assert client.post("/predict/disease", files={"file": ("a.png", io.BytesIO(buf))}).status_code == 501


def test_gradcam_under_mixed_precision() -> None:
    """Regresión: con mixed_float16 el gradiente no debe ser None."""
    from src.evaluation.evaluate_eye import compute_gradcam
    from src.models.eye_detector import build_eye_detector

    prev = tf.keras.mixed_precision.global_policy().name
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        m = build_eye_detector((64, 64, 3), 6, weights=None)
        cam = compute_gradcam(m, np.random.default_rng(0).random((64, 64, 3)).astype(np.float32))
    finally:
        tf.keras.mixed_precision.set_global_policy(prev)
    assert cam.shape == (64, 64) and np.isfinite(cam).all()
