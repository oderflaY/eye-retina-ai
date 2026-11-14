"""Parte B: preprocesado, manifiesto, U-Net, pérdidas, métricas, entrenamiento corto, inferencia y API.

Usa fondos de ojo sintéticos (círculo rojizo con "vasos" oscuros); no descarga nada.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


def synthetic_fundus(h: int = 300, w: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Imagen RGB con FOV circular y máscara de vasos (líneas)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    c, r = (w // 2, h // 2), int(min(h, w) * 0.45)
    cv2.circle(img, c, r, (180, 80, 40), -1)
    ves = np.zeros((h, w), np.uint8)
    for _ in range(8):
        a = rng.uniform(0, 2 * np.pi)
        p2 = (int(c[0] + r * np.cos(a)), int(c[1] + r * np.sin(a)))
        cv2.line(ves, c, p2, 255, int(rng.integers(2, 5)))
    fov = np.zeros((h, w), np.uint8)
    cv2.circle(fov, c, r, 1, -1)
    ves = ves * fov
    img[ves > 0] = (90, 30, 20)
    return img, ves


# --------------------------------------------------------------------------- #
# Preprocesado
# --------------------------------------------------------------------------- #
def test_fov_and_preprocess_geometry() -> None:
    from src.data.retina_preprocess import estimate_fov_mask, preprocess_fundus, restore_to_original

    img, ves = synthetic_fundus()
    fov = estimate_fov_mask(img)
    assert 0.3 < fov.mean() < 0.8
    out, fov_s, ves_s, g = preprocess_fundus(img, size=128, vessels=ves, clahe=True)
    assert out.shape == (128, 128, 3) and fov_s.shape == ves_s.shape == (128, 128)
    assert set(np.unique(ves_s)) <= {0, 1} and ves_s.sum() > 0
    assert out[fov_s == 0].max() == 0  # fondo negro fuera del FOV
    back = restore_to_original(ves_s.astype(np.float32), g)
    assert back.shape == img.shape[:2]
    inter = ((back > 0.5) & (ves > 0)).sum()
    assert inter / (ves > 0).sum() > 0.6  # la máscara regresa a su sitio


def test_prepare_scans_fives_and_hrf(tmp_path: Path) -> None:
    from src.data.prepare_retina import prepare

    raw = tmp_path / "raw"
    for split, n in (("train", 12), ("test", 4)):
        for sub in ("Original", "Ground truth"):
            (raw / "FIVES" / "x" / split / sub).mkdir(parents=True)
        for i in range(n):
            img, ves = synthetic_fundus(seed=i)
            name = f"{i}_{'ADGN'[i % 4]}.png"
            cv2.imwrite(str(raw / "FIVES/x" / split / "Original" / name), img[..., ::-1])
            cv2.imwrite(str(raw / "FIVES/x" / split / "Ground truth" / name), ves)
    for sub in ("images", "manual1", "mask"):
        (raw / "HRF/all" / sub).mkdir(parents=True)
    img, ves = synthetic_fundus(seed=99)
    cv2.imwrite(str(raw / "HRF/all/images/01_h.jpg"), img[..., ::-1])
    cv2.imwrite(str(raw / "HRF/all/manual1/01_h.tif"), ves)
    cv2.imwrite(str(raw / "HRF/all/mask/01_h_mask.tif"), (ves * 0 + 255).astype(np.uint8))

    m = prepare(raw, tmp_path / "out", size=64, val_size=0.25, workers=1)
    assert set(m.split) == {"train", "val", "test", "external"}
    assert (m.split == "val").sum() == 3 and (m.split == "external").sum() == 1
    assert set(m[m.dataset == "FIVES"].category) == {"dmae", "retinopatia_diabetica", "glaucoma", "normal"}
    assert json.loads((tmp_path / "out/prepare_config.json").read_text())["size"] == 64


# --------------------------------------------------------------------------- #
# Modelo, pérdidas y métricas
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("encoder", ["plain", "resnet50"])
def test_unet_shapes_any_size(encoder: str) -> None:
    from src.models.unet import build_unet, encoder_layers, set_encoder_trainable

    m = build_unet(encoder, weights=None, decoder_filters=(16, 8, 8, 8, 4), plain_base=4)
    for s in (64, 96):
        assert m(np.zeros((1, s, s, 3), np.float32)).shape == (1, s, s, 1)
    if encoder == "plain":
        assert encoder_layers(m) == []
    else:
        assert set_encoder_trainable(m, False) > 100
        assert not any(layer.trainable for layer in encoder_layers(m))
        set_encoder_trainable(m, True)
        bns = [la for la in encoder_layers(m) if isinstance(la, tf.keras.layers.BatchNormalization)]
        assert bns and not any(b.trainable for b in bns)
        assert m.get_layer("dec0_conv1").trainable


def test_dice_bce_loss_and_metric() -> None:
    from src.models.seg_losses import DiceAtThreshold, DiceBCELoss

    y = np.zeros((2, 8, 8, 1), np.float32)
    y[:, 2:5, 2:5] = 1
    good = np.where(y > 0, 10.0, -10.0).astype(np.float32)
    bad = -good
    loss = DiceBCELoss()
    assert float(loss(y, good)) < 0.01 < float(loss(y, bad))
    met = DiceAtThreshold()
    met.update_state(y, good)
    assert float(met.result()) == pytest.approx(1.0)


def test_seg_metrics() -> None:
    from src.evaluation.seg_metrics import best_threshold, hd95, image_metrics

    gt = np.zeros((32, 32), np.uint8)
    gt[10:20, 10:20] = 1
    prob = gt.astype(np.float32) * 0.9
    fov = np.ones_like(gt)
    m = image_metrics(prob, gt, fov, 0.5)
    assert m["dice"] == pytest.approx(1.0) and m["auc"] == pytest.approx(1.0) and m["hd95"] == 0.0
    shifted = np.roll(gt, 3, axis=1)
    assert hd95(shifted, gt) == pytest.approx(3.0)
    thr, d = best_threshold([prob], [gt], [fov])
    assert d == pytest.approx(1.0) and thr <= 0.8


# --------------------------------------------------------------------------- #
# Entrenamiento corto + inferencia + API
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trained_retina(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Entrena 2 épocas una U-Net plain diminuta sobre 8 fondos sintéticos."""
    from src.data.loader_retina import make_dataset
    from src.data.prepare_retina import prepare
    from src.models.seg_losses import DiceAtThreshold, DiceBCELoss
    from src.models.unet import build_unet

    tmp = tmp_path_factory.mktemp("retina")
    raw = tmp / "raw"
    for split in ("train", "test"):
        for sub in ("Original", "Ground truth"):
            (raw / "FIVES" / split / sub).mkdir(parents=True)
        for i in range(8 if split == "train" else 2):
            img, ves = synthetic_fundus(seed=i)
            cv2.imwrite(str(raw / "FIVES" / split / "Original" / f"{i}_N.png"), img[..., ::-1])
            cv2.imwrite(str(raw / "FIVES" / split / "Ground truth" / f"{i}_N.png"), ves)
    m = prepare(raw, tmp / "proc", size=64, val_size=0.25, workers=1)
    m = m.assign(**{c: m[c].map(lambda p: str(tmp / "proc" / p)) for c in ("image", "vessel", "fov")})
    ds = make_dataset(m[m.split == "train"], 2, True, crop=32)
    model = build_unet("plain", None, decoder_filters=(16, 8, 8, 8, 4), plain_base=4)
    model.compile(tf.keras.optimizers.Adam(1e-2), DiceBCELoss(), metrics=[DiceAtThreshold()], jit_compile=False)
    model.fit(ds, epochs=2, verbose=0)
    out = tmp / "model"
    out.mkdir()
    model.save(out / "best.keras")
    (out / "preprocess.json").write_text(json.dumps({"size": 64, "clahe": True, "threshold": 0.5}))
    return out / "best.keras"


def test_segmenter_roundtrip(trained_retina: Path) -> None:
    from src.deployment.inference_retina import RetinaSegmenter

    seg = RetinaSegmenter(trained_retina)  # recarga desde disco (capas serializables)
    img, _ = synthetic_fundus(250, 330, seed=5)
    out = seg.predict(img)
    assert out["prob"].shape == out["mask"].shape == img.shape[:2]
    assert 0.0 <= out["vessel_density"] <= 1.0 and out["threshold"] == 0.5
    assert not out["mask"][~out["fov"]].any()
    assert seg.overlay(img, out["mask"]).shape == img.shape


def test_api_retina(trained_retina: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import base64

    from fastapi.testclient import TestClient

    import src.deployment.api as api

    monkeypatch.setenv("RETINA_MODEL_PATH", str(trained_retina))
    api.get_retina_segmenter.cache_clear()
    client = TestClient(api.app)
    assert client.get("/").json()["models"]["retina"]["available"] is True

    img, _ = synthetic_fundus(seed=7)
    ok, buf = cv2.imencode(".png", img[..., ::-1])
    r = client.post("/predict/retina?include=both", files={"file": ("f.png", io.BytesIO(buf.tobytes()), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_size"] == [400, 300] and 0 <= body["vessel_density"] <= 1
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(body["mask_png"]), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (300, 400)
    assert body["overlay_png"]

    r = client.post("/predict/retina", files={"file": ("x.png", io.BytesIO(b"no es imagen"), "image/png")})
    assert r.status_code == 400 and r.json()["error"]["code"] == 400

    monkeypatch.setenv("RETINA_MODEL_PATH", "/no/existe.keras")
    api.get_retina_segmenter.cache_clear()
    r = client.post("/predict/retina", files={"file": ("f.png", io.BytesIO(buf.tobytes()), "image/png")})
    assert r.status_code == 503
    api.get_retina_segmenter.cache_clear()


def test_api_capture_eye(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """La app OjoScan guarda recortes con consentimiento como CSV que entiende el cargador."""
    from fastapi.testclient import TestClient

    import src.deployment.api as api

    monkeypatch.setenv("CAPTURES_DIR", str(tmp_path / "caps"))
    client = TestClient(api.app)
    img = np.full((120, 120, 3), 128, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    meta = {"ojo": "izquierdo", "caja": [30, 50, 60, 20], "consentimiento": True, "apertura": 0.93,
            "puntos": {"inner_corner": [32, 60], "outer_corner": [88, 61], "upper_lid": [60, 50], "lower_lid": [60, 70]}}

    def send(m: dict) -> "object":
        return client.post("/captures/eye", files={"file": ("o.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
                           data={"meta": json.dumps(m)})

    r = send(meta)
    assert r.status_code == 200, r.text
    assert (tmp_path / "caps" / r.json()["image"]).exists()
    assert send({**meta, "consentimiento": False}).status_code == 400
    assert send({**meta, "ojo": "tercero"}).status_code == 400
    send({**meta, "ojo": "derecho", "caja": None})

    import pandas as pd

    from src.data.eye_labels import load_eye_samples

    df = pd.read_csv(tmp_path / "caps" / "annotations.csv")
    assert len(df) == 2 and df.loc[0, "inner_corner_x"] == 32 and pd.isna(df.loc[0, "pupil_x"])
    # La carpeta sirve tal cual como datos del detector (formato CSV de FORMATO_ETIQUETAS.md).
    samples = load_eye_samples(tmp_path / "caps", "csv")
    with_box = [s for s in samples if s.num_eyes]
    assert len(with_box) == 1 and with_box[0].boxes[0].tolist() == [30, 50, 60, 20]


def test_api_eye_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """/predict/eye/analysis devuelve hallazgo principal y probabilidades que suman 1."""
    from fastapi.testclient import TestClient

    import src.deployment.api as api
    from src.data.prepare_ojo_externo import CLASES

    x = tf.keras.Input((None, None, 3))
    y = tf.keras.layers.Dense(len(CLASES))(tf.keras.layers.GlobalAveragePooling2D()(x))
    tf.keras.Model(x, y).save(tmp_path / "best.keras")
    (tmp_path / "config.json").write_text(json.dumps({"clases": list(CLASES), "tam_entrada": 64}))
    monkeypatch.setenv("EYE_ANALYSIS_MODEL_PATH", str(tmp_path / "best.keras"))
    api.get_eye_analyzer.cache_clear()
    client = TestClient(api.app)
    assert client.get("/").json()["models"]["eye_analysis"]["available"] is True

    ok, buf = cv2.imencode(".jpg", np.full((90, 140, 3), 150, np.uint8))
    r = client.post("/predict/eye/analysis", files={"file": ("o.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["principal"] in CLASES and len(b["hallazgos"]) == len(CLASES)
    assert abs(sum(h["probabilidad"] for h in b["hallazgos"]) - 1) < 1e-3
    assert b["hallazgos"][0]["clave"] == b["principal"] and b["aviso"]
    api.get_eye_analyzer.cache_clear()
