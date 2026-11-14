"""Medición de palidez de la conjuntiva (sin modelos: imágenes sintéticas)."""

from __future__ import annotations

import numpy as np

from src.deployment.inference_anemia import blanco_de_referencia, letterbox, mayor_componente, medir_color


def ojo_sintetico(conj_rgb: tuple[float, float, float], luz: tuple[float, float, float] = (1, 1, 1)):
    """Esclerótica blanca arriba, conjuntiva del color dado abajo, todo bajo una luz de color."""
    img = np.full((200, 200, 3), 0.25, np.float32)  # piel/fondo oscuro
    img[40:110, 30:170] = (0.92, 0.90, 0.88)          # blanco del ojo
    img[120:160, 30:170] = conj_rgb                    # conjuntiva
    rng = np.random.default_rng(0)
    img = np.clip(img * np.array(luz, np.float32) + rng.normal(0, 0.01, img.shape), 0, 1).astype(np.float32)
    conj = np.zeros((200, 200), bool)
    conj[120:160, 30:170] = True
    util = np.ones((200, 200), bool)
    return img, conj, util


ROJA = (0.80, 0.35, 0.38)   # conjuntiva bien irrigada
PALIDA = (0.85, 0.66, 0.64)  # conjuntiva pálida


def medir(conj_rgb, luz=(1, 1, 1)) -> dict[str, float]:
    img, conj, util = ojo_sintetico(conj_rgb, luz)
    blanco, n = blanco_de_referencia(img, conj, util)
    assert n >= 150, "debe encontrar el blanco del ojo"
    return medir_color(img, conj, blanco)


def test_roja_mas_eritema_que_palida() -> None:
    r, p = medir(ROJA), medir(PALIDA)
    assert r["eritema"] > p["eritema"] + 0.2
    assert r["a_media"] > p["a_media"] + 10


def test_balance_de_blancos_corrige_luz_amarilla() -> None:
    # Con un foco cálido, sin corrección la conjuntiva pálida parecería más roja.
    calida = (1.0, 0.85, 0.65)
    p_neutra, p_calida = medir(PALIDA), medir(PALIDA, calida)
    assert abs(p_neutra["eritema"] - p_calida["eritema"]) < 0.06
    # y sigue distinguiéndose de una roja bajo la misma luz
    assert medir(ROJA, calida)["eritema"] > p_calida["eritema"] + 0.2


def test_letterbox_y_componente() -> None:
    img = np.zeros((300, 600, 3), np.uint8)
    out, util = letterbox(img, 512)
    assert out.shape == (512, 512, 3) and util.sum() == 512 * 256
    m = np.zeros((50, 50), bool)
    m[5:10, 5:10] = True
    m[20:40, 20:40] = True
    assert mayor_componente(m).sum() == 400


def test_api_anemia(tmp_path, monkeypatch) -> None:
    """/predict/anemia: 503 sin modelos; con un segmentador falso responde «sin conjuntiva» o un
    resultado con probabilidad, siempre con el aviso de que no es diagnóstico."""
    import io

    import cv2
    import joblib
    import tensorflow as tf
    from fastapi.testclient import TestClient
    from sklearn.linear_model import LogisticRegression

    import src.deployment.api as api
    from src.deployment.inference_anemia import CARACTERISTICAS

    monkeypatch.setenv("CONJUNTIVA_MODEL_PATH", str(tmp_path / "no.keras"))
    monkeypatch.setenv("ANEMIA_MODEL_PATH", str(tmp_path / "no.joblib"))
    api.get_anemia_analyzer.cache_clear()
    client = TestClient(api.app)
    assert client.get("/").json()["models"]["anemia"]["available"] is False
    ok, buf = cv2.imencode(".jpg", np.full((120, 160, 3), 150, np.uint8))
    archivo = {"file": ("o.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
    assert client.post("/predict/anemia", files=archivo).status_code == 503

    # Segmentador falso: «conjuntiva» = lo que es más rojo que verde (conv 1×1: 20·R − 20·G − 2).
    x = tf.keras.Input((None, None, 3))
    capa = tf.keras.layers.Conv2D(1, 1)
    m = tf.keras.Model(x, capa(x))
    capa.set_weights([np.array([20.0, -20.0, 0.0], np.float32).reshape(1, 1, 3, 1), np.array([-2.0], np.float32)])
    m.save(tmp_path / "seg.keras")
    X = np.random.default_rng(0).normal(size=(20, len(CARACTERISTICAS)))
    clf = LogisticRegression().fit(X, np.arange(20) % 2)
    joblib.dump({"modelo": clf, "umbral": 0.5, "validacion": {"auc_cv": 0.8}}, tmp_path / "clf.joblib")
    monkeypatch.setenv("CONJUNTIVA_MODEL_PATH", str(tmp_path / "seg.keras"))
    monkeypatch.setenv("ANEMIA_MODEL_PATH", str(tmp_path / "clf.joblib"))
    api.get_anemia_analyzer.cache_clear()
    img = np.zeros((400, 400, 3), np.uint8)
    img[:200] = (230, 228, 225)   # blanco arriba
    img[200:] = (200, 90, 95)     # rojo abajo
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    r = client.post("/predict/anemia", files={"file": ("o.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["valida"] is True and 0 <= b["probabilidad_anemia"] <= 1 and b["aviso"]
    assert b["fraccion_conjuntiva"] > 0.4
    api.get_anemia_analyzer.cache_clear()
