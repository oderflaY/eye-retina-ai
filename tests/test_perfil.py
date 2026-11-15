"""Perfil médico: reglas clínicas y modelos de ~/Deep_Learning."""
from pathlib import Path

import pytest

from src.deployment.inference_perfil import AnalizadorPerfil

CARPETA = Path("models/perfil")
pytestmark = pytest.mark.skipif(not (CARPETA / "resultados.json").exists(), reason="entrena train_perfil primero")


@pytest.fixture(scope="module")
def a():
    return AnalizadorPerfil(CARPETA)


def por_clave(r):
    return {i["clave"]: i for i in r["indicadores"]}


def test_hombre_sano(a):
    r = por_clave(a.predict({"edad": 30, "sexo": "M", "peso_kg": 72, "estatura_cm": 178, "cuello_cm": 38,
                             "abdomen_cm": 82, "cadera_cm": 95, "presion_sistolica": 115, "presion_diastolica": 75}))
    assert r["imc"]["nivel"] == "ok" and r["presion"]["nivel"] == "ok"
    assert 5 < float(r["grasa"]["valor"].split()[0]) < 20
    assert "bodyfat" in r["grasa"]["fuente"]
    assert "corazon" not in r  # sin dolor de pecho no se aplica el modelo


def test_dolor_tipico_con_esfuerzo_es_alto(a):
    r = por_clave(a.predict({"edad": 62, "sexo": "M", "presion_sistolica": 150, "presion_diastolica": 95,
                             "dolor_pecho": "tipico", "dolor_con_ejercicio": True}))
    assert r["corazon"]["nivel"] == "alta"
    assert r["presion"]["nivel"] == "alta"


def test_dolor_no_anginoso_joven_es_bajo(a):
    r = por_clave(a.predict({"edad": 28, "sexo": "F", "presion_sistolica": 110, "presion_diastolica": 70,
                             "dolor_pecho": "no_anginoso"}))
    assert r["corazon"]["nivel"] == "ok"


def test_mujer_usa_formula(a):
    r = por_clave(a.predict({"sexo": "F", "estatura_cm": 162, "cuello_cm": 32, "abdomen_cm": 78, "cadera_cm": 98}))
    assert "mujeres" in r["grasa"]["fuente"]


def test_datos_vacios_no_truenan(a):
    r = a.predict({})
    assert r["indicadores"] == [] and r["faltan"]


def test_pulso_de_pulsera():
    from pathlib import Path
    from src.deployment.inference_perfil import AnalizadorPerfil
    a = AnalizadorPerfil(Path("models/perfil"))
    r = a.predict({"pulso_reposo": 112, "pulso_min": 105, "pulso_max": 130, "pulso_segundos": 240, "pulso_irregular": 0.4})
    claves = {i["clave"]: i for i in r["indicadores"]}
    assert claves["pulso"]["nivel"] == "media" and claves["ritmo"]["nivel"] == "media"
    r = a.predict({"edad": 30, "sexo": "M", "pulso_reposo": 68, "pulso_segundos": 30})
    p = [i for i in r["indicadores"] if i["clave"] == "pulso"][0]
    assert p["nivel"] == "ok" and "30 s" in p["texto"]
    assert not any(i["clave"] == "ritmo" for i in r["indicadores"])


def test_historial_de_la_pulsera():
    from pathlib import Path
    from src.deployment.inference_perfil import AnalizadorPerfil
    a = AnalizadorPerfil(Path("models/perfil"))
    h = {"dias": 30, "pulso_reposo": 72, "pulso_reposo_7d": 82, "pulso_reposo_base": 70, "spo2_media": 95.5,
         "spo2_bajas": 0.08, "spo2_n": 200, "sueno_horas": 5.4, "sueno_noches": 25, "pasos": 3500, "pasos_dias": 28}
    r = a.predict({"edad": 40, "sexo": "M", "peso_kg": 100, "estatura_cm": 175, "historial": h})
    k = {i["clave"]: i["nivel"] for i in r["indicadores"]}
    assert k["pulso_historial"] == "ok" and k["pulso_tendencia"] == "media"
    assert k["spo2"] == "media" and k["sueno"] == "media" and k["pasos"] == "media" and k["apnea"] == "media"
    # sin historial no aparece nada nuevo
    r = a.predict({"edad": 40, "sexo": "M"})
    assert not any(i["clave"] in ("spo2", "sueno", "pasos") for i in r["indicadores"])
