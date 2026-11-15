"""Vigilancia cardiaca con los datos de la pulsera (Xiaomi Smart Band) para prevenir accidentes.

Junta tres cosas y las compara entre sí:
  1. IA entrenada con latidos reales de PhysioNet convertidos a la señal de la pulsera (train_corazon.py):
     fibrilación auricular (fa) y patrón de insuficiencia cardiaca (ic).
  2. Reglas clínicas de pulso en reposo, taquicardia/bradicardia sostenida, oxígeno y síntomas de alarma.
  3. Tu propia línea base: el historial de 30 días de Mi Fitness (Health Connect) y lo normal para tu edad.

No es un dispositivo médico: ante síntomas de alarma siempre dice que llames al 911.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.corazon_senal import variables

ORDEN = {"ok": 0, "vigilar": 1, "alto": 2, "urgente": 3}
SINTOMAS_ALARMA = {
    "dolor_pecho": "dolor u opresión en el pecho",
    "falta_aire": "falta de aire",
    "desmayo": "desmayo o casi desmayo",
    "mareo": "mareo fuerte",
    "palpitaciones": "palpitaciones",
}


def _n(d: dict | None, k: str) -> float | None:
    if not d:
        return None
    v = d.get(k)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _alerta(nivel: str, titulo: str, texto: str, accion: str) -> dict:
    return {"nivel": nivel, "titulo": titulo, "texto": texto, "accion": accion}


def _cmp(clave: str, nombre: str, tuyo: str, referencia: str, nivel: str, texto: str) -> dict:
    return {"clave": clave, "nombre": nombre, "tuyo": tuyo, "referencia": referencia, "nivel": nivel, "texto": texto}


def _sostenido(x: np.ndarray, cond, segundos: int = 60) -> bool:
    """¿La condición se cumple al menos `segundos` seguidos (tolerando huecos de 3 s)?"""
    racha, hueco = 0, 0
    for v in x:
        if cond(v):
            racha += 1 + hueco
            hueco = 0
            if racha >= segundos:
                return True
        else:
            hueco += 1
            if hueco > 3:
                racha, hueco = 0, 0
    return False


# clave: (nombre, cómo resumir las ventanas, alerta alto, alerta vigilar). Las alertas son (título, texto, acción).
MODELOS: dict[str, tuple] = {
    "fa": ("Fibrilación auricular", "mediana",
           ("Ritmo irregular: posible fibrilación auricular",
            "La IA reconoce en tu pulso el patrón de la fibrilación auricular ({p:.0%}). Sin tratamiento aumenta 5 veces "
            "el riesgo de embolia cerebral.",
            "Pide un electrocardiograma esta semana. Si además tienes falta de aire, dolor o debilidad de un lado del "
            "cuerpo, llama al 911."),
           ("Pulso algo irregular", "Hay saltos en tu pulso que se parecen a una arritmia, pero pueden ser por movimiento.",
            "Quédate quieto 2 minutos con la pulsera ajustada y vuelve a medir.")),
    "taqui": ("Taquicardia en episodio", "maxima",
              ("Posible episodio de taquicardia",
               "La IA vio un tramo con el patrón de una taquicardia ventricular o supraventricular ({p:.0%}).",
               "Siéntate y no manejes. Si sientes mareo, dolor de pecho o falta de aire, llama al 911. Si se repite, "
               "pide un Holter."),
              ("Aceleración brusca del pulso", "Tu pulso subió de golpe en un tramo, como en algunas taquicardias.",
               "Si no estabas moviéndote, quédate en reposo y vuelve a medir.")),
    "ectopia": ("Extrasístoles frecuentes", "mediana",
                ("Muchos latidos adelantados (extrasístoles)",
                 "Tu pulso tiene el patrón de extrasístoles frecuentes o bigeminia ({p:.0%}). Muchas son benignas, pero "
                 "si son muy seguidas conviene revisarlas.",
                 "Evita café, alcohol y desvelos. Consulta y pide un electrocardiograma o un Holter de 24 h."),
                ("Algunos latidos adelantados", "Hay saltos que parecen extrasístoles aisladas.",
                 "Suele ser benigno. Si sientes palpitaciones seguido, coméntalo con tu médico.")),
    "ic": ("Patrón de insuficiencia cardiaca", "mediana",
           ("Variabilidad del pulso muy baja",
            "Tu pulso cambia muy poco, como en personas con insuficiencia cardiaca ({p:.0%}). También pasa con estrés, "
            "fiebre o algunos medicamentos.",
            "Si te falta el aire al subir escaleras, se te hinchan los pies o duermes con varias almohadas, consulta a "
            "cardiología."),
           ("Variabilidad del pulso baja", "Tu pulso cambia poco; puede ser cansancio, estrés o fiebre.",
            "Repite la medición en reposo otro día.")),
    "ms": ("Patrón previo a paro cardiaco (experimental)", "mediana",
           ("Patrón de riesgo en tu ritmo",
            "La IA encontró un patrón parecido al de Holters de personas horas antes de un paro cardiaco ({p:.0%}). "
            "Es experimental y puede equivocarse, pero no lo ignores.",
            "Consulta a cardiología pronto. Si tienes desmayos, dolor de pecho o palpitaciones fuertes, llama al 911."),
           None),
}
# Para avisar se usa el umbral; las combinaciones contradictorias se descartan (p. ej. "pulso que cambia poco" con FA).
EXCLUYE = {"ic": ("fa", "taqui", "ectopia"), "ectopia": ("fa",), "ms": ("fa",)}


class AnalizadorCorazon:
    def __init__(self, carpeta: Path):
        self.val = json.loads((carpeta / "resultados.json").read_text())
        self.modelos = {k: joblib.load(carpeta / f"{k}.joblib") for k in MODELOS if (carpeta / f"{k}.joblib").exists()}

    def _ia(self, serie: np.ndarray) -> dict | None:
        """Cada modelo con su ventana (paso de media ventana). Episodios: el máximo; patrones: la mediana."""
        out: dict[str, Any] = {"modelos": []}
        for clave, m in self.modelos.items():
            largo = m["ventana_s"]
            filas = [v for ini in range(0, max(1, len(serie) - largo + 1), largo // 2)
                     if (v := variables(serie[ini:ini + largo]))]
            if not filas:
                continue
            p = m["modelo"].predict_proba(pd.DataFrame(filas)[m["variables"]].values)[:, 1]
            val = self.val[clave]
            pm = float(p.max() if MODELOS[clave][1] == "maxima" else np.median(p))
            nivel = "alto" if pm >= val["umbral_alto"] else ("vigilar" if pm >= val["umbral_vigilar"] else "ok")
            r = {"clave": clave, "nombre": MODELOS[clave][0], "probabilidad": round(pm, 3), "nivel": nivel,
                 "ventanas": len(filas),
                 "validacion": {"auc": val["auc_ventana"], "auc_persona": val["auc_persona"],
                                "sensibilidad": val["sens_vigilar"], "especificidad": val["esp_vigilar"],
                                "personas": val["personas"]}}
            out["modelos"].append(r)
            out[clave] = r
        if not out["modelos"]:
            return None
        out["ventanas"] = max(r["ventanas"] for r in out["modelos"])
        # descartar lo que contradice a un hallazgo más específico
        for clave, otros in EXCLUYE.items():
            if clave in out and any(out.get(o, {}).get("nivel", "ok") != "ok" for o in otros):
                out[clave]["nivel"] = "ok"
        return out

    def predict(self, d: dict) -> dict:
        crudo = d.get("serie") or []
        serie = np.array([float(s["lpm"]) if isinstance(s, dict) else float(s) for s in crudo], dtype=float)
        serie = serie[(serie >= 30) & (serie <= 230)] if len(serie) else serie
        h = d.get("historial") or {}
        edad = _n(d, "edad")
        reposo_mov = (d.get("actividad") or "reposo") == "reposo"
        sintomas = [s for s in (d.get("sintomas") or []) if s in SINTOMAS_ALARMA]
        alertas: list[dict] = []
        comparacion: list[dict] = []

        # ── Ahora (pulsera en vivo) ──────────────────────────────────────────
        ahora = None
        if len(serie) >= 10:
            ahora = {"lpm": int(serie[-5:].mean().round()), "promedio": int(serie.mean().round()),
                     "minimo": int(serie.min()), "maximo": int(serie.max()), "segundos": int(len(serie))}
            if reposo_mov and _sostenido(serie, lambda v: v >= 150):
                alertas.append(_alerta("urgente", "Pulso muy rápido en reposo",
                                       "Más de 150 latidos por minuto durante más de un minuto sin hacer ejercicio.",
                                       "Siéntate, respira lento. Si no baja en 5 minutos o tienes mareo o dolor, llama al 911."))
            elif reposo_mov and _sostenido(serie, lambda v: v >= 120):
                alertas.append(_alerta("alto", "Taquicardia en reposo",
                                       "Más de 120 lpm sostenido estando quieto.",
                                       "Descansa 10 minutos y vuelve a medir. Si sigue alto, acude a revisión hoy."))
            if _sostenido(serie, lambda v: v < 40):
                alertas.append(_alerta("urgente", "Pulso muy lento",
                                       "Menos de 40 lpm durante más de un minuto.",
                                       "Si sientes mareo, debilidad o te desmayas, llama al 911. Evita manejar."))
            elif _sostenido(serie, lambda v: v < 48, 120):
                alertas.append(_alerta("vigilar", "Pulso lento",
                                       "Menos de 48 lpm sostenido. En deportistas puede ser normal.",
                                       "Si no haces deporte o tienes mareo o cansancio, consulta y pide un electrocardiograma."))

        # ── IA cardiaca ──────────────────────────────────────────────────────
        ia = self._ia(serie) if len(serie) >= 60 else None
        if ia:
            for r in ia["modelos"]:
                _, _, alto, vigilar = MODELOS[r["clave"]]
                if r["nivel"] == "alto":
                    alertas.append(_alerta("alto", alto[0], alto[1].format(p=r["probabilidad"]), alto[2]))
                elif r["nivel"] == "vigilar" and vigilar and (r["clave"] != "ic" or r["ventanas"] >= 2):
                    alertas.append(_alerta("vigilar", *vigilar))

        # ── Comparación con tu historial y con lo normal para tu edad ───────
        base = _n(h, "pulsoReposo")
        if ahora and base:
            dif = ahora["promedio"] - base
            niv = "alto" if dif >= 25 and reposo_mov else ("vigilar" if dif >= 15 and reposo_mov else "ok")
            comparacion.append(_cmp("vs_historial", "Ahora contra tu reposo habitual", f"{ahora['promedio']} lpm",
                                    f"{base:.0f} lpm (30 días)", niv,
                                    f"{dif:+.0f} lpm respecto a tu pulso de reposo de Mi Fitness." +
                                    (" Si estás en reposo, es una subida grande." if niv != "ok" else "")))
        if ahora and edad:
            fcmax = 220 - edad
            frac = ahora["maximo"] / fcmax
            niv = "alto" if frac >= 0.95 else ("vigilar" if frac >= 0.85 else "ok")
            comparacion.append(_cmp("vs_edad", "Pulso máximo contra el límite para tu edad", f"{ahora['maximo']} lpm",
                                    f"{fcmax:.0f} lpm (220 − edad)", niv,
                                    f"Llegaste al {frac:.0%} de tu frecuencia máxima teórica." +
                                    (" Baja la intensidad: cerca del máximo aumenta el riesgo de un evento." if niv != "ok" else "")))
            if niv == "alto":
                alertas.append(_alerta("alto", "Esfuerzo al límite",
                                       f"Llegaste a {ahora['maximo']} lpm, casi tu máximo para {edad:.0f} años.",
                                       "Detente y recupera. Si hay dolor de pecho o mareo, llama al 911."))
        r7, rb = _n(h, "pulsoReposo7d"), _n(h, "pulsoReposoBase")
        if r7 and rb:
            dif = r7 - rb
            niv = "vigilar" if dif >= 7 else "ok"
            comparacion.append(_cmp("tendencia", "Esta semana contra tus semanas anteriores", f"{r7:.0f} lpm",
                                    f"{rb:.0f} lpm", niv,
                                    "Tu pulso de reposo subió: puede ser infección, estrés, poco sueño, deshidratación o anemia."
                                    if niv != "ok" else "Tu pulso de reposo está estable."))
            if niv != "ok":
                alertas.append(_alerta("vigilar", "Tu pulso de reposo va subiendo",
                                       f"+{dif:.0f} lpm esta semana contra tu promedio.",
                                       "Duerme bien, hidrátate y evita el alcohol unos días. Si sigue subiendo, consulta."))
        if base:
            niv = "alto" if base >= 100 else ("vigilar" if base >= 90 or base < 45 else "ok")
            comparacion.append(_cmp("reposo", "Pulso en reposo (30 días)", f"{base:.0f} lpm", "50–90 lpm", niv,
                                    "Un reposo alto sostenido se asocia a más eventos cardiovasculares." if base >= 90 else
                                    ("Reposo muy bajo: normal en deportistas; si no lo eres, consulta." if base < 45 else
                                     "En rango sano.")))
        smin, sbajas, smed = _n(h, "spo2Min"), _n(h, "spo2Bajas"), _n(h, "spo2Media")
        if smed:
            niv = "alto" if (smed < 92 or (sbajas or 0) >= 0.15) else ("vigilar" if (smed < 95 or (sbajas or 0) >= 0.05) else "ok")
            comparacion.append(_cmp("spo2", "Oxígeno en sangre", f"{smed:.1f} %" + (f" (mín {smin:.0f} %)" if smin else ""),
                                    "≥ 95 %", niv,
                                    "Bajadas de oxígeno frecuentes cansan al corazón (posible apnea del sueño)." if niv != "ok"
                                    else "Oxigenación normal."))
            if niv == "alto":
                alertas.append(_alerta("alto", "Oxígeno bajo frecuente",
                                       "Tu oxígeno baja de 90 % con frecuencia; de noche eso puede provocar arritmias.",
                                       "Consulta para un estudio de sueño. Si tienes falta de aire ahora, llama al 911."))
        horas = _n(h, "suenoHoras")
        if horas:
            niv = "vigilar" if horas < 6 else "ok"
            comparacion.append(_cmp("sueno", "Sueño", f"{horas:.1f} h", "7–9 h", niv,
                                    "Dormir menos de 6 h sube la presión y el riesgo cardiaco." if niv != "ok" else "Descanso suficiente."))

        # ── Síntomas de alarma ───────────────────────────────────────────────
        if sintomas:
            graves = [s for s in sintomas if s in ("dolor_pecho", "falta_aire", "desmayo")]
            lista = ", ".join(SINTOMAS_ALARMA[s] for s in sintomas)
            if graves:
                alertas.insert(0, _alerta("urgente", "Síntomas de alarma",
                                          f"Reportaste {lista}.",
                                          "Llama al 911 ahora. No manejes. Si tienes aspirina y no eres alérgico, "
                                          "mastica una mientras llega la ayuda."))
            elif any(a["nivel"] in ("alto", "urgente") for a in alertas):
                alertas.insert(0, _alerta("urgente", "Síntomas con pulso anormal",
                                          f"Reportaste {lista} junto con un pulso fuera de rango.",
                                          "Siéntate o acuéstate y llama al 911."))
            else:
                alertas.append(_alerta("vigilar", "Síntomas", f"Reportaste {lista}.",
                                       "Mide de nuevo en reposo. Si se repite, consulta."))

        alertas.sort(key=lambda a: -ORDEN[a["nivel"]])
        riesgo = max((a["nivel"] for a in alertas), key=ORDEN.get, default="ok")
        if not alertas and not ahora and not h:
            riesgo = "sin_datos"
        titulo = {
            "urgente": "Llama al 911",
            "alto": "Atención: consulta pronto",
            "vigilar": "Vigila tu corazón",
            "ok": "Corazón estable",
            "sin_datos": "Conecta tu pulsera",
        }[riesgo]
        return {
            "riesgo": riesgo, "titulo": titulo, "ahora": ahora, "ia": ia,
            "comparacion": comparacion, "alertas": alertas,
            "modelo": {"fuente": "PhysioNet: " + ", ".join(self.val.get("fuentes", {}).keys()),
                       "senal": self.val.get("senal")},
            "aviso": "Herramienta de prevención, no un diagnóstico. Una pulsera no sustituye un electrocardiograma.",
        }
