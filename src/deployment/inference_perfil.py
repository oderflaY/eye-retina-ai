"""Perfil médico: lo que el paciente escribe (edad, medidas, presión, síntomas, laboratorio).

Modelos entrenados con los datos de ~/Deep_Learning (ver src/training/train_perfil.py):
- grasa corporal (bodyfat.csv, hombres) → regresión lineal con cinta métrica, error medio ±3.6 %.
- riesgo coronario (UCI Cleveland, el dataset de analysis.ipynb) → logística con cuestionario +
  presión (AUC 0.85) o + laboratorio (AUC 0.86).
Lo demás son reglas clínicas publicadas (OMS, AHA 2017, ADA), no modelos.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

AVISO = ("Orientativo. Los modelos se entrenaron con pocos datos (252 hombres para grasa corporal, "
         "303 pacientes de cardiología para el corazón). No es diagnóstico.")

DOLORES = {"tipico": 1, "atipico": 2, "no_anginoso": 3}


def _num(d: dict, k: str) -> float | None:
    v = d.get(k)
    try:
        v = float(v)
        return v if math.isfinite(v) and v > 0 else None
    except (TypeError, ValueError):
        return None


def _ind(clave: str, nombre: str, valor: str, nivel: str, texto: str, fuente: str, **extra: Any) -> dict:
    return {"clave": clave, "nombre": nombre, "valor": valor, "nivel": nivel, "texto": texto, "fuente": fuente, **extra}


class AnalizadorPerfil:
    def __init__(self, carpeta: Path):
        self.grasa = joblib.load(carpeta / "grasa_hombres.joblib")
        self.cor_basico = joblib.load(carpeta / "corazon_basico.joblib")
        self.cor_lab = joblib.load(carpeta / "corazon_con_laboratorio.joblib")
        r = json.loads((carpeta / "resultados.json").read_text())
        self.val_grasa = r["grasa_corporal"]["modelos"]["cinta_metrica"]["mae_cv"]
        self.val_cor = {k: r["corazon"]["modelos"][k] for k in ("basico", "con_laboratorio")}

    @staticmethod
    def _pulso(d: dict) -> list[dict]:
        reposo, seg = _num(d, "pulso_reposo"), _num(d, "pulso_segundos") or 0
        if not reposo:
            return []
        out = []
        corto = seg < 60
        nota = " Medido solo %d s; para un pulso en reposo fiable, quédate sentado 5 minutos." % seg if corto else ""
        rango = f"(mín {_num(d, 'pulso_min') or reposo:.0f} · máx {_num(d, 'pulso_max') or reposo:.0f}, {seg:.0f} s)"
        if reposo < 40: niv, t = "alta", "Pulso muy bajo (bradicardia). Si hay mareo, desmayo o falta de aire, acude a urgencias."
        elif reposo < 50: niv, t = "media", "Pulso bajo. Es normal en deportistas; si te mareas o te cansas mucho, consulta."
        elif reposo <= 100: niv, t = "ok", "Pulso en reposo normal (50–100)."
        elif reposo <= 120: niv, t = "media", ("Pulso en reposo alto (taquicardia). Puede ser nervios, café, fiebre, deshidratación "
                                               "o anemia; si sigue alto en reposo, consulta.")
        else: niv, t = "alta", "Pulso en reposo muy alto. Si hay dolor de pecho, mareo o falta de aire, acude a urgencias."
        out.append(_ind("pulso", "Pulso en reposo (pulsera)", f"{reposo:.0f} lpm {rango}", niv, t + nota,
                        "Pulsera por Bluetooth · rangos AHA", taquicardia=reposo > 100))
        irr, rmssd = _num(d, "pulso_irregular"), _num(d, "pulso_rmssd")
        if irr is not None and seg >= 60 and irr > 0.25:
            out.append(_ind("ritmo", "Ritmo del pulso", f"{irr * 100:.0f} % de latidos irregulares", "media",
                            "El ritmo se vio irregular. Las pulseras confunden el movimiento con arritmia: repítelo quieto. "
                            "Si sigue irregular o sientes palpitaciones, pide un electrocardiograma.",
                            "Intervalos entre latidos de la pulsera"))
        elif rmssd is not None and seg >= 60:
            out.append(_ind("ritmo", "Ritmo del pulso", f"regular · VFC {rmssd:.0f} ms", "ok",
                            "Los latidos llegaron a un ritmo regular.", "Intervalos entre latidos de la pulsera"))
        return out

    @staticmethod
    def _historial(h: dict, imc: float | None, sis: float | None) -> list[dict]:
        if not h:
            return []
        out: list[dict] = []
        fuente = f"Pulsera, últimos {int(h.get('dias') or 30)} días (Health Connect)"
        g = lambda k: _num(h, k)  # noqa: E731

        # Pulso en reposo de 30 días + tendencia de la última semana
        pr, p7, base = g("pulso_reposo"), g("pulso_reposo_7d"), g("pulso_reposo_base")
        if pr:
            if pr < 40: niv, t = "alta", "Pulso en reposo muy bajo de forma sostenida. Consulta con cardiología."
            elif pr < 50: niv, t = "media", "Pulso en reposo bajo. Normal en deportistas; si hay mareo o cansancio, consulta."
            elif pr <= 90: niv, t = "ok", "Pulso en reposo normal."
            elif pr <= 100: niv, t = "media", ("Pulso en reposo en el límite alto. Se asocia a menos condición física y más riesgo "
                                               "cardiovascular; el ejercicio regular lo baja.")
            else: niv, t = "alta", ("Pulso en reposo alto todo el mes (taquicardia). Causas comunes: anemia, tiroides, "
                                    "deshidratación, ansiedad. Pide revisión médica.")
            out.append(_ind("pulso_historial", "Pulso en reposo (30 días)", f"{pr:.0f} lpm", niv, t, fuente + " · AHA"))
        if p7 and base and p7 - base >= 7:
            out.append(_ind("pulso_tendencia", "Pulso en reposo esta semana", f"{p7:.0f} lpm (antes {base:.0f})", "media",
                            f"Subió {p7 - base:.0f} lpm respecto a tus semanas anteriores. Suele pasar con infecciones, fiebre, "
                            "poco sueño, estrés, alcohol o anemia. Si sigue alto o te sientes mal, consulta.", fuente))

        # Oxígeno en sangre (la muñeca falla con movimiento: se pide que sea sostenido)
        spo2, bajas, n_spo2 = g("spo2_media"), g("spo2_bajas"), int(g("spo2_n") or 0)
        if spo2 and n_spo2 >= 5:
            if spo2 < 92: niv, t = "alta", "Oxígeno en sangre bajo en promedio. Confírmalo con un oxímetro de dedo y consulta."
            elif spo2 < 95 or (bajas or 0) > 0.05: niv, t = "media", ("Oxígeno en el límite o con bajadas frecuentes (< 90 %). "
                                                                      "Puede ser apnea del sueño o un problema pulmonar; confírmalo con oxímetro de dedo.")
            else: niv, t = "ok", "Oxígeno en sangre normal (≥ 95 %)."
            extra = f", {bajas * 100:.0f} % de lecturas < 90" if bajas else ""
            out.append(_ind("spo2", "Oxígeno en sangre (SpO₂)", f"{spo2:.1f} % (mín {g('spo2_min') or spo2:.0f}{extra})", niv, t, fuente))

        # Sueño (AASM: adultos ≥ 7 h)
        horas, noches = g("sueno_horas"), int(g("sueno_noches") or 0)
        if horas and noches >= 5:
            prof = g("sueno_profundo")
            if horas < 6: niv, t = "media", "Duermes menos de 6 h en promedio. Dormir poco sube la presión, la glucosa y el peso."
            elif horas < 7: niv, t = "media", "Duermes algo menos de las 7 h recomendadas."
            elif horas > 10: niv, t = "media", "Duermes más de 10 h: puede ser cansancio por anemia, tiroides o depresión. Coméntalo con tu médico."
            else: niv, t = "ok", "Duermes lo recomendado (7–9 h)."
            out.append(_ind("sueno", "Sueño", f"{horas:.1f} h/noche ({noches} noches" + (f", {prof * 100:.0f} % profundo)" if prof else ")"),
                            niv, t, fuente + " · AASM"))

        # Actividad (Paluch 2022: ≥ 7,000 pasos/día, menos mortalidad)
        pasos, dias_p = g("pasos"), int(g("pasos_dias") or 0)
        if pasos and dias_p >= 5:
            if pasos < 5000: niv, t = "media", "Vida sedentaria (< 5,000 pasos al día). Caminar más baja la presión, la glucosa y el riesgo del corazón."
            elif pasos < 7000: niv, t = "ok", "Actividad moderada; llegar a 7,000–8,000 pasos da más beneficio."
            else: niv, t = "ok", "Buena actividad diaria."
            out.append(_ind("pasos", "Actividad", f"{pasos:,.0f} pasos/día".replace(",", ","), niv, t, fuente + " · Paluch 2022"))

        # Cruce: sospecha de apnea del sueño (oxígeno que baja de noche + factores de riesgo)
        if bajas is not None and bajas > 0.03 and ((imc or 0) >= 30 or (sis or 0) >= 130 or (horas or 8) < 6):
            factores = [x for x, c in [("obesidad", (imc or 0) >= 30), ("presión alta", (sis or 0) >= 130), ("poco sueño", (horas or 8) < 6)] if c]
            out.append(_ind("apnea", "Posible apnea del sueño", "bajadas de oxígeno + " + ", ".join(factores), "media",
                            "Este patrón es típico de la apnea del sueño (ronquido, pausas al respirar, cansancio de día). "
                            "Se confirma con un estudio de sueño.", fuente + " · STOP-BANG"))
        return out

    # ─────────────────────────────────────────────
    def predict(self, d: dict) -> dict:
        edad = _num(d, "edad")
        sexo = (d.get("sexo") or "").upper()[:1]  # "M" / "F"
        peso, est = _num(d, "peso_kg"), _num(d, "estatura_cm")
        cuello, abdomen, cadera = _num(d, "cuello_cm"), _num(d, "abdomen_cm"), _num(d, "cadera_cm")
        sis, dia = _num(d, "presion_sistolica"), _num(d, "presion_diastolica")
        col, glu, fcmax = _num(d, "colesterol"), _num(d, "glucosa_ayunas"), _num(d, "fc_maxima")
        dolor = d.get("dolor_pecho") or "ninguno"
        dolor_ej = bool(d.get("dolor_con_ejercicio"))

        ind: list[dict] = []
        faltan: list[str] = []

        # IMC (OMS)
        if peso and est:
            imc = peso / (est / 100) ** 2
            if imc < 18.5: niv, t = "media", "Bajo peso"
            elif imc < 25: niv, t = "ok", "Peso normal"
            elif imc < 30: niv, t = "media", "Sobrepeso"
            else: niv, t = "alta", "Obesidad"
            ind.append(_ind("imc", "Índice de masa corporal", f"{imc:.1f}", niv, t, "OMS"))
        else:
            faltan.append("peso y estatura")

        # Cintura / estatura (> 0.5 = más grasa en el abdomen, riesgo cardiometabólico)
        if abdomen and est:
            r = abdomen / est
            niv = "ok" if r < 0.5 else ("media" if r < 0.6 else "alta")
            ind.append(_ind("cintura_estatura", "Cintura / estatura", f"{r:.2f}", niv,
                            "Cintura sana" if r < 0.5 else "Grasa acumulada en el abdomen (más riesgo de diabetes y corazón)",
                            "Regla 0.5 (Ashwell 2012)"))

        # Grasa corporal
        ind_grasa = self._grasa(sexo, edad, peso, est, cuello, abdomen, cadera)
        if ind_grasa: ind.append(ind_grasa)
        else: faltan.append("cuello, abdomen y cadera (cinta métrica) para la grasa corporal")

        # Presión (AHA 2017)
        if sis and dia:
            if sis >= 180 or dia >= 120: niv, t = "alta", "Crisis hipertensiva: busca atención médica hoy"
            elif sis >= 140 or dia >= 90: niv, t = "alta", "Hipertensión etapa 2"
            elif sis >= 130 or dia >= 80: niv, t = "media", "Hipertensión etapa 1"
            elif sis >= 120: niv, t = "media", "Presión elevada"
            else: niv, t = "ok", "Presión normal"
            ind.append(_ind("presion", "Presión arterial", f"{sis:.0f}/{dia:.0f} mmHg", niv, t, "AHA 2017"))
        else:
            faltan.append("presión arterial")

        # Glucosa en ayunas (ADA) y colesterol total
        if glu:
            niv, t = ("alta", "En rango de diabetes: confírmalo con tu médico") if glu >= 126 else \
                     ("media", "Prediabetes") if glu >= 100 else ("ok", "Glucosa normal")
            ind.append(_ind("glucosa", "Glucosa en ayunas", f"{glu:.0f} mg/dL", niv, t, "ADA"))
        if col:
            niv, t = ("alta", "Colesterol alto") if col >= 240 else ("media", "Colesterol limítrofe") if col >= 200 else ("ok", "Colesterol deseable")
            ind.append(_ind("colesterol", "Colesterol total", f"{col:.0f} mg/dL", niv, t, "NCEP ATP III"))

        # Pulso en vivo de la pulsera (Bluetooth, durante el chequeo)
        ind += self._pulso(d)
        # Historial de 30 días de la pulsera (Mi Fitness → Health Connect)
        imc_v = peso / (est / 100) ** 2 if peso and est else None
        ind += self._historial(d.get("historial") or {}, imc_v, sis)

        # Corazón (solo con dolor de pecho: así es la población del dataset)
        cor = self._corazon(edad, sexo, dolor, dolor_ej, sis, col, glu, fcmax)
        if cor: ind.append(cor)

        orden = {"alta": 0, "media": 1, "ok": 2}
        ind.sort(key=lambda x: orden[x["nivel"]])
        return {"indicadores": ind, "faltan": faltan, "aviso": AVISO,
                "fuente_modelos": "Datos de ~/Deep_Learning: bodyfat.csv y UCI Cleveland (analysis.ipynb)"}

    # ─────────────────────────────────────────────
    def _grasa(self, sexo, edad, peso, est, cuello, abdomen, cadera) -> dict | None:
        if sexo == "M" and all([edad, peso, est, cuello, abdomen, cadera]):
            x = pd.DataFrame([[edad, peso, est, cuello, abdomen, cadera]], columns=self.grasa["variables"])
            g = float(np.clip(self.grasa["modelo"].predict(x)[0], 2, 60))
            fuente = f"Modelo entrenado con bodyfat.csv (252 hombres), error medio ±{self.val_grasa} %"
        elif sexo == "M" and all([est, cuello, abdomen]) and abdomen > cuello:
            g = 495 / (1.0324 - 0.19077 * math.log10(abdomen - cuello) + 0.15456 * math.log10(est)) - 450
            fuente = "Fórmula de la Marina de EE. UU. (faltan medidas para el modelo)"
        elif sexo == "F" and all([est, cuello, abdomen, cadera]) and abdomen + cadera > cuello:
            g = 495 / (1.29579 - 0.35004 * math.log10(abdomen + cadera - cuello) + 0.22100 * math.log10(est)) - 450
            fuente = "Fórmula de la Marina de EE. UU. para mujeres (el dataset solo tiene hombres)"
        else:
            return None
        # Rangos del American Council on Exercise
        lim = (6, 14, 18, 25) if sexo == "M" else (14, 21, 25, 32)
        if g < lim[0]: niv, t = "media", "Grasa muy baja"
        elif g < lim[2]: niv, t = "ok", "Grasa saludable"
        elif g < lim[3]: niv, t = "media", "Grasa algo alta"
        else: niv, t = "alta", "Grasa alta (obesidad por grasa corporal)"
        return _ind("grasa", "Grasa corporal", f"{g:.1f} %", niv, t, fuente)

    def _corazon(self, edad, sexo, dolor, dolor_ej, sis, col, glu, fcmax) -> dict | None:
        if dolor not in DOLORES:
            # En Cleveland los "sin dolor" eran pacientes enviados por otra razón y casi todos estaban
            # enfermos; aplicar el modelo a alguien sano sin síntomas daría un riesgo falso.
            return None
        if not (edad and sexo in ("M", "F") and sis):
            return _ind("corazon", "Riesgo coronario", "—", "media",
                        "Tienes dolor de pecho: completa edad, sexo y presión para estimar el riesgo. "
                        "Si el dolor es fuerte, opresivo o con falta de aire, ve a urgencias.", "—")
        base = {"age": edad, "sex": 1 if sexo == "M" else 0, "dolor_tipico": int(dolor == "tipico"),
                "dolor_atipico": int(dolor == "atipico"), "dolor_no_anginoso": int(dolor == "no_anginoso"),
                "exang": int(dolor_ej), "trestbps": sis}
        if col and glu and fcmax:
            m, val = self.cor_lab, self.val_cor["con_laboratorio"]
            base |= {"chol": col, "fbs": int(glu > 120), "thalach": fcmax}
        else:
            m, val = self.cor_basico, self.val_cor["basico"]
        p = float(m["modelo"].predict_proba(pd.DataFrame([base])[m["variables"]])[0, 1])
        niv = "alta" if p >= 0.6 else ("media" if p >= 0.3 else "ok")
        t = {"alta": "Probabilidad alta de enfermedad coronaria: consulta con cardiología pronto",
             "media": "Probabilidad intermedia: conviene una valoración médica (electrocardiograma)",
             "ok": "Probabilidad baja de que el dolor sea del corazón"}[niv]
        if dolor == "tipico" and dolor_ej:
            t += ". Dolor opresivo que aparece con el esfuerzo: si dura más de 15 min o es en reposo, ve a urgencias"
        return _ind("corazon", "Riesgo coronario", f"{p:.0%}", niv, t,
                    f"Modelo de analysis.ipynb (UCI Cleveland), AUC {val['auc_cv']} validación cruzada",
                    probabilidad=round(p, 4), validacion={"auc": val["auc_cv"], "sensibilidad": val["sensibilidad"],
                                                          "especificidad": val["especificidad"], "n": val["n"]})
