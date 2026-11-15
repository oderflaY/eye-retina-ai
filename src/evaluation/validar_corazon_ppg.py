"""Validación externa del modelo de FA con PPG REAL (sensor óptico, como el de la pulsera).

MIMIC PERform AF (Zenodo 15906524, ODbL): 35 pacientes de UCI, 20 min de PPG de dedo a 125 Hz,
19 con fibrilación auricular y 16 sin. Ninguno se usó para entrenar.

Latidos del PPG (picos) → misma señal de pulsera que en el entrenamiento → modelo fa.
Uso: .venv/bin/python -m src.evaluation.validar_corazon_ppg
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics import roc_auc_score

from src.config import PROJECT_ROOT
from src.corazon_senal import latidos_a_banda, variables

DATOS = Path.home() / "TemrariosSpace/datasets/Corazon/mimic_af"
MODELOS = PROJECT_ROOT / "models/corazon"
FS = 125


def latidos_ppg(ppg: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(ppg - np.nanmean(ppg))
    b, a = butter(3, [0.5 / (FS / 2), 8 / (FS / 2)], "band")
    f = filtfilt(b, a, x)
    picos, _ = find_peaks(f, distance=int(0.27 * FS), prominence=0.25 * np.std(f))
    return picos / FS


def main() -> None:
    m = joblib.load(MODELOS / "fa.joblib")
    res_prev = json.loads((MODELOS / "resultados.json").read_text())
    u_vig = res_prev["fa"]["umbral_vigilar"]
    rng = np.random.default_rng(3)
    filas = []
    for csv in sorted(DATOS.glob("*/*_data.csv")):
        fa = int("non_af" not in csv.name)
        d = pd.read_csv(csv)
        t = latidos_ppg(d["PPG"].values)
        if len(t) < 100:
            continue
        _, serie = latidos_a_banda(t, rng, suave_s=5.0)
        for k in range(len(serie) // 120):
            v = variables(serie[k * 120:(k + 1) * 120])
            if v:
                p = float(m["modelo"].predict_proba(pd.DataFrame([v])[m["variables"]].values)[0, 1])
                filas.append({"persona": csv.stem, "y": fa, "p": p})
    df = pd.DataFrame(filas)
    per = df.groupby("persona").agg(y=("y", "first"), p=("p", "median"))
    out = {
        "fuente": "MIMIC PERform AF (Zenodo 15906524): PPG de dedo, 125 Hz, pacientes de UCI no vistos",
        "personas": int(len(per)), "personas_fa": int(per.y.sum()), "ventanas": int(len(df)),
        "auc_ventana": round(roc_auc_score(df.y, df.p), 3),
        "auc_persona": round(roc_auc_score(per.y, per.p), 3),
        "sens_persona": round(float((per[per.y == 1].p >= u_vig).mean()), 3),
        "esp_persona": round(float((per[per.y == 0].p < u_vig).mean()), 3),
    }
    print(json.dumps(out, ensure_ascii=False))
    res_prev["validacion_ppg"] = out
    (MODELOS / "resultados.json").write_text(json.dumps(res_prev, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
