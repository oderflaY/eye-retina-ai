"""Ictericia por el color del blanco del ojo (medidas de rasgos_color_rostro.py → regresión logística).

    python -m src.training.train_rostro_ictericia

Positivos: Jaundice_Dataset (CC BY-NC, fotos con la cara visible). Negativos: sanos del mismo
dataset + caras de FFHQ (muchas luces y tonos de piel distintos: miden las falsas alarmas).
Validación cruzada estratificada 5 × 10. Modelo deliberadamente simple (pocas medidas con sentido
clínico) porque hay solo ~40 casos positivos.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT

CARACTERISTICAS = ["esp_b", "gw_esp_b", "esp_b_menos_piel", "esp_a", "gw_esp_a", "esp_L", "piel_b"]
UMBRAL = 0.5


def main() -> None:
    d = pd.read_csv(PROJECT_ROOT / "data/processed/rostro/color.csv").dropna(subset=CARACTERISTICAS).reset_index(drop=True)
    X, y = d[CARACTERISTICAS].to_numpy(), d.clase.to_numpy()
    modelo = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"))
    oof, n = np.zeros(len(y)), np.zeros(len(y))
    aucs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42).split(X, y):
        p = clone(modelo).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        oof[te] += p
        n[te] += 1
    oof /= n
    pred = oof >= UMBRAL
    ffhq = (d.fuente == "ffhq").to_numpy()
    regla = d.esp_b_menos_piel.to_numpy() > 0  # referencia: "ojo más amarillo que la piel"
    res = {
        "n": int(len(y)), "n_ictericia": int(y.sum()), "n_ffhq": int(ffhq.sum()),
        "auc_cv": round(float(np.mean(aucs)), 4), "auc_sd": round(float(np.std(aucs)), 4),
        "sensibilidad": round(float(pred[y == 1].mean()), 4),
        "especificidad": round(float((~pred[y == 0]).mean()), 4),
        "falsas_alarmas_ffhq": round(float(pred[ffhq].mean()), 4),
        "falsas_alarmas_sanos_dataset": round(float(pred[(~ffhq) & (y == 0)].mean()), 4),
        "regla_simple": {"sensibilidad": round(float(regla[y == 1].mean()), 4),
                         "falsas_alarmas_ffhq": round(float(regla[ffhq].mean()), 4)},
        "umbral": UMBRAL, "metodo": "validación cruzada estratificada 5×10",
        "datos": "Jaundice_Dataset (CC BY-NC) + FFHQ como control sano",
    }
    final = clone(modelo).fit(X, y)
    sal = PROJECT_ROOT / "models/rostro"
    sal.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": final, "caracteristicas": CARACTERISTICAS, "umbral": UMBRAL, "validacion": res},
                sal / "ictericia.joblib")
    rep = PROJECT_ROOT / "reports/rostro"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "ictericia.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
