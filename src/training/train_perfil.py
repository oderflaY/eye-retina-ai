"""Modelos del perfil médico (datos que el paciente escribe, sin fotos).

Origen: el repositorio de prácticas ~/Deep_Learning (libro *Practical Deep Learning*):
- Grasa corporal: `bodyfat - bodyfat.csv` (252 hombres, grasa medida por densidad bajo el agua).
- Corazón: el `heart.xls` de `analysis.ipynb` es UCI Heart Disease, Cleveland (303 pacientes
  enviados a angiografía). Se usa el original de UCI (CC BY 4.0):
  https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
- Cáncer de mama (Wisconsin): NO se usa; pide medidas de células de una biopsia.

Solo se usan variables que una persona sabe o mide en casa. Se reporta validación cruzada
repetida (no una sola partición) para que los números sean honestos con tan pocos datos.

    python -m src.training.train_perfil
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATOS = Path.home() / "TemrariosSpace/datasets/Perfil"
SALIDA = Path("models/perfil")


# ───────────────────────── Grasa corporal ─────────────────────────
def marina_hombre(abdomen_cm, cuello_cm, estatura_cm):
    """Fórmula de la Marina de EE. UU. (Hodgdon y Beckett, 1984), hombres."""
    return 495 / (1.0324 - 0.19077 * np.log10(abdomen_cm - cuello_cm) + 0.15456 * np.log10(estatura_cm)) - 450


def grasa() -> dict:
    df = pd.read_csv(DATOS / "bodyfat.csv")
    df["peso_kg"] = df.Weight * 0.453592
    df["estatura_cm"] = df.Height * 2.54
    n0 = len(df)
    # Errores conocidos del dataset: grasa 0 % (fila 182) y estatura 29.5 in (fila 42)
    df = df[(df.BodyFat >= 3) & (df.Height > 50)].reset_index(drop=True)
    y = df.BodyFat.to_numpy()
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)

    conjuntos = {
        "abdomen_peso": ["Abdomen", "peso_kg"],
        "cinta_metrica": ["Age", "peso_kg", "estatura_cm", "Neck", "Abdomen", "Hip"],
    }
    res = {"n": int(len(df)), "eliminadas": int(n0 - len(df)), "modelos": {}}
    for nombre, cols in conjuntos.items():
        maes = []
        for i, (tr, te) in enumerate(cv.split(df)):
            m = LinearRegression().fit(df.loc[tr, cols], y[tr])
            maes.append(mean_absolute_error(y[te], m.predict(df.loc[te, cols])))
        res["modelos"][nombre] = {"variables": cols, "mae_cv": round(float(np.mean(maes)), 2)}
    pm = marina_hombre(df.Abdomen, df.Neck, df.estatura_cm)
    res["modelos"]["formula_marina"] = {"mae": round(float(mean_absolute_error(y, pm)), 2), "nota": "sin ajustar a estos datos"}
    res["mae_sin_modelo"] = round(float(np.mean(np.abs(y - y.mean()))), 2)

    final = conjuntos["cinta_metrica"]
    modelo = LinearRegression().fit(df[final], y)
    joblib.dump({"modelo": modelo, "variables": final}, SALIDA / "grasa_hombres.joblib")
    res["coeficientes"] = {c: round(float(v), 4) for c, v in zip(final, modelo.coef_)}
    return res


# ───────────────────────── Corazón ─────────────────────────
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]


def preparar_corazon() -> pd.DataFrame:
    df = pd.read_csv(DATOS / "processed.cleveland.data", names=COLS, na_values="?")
    df["enfermo"] = (df.num > 0).astype(int)
    # cp en UCI: 1 típica, 2 atípica, 3 no anginosa, 4 sin dolor
    for k, n in ((1, "dolor_tipico"), (2, "dolor_atipico"), (3, "dolor_no_anginoso")):
        df[n] = (df.cp == k).astype(int)
    return df


def corazon() -> dict:
    df = preparar_corazon()
    y = df.enfermo.to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
    dolor = ["dolor_tipico", "dolor_atipico", "dolor_no_anginoso"]
    conjuntos = {
        # Lo que cualquiera responde o mide en casa
        "basico": ["age", "sex", *dolor, "exang", "trestbps"],
        # + análisis de sangre comunes y frecuencia máxima (reloj/banda en ejercicio)
        "con_laboratorio": ["age", "sex", *dolor, "exang", "trestbps", "chol", "fbs", "thalach"],
        # Referencia: el de analysis.ipynb usa estudios de hospital (no se puede en la app)
        "hospital_referencia": ["age", "sex", *dolor, "exang", "trestbps", "chol", "fbs", "thalach", "oldpeak", "ca", "thal", "slope", "restecg"],
    }
    res = {"n": int(len(df)), "prevalencia": round(float(y.mean()), 3), "modelos": {}}
    for nombre, cols in conjuntos.items():
        d = df.dropna(subset=cols)
        yy = d.enfermo.to_numpy()
        aucs = []
        for tr, te in cv.split(d, yy):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(d.iloc[tr][cols], yy[tr])
            aucs.append(roc_auc_score(yy[te], m.predict_proba(d.iloc[te][cols])[:, 1]))
        aucs = np.array(aucs)
        info = {"variables": cols, "n": int(len(d)), "auc_cv": round(float(aucs.mean()), 3),
                "auc_ic90": [round(float(np.percentile(aucs, 5)), 3), round(float(np.percentile(aucs, 95)), 3)]}
        if nombre != "hospital_referencia":
            # Sensibilidad/especificidad en el umbral 0.5 con predicciones fuera de muestra
            p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
                                  d[cols], yy, cv=5, method="predict_proba")[:, 1]
            info["sensibilidad"] = round(float(((p >= 0.5) & (yy == 1)).sum() / (yy == 1).sum()), 3)
            info["especificidad"] = round(float(((p < 0.5) & (yy == 0)).sum() / (yy == 0).sum()), 3)
            final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(d[cols], yy)
            joblib.dump({"modelo": final, "variables": cols}, SALIDA / f"corazon_{nombre}.joblib")
        res["modelos"][nombre] = info
    return res


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    res = {"grasa_corporal": grasa(), "corazon": corazon(),
           "fuente": "~/Deep_Learning (bodyfat.csv, analysis.ipynb) + UCI Cleveland"}
    (SALIDA / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
