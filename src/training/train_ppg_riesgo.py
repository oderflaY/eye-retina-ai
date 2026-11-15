"""Riesgos a partir de la FORMA del pulso PPG (hipertensión, diabetes, enfermedad cerebrovascular).

Datos:
  * PPG-BP (Liang et al., Sci Data 2018; figshare 5459299, CC0): 219 pacientes del Hospital de Guilin, 3 segmentos de
    2.1 s a 1 kHz (sensor de dedo clínico), presión con Omron, diagnósticos de hipertensión, diabetes tipo 2,
    infarto cerebral e insuficiencia/enfermedad cerebrovascular.
  * BUT PPG 2.0.0 (CC BY 4.0): 38 personas con presión medida + grabaciones de celular a 30 Hz → validación EXTERNA
    de hipertensión en condiciones reales de la app.

Honestidad:
  - Las señales de PPG-BP se bajan a 30 Hz (como la cámara) antes de sacar rasgos; también se reporta 1 kHz (techo).
  - Siempre se compara contra un modelo SOLO con edad, sexo e IMC: la edad sola ya predice mucho de esto.
  - Validación cruzada agrupada por persona (los 3 segmentos de alguien nunca están en train y test a la vez),
    repetida 20 veces; AUC con intervalo (percentiles 2.5–97.5 de las repeticiones).

Uso: .venv-rag/bin/python src/training/train_ppg_riesgo.py
Salida: reports/ppg/riesgo.json, models/ppg/riesgo.json (coeficientes, si algo sirve)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
PPGBP = Path.home() / "TemrariosSpace/datasets/PPG/ppgbp/Data File"
BUT = Path.home() / "TemrariosSpace/datasets/PPG/butppg/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0"


# ── rasgos de forma de onda ───────────────────────────────────────────────────
def latido_promedio(y: np.ndarray, fs: float) -> tuple[np.ndarray | None, float]:
    """Latidos pie-a-pie normalizados a 100 puntos y promediados. y con el latido hacia ARRIBA."""
    b, a = sg.butter(2, [0.5, min(8.0, fs / 2 * 0.9)], btype="band", fs=fs)
    z = sg.filtfilt(b, a, y - y.mean(), padlen=min(3 * 5, len(y) - 1))
    dist = int(fs * 0.33)
    pies, _ = sg.find_peaks(-z, distance=max(dist, 2), prominence=0.3 * z.std())
    latidos = []
    for p0, p1 in zip(pies, pies[1:]):
        dur = (p1 - p0) / fs
        if 0.33 <= dur <= 1.5:
            seg = z[p0:p1 + 1]
            seg = (seg - seg.min()) / max(seg.max() - seg.min(), 1e-9)
            latidos.append(np.interp(np.linspace(0, len(seg) - 1, 100), np.arange(len(seg)), seg))
    if not latidos:
        return None, np.nan
    periodo = float(np.median(np.diff(pies) / fs)) if len(pies) > 1 else np.nan
    return np.mean(latidos, 0), periodo


def rasgos_forma(y: np.ndarray, fs: float) -> dict | None:
    """Rasgos clásicos de forma de pulso (rigidez arterial) sobre el latido promedio."""
    if fs < 100:  # cámara: interpolar a 300 Hz antes de medir la forma
        t = np.arange(len(y)) / fs
        t2 = np.arange(0, t[-1], 1 / 300)
        y = np.interp(t2, t, y)
        fs = 300.0
    lat, periodo = latido_promedio(y, fs)
    if lat is None:
        return None
    ip = int(np.argmax(lat))
    d1 = np.gradient(lat)
    d2 = np.gradient(d1)
    # segunda derivada (APG): a = primer máximo, b = mínimo siguiente → b/a se asocia a edad y rigidez arterial
    a_i = int(np.argmax(d2[:max(ip, 2)]))
    b_i = a_i + int(np.argmin(d2[a_i:ip + 5]))
    a_v, b_v = d2[a_i], d2[b_i]
    sobre50 = np.where(lat >= 0.5)[0]
    return {
        "subida": ip / 100,                                    # tiempo pie→pico / periodo
        "ancho50": (sobre50[-1] - sobre50[0]) / 100 if len(sobre50) else 0.0,
        "area_sistolica": float(lat[:ip + 1].sum() / max(lat.sum(), 1e-9)),
        "pendiente_max": float(d1.max() * 100),
        "b_a": float(b_v / a_v) if a_v > 1e-9 else 0.0,
        "caida_tardia": float(lat[70:].mean()),               # cuánto queda arriba al final (reflexión de onda)
        "asimetria": float(pd.Series(lat).skew()),
        "periodo": periodo,
    }


FORMA = ["subida", "ancho50", "area_sistolica", "pendiente_max", "b_a", "caida_tardia", "asimetria", "periodo"]
DEMO = ["edad", "hombre", "imc"]


def cargar_ppgbp() -> pd.DataFrame:
    t = pd.read_excel(PPGBP / "PPG-BP dataset.xlsx", header=1)
    filas = []
    for _, r in t.iterrows():
        sid = int(r.subject_ID)
        for k in (1, 2, 3):
            f = PPGBP / "0_subject" / f"{sid}_{k}.txt"
            if not f.exists():
                continue
            x = np.loadtxt(f)
            x30 = sg.resample_poly(x, 3, 100)  # 1000 Hz → 30 Hz, como la cámara
            f1 = rasgos_forma(x, 1000.0)
            f30 = rasgos_forma(x30, 30.0)
            if f1 is None or f30 is None:
                continue
            fila = {"persona": sid, "edad": r["Age(year)"], "hombre": int(r["Sex(M/F)"] == "Male"), "imc": r["BMI(kg/m^2)"],
                    "sbp": r["Systolic Blood Pressure(mmHg)"], "dbp": r["Diastolic Blood Pressure(mmHg)"],
                    "hipertension": int(str(r.Hypertension).startswith("Stage")),
                    "diabetes": int(pd.notna(r.Diabetes)),
                    "cerebrovascular": int(pd.notna(r["cerebral infarction"]) or pd.notna(r["cerebrovascular disease"]))}
            fila.update({f"{k}_1k": v for k, v in f1.items()})
            fila.update({f"{k}_30": v for k, v in f30.items()})
            filas.append(fila)
    return pd.DataFrame(filas)


def auc_cv(df: pd.DataFrame, cols: list[str], y: str, reps: int = 20) -> dict:
    """AUC por PERSONA (promedio de la probabilidad de sus segmentos), validación agrupada repetida."""
    X = df[cols].to_numpy(float)
    yy = df[y].to_numpy(int)
    g = df.persona.to_numpy()
    aucs = []
    for rep in range(reps):
        p = np.zeros(len(df))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, yy, g):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))
            p[te] = m.fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
        pp = pd.DataFrame({"g": g, "y": yy, "p": p}).groupby("g").mean()
        aucs.append(roc_auc_score(pp.y, pp.p))
    a = np.array(aucs)
    return {"auc": round(float(a.mean()), 3), "ic95": [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]}


def cargar_but_presion() -> pd.DataFrame:
    import wfdb
    s = pd.read_csv(BUT / "subject-info.csv", encoding="utf-8-sig")
    q = pd.read_csv(BUT / "quality-hr-ann.csv", encoding="utf-8-sig")
    s = s.merge(q, on="ID")
    s = s[s["Blood pressure [mmHg]"].notna() & (s["Ear/finger"] == 1) & (s.Quality == 1)]
    filas = []
    for _, r in s.iterrows():
        rid = int(r.ID)
        x = wfdb.rdrecord(str(BUT / f"{rid}/{rid}_PPG")).p_signal
        rojo = -x.reshape(-1) if x.shape[0] == 1 else x[:, 0]
        f = rasgos_forma(-(rojo - rojo.mean()), 30.0)  # la sangre absorbe: invertir para tener el latido hacia arriba
        if f is None:
            continue
        sbp, dbp = (int(v) for v in str(r["Blood pressure [mmHg]"]).split("/"))
        filas.append({"persona": rid // 1000, "edad": r["Age [years]"], "hombre": int(r.Gender == "M"),
                      "imc": r["Weight [kg]"] / (r["Height [cm]"] / 100) ** 2 if pd.notna(r["Height [cm]"]) else np.nan,
                      "sbp": sbp, "dbp": dbp, "hipertension": int(sbp >= 140 or dbp >= 90),
                      **{f"{k}_30": v for k, v in f.items()}})
    return pd.DataFrame(filas)


def main() -> None:
    df = cargar_ppgbp()
    print(f"PPG-BP: {len(df)} segmentos, {df.persona.nunique()} personas; "
          f"hipertensión {df.groupby('persona').hipertension.first().sum()}, diabetes {df.groupby('persona').diabetes.first().sum()}, "
          f"cerebrovascular {df.groupby('persona').cerebrovascular.first().sum()}")
    f30 = [f"{k}_30" for k in FORMA]
    f1k = [f"{k}_1k" for k in FORMA]
    res: dict = {"ppgbp": {"segmentos": len(df), "personas": int(df.persona.nunique())}}
    for y in ["hipertension", "diabetes", "cerebrovascular"]:
        res[y] = {
            "solo_edad_sexo_imc": auc_cv(df, DEMO, y),
            "solo_pulso_30hz": auc_cv(df, f30, y),
            "solo_pulso_1khz": auc_cv(df, f1k, y),
            "edad_sexo_imc_mas_pulso_30hz": auc_cv(df, DEMO + f30, y),
            "positivos": int(df.groupby("persona")[y].first().sum()),
        }
        print(y, json.dumps(res[y]))
    # ¿la forma de pulso refleja la edad (rigidez arterial)? correlación por persona
    pp = df.groupby("persona")[["edad", "b_a_30", "subida_30", "b_a_1k"]].mean()
    res["correlacion_edad"] = {c: round(float(pp[c].corr(pp.edad)), 3) for c in ["b_a_30", "subida_30", "b_a_1k"]}
    print("correlación con la edad:", res["correlacion_edad"])

    # Externo: entrenar en PPG-BP a 30 Hz, probar en BUT PPG (celular) para hipertensión ≥ 140/90
    bu = cargar_but_presion()
    if len(bu):
        bp = bu.groupby("persona").agg({**{c: "mean" for c in f30 + ["edad", "hombre", "imc", "sbp", "dbp"]}, "hipertension": "first"})
        res["but_externo"] = {"personas": len(bp), "hipertensas_140_90": int(bp.hipertension.sum())}
        for nombre, cols in {"solo_pulso_30hz": f30, "edad_sexo_mas_pulso": ["edad", "hombre"] + f30, "solo_edad_sexo": ["edad", "hombre"]}.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))
            tr = df.groupby("persona").agg({**{c: "mean" for c in cols}, "hipertension": "first"})
            m.fit(tr[cols].to_numpy(float), tr.hipertension.to_numpy(int))
            ok = bp[cols].notna().all(1)
            if bp[ok].hipertension.nunique() == 2:
                res["but_externo"][nombre] = round(float(roc_auc_score(bp[ok].hipertension, m.predict_proba(bp[ok][cols].to_numpy(float))[:, 1])), 3)
        print("BUT externo:", res["but_externo"])

    (RAIZ / "reports/ppg").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RAIZ / "reports/ppg/riesgo.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
