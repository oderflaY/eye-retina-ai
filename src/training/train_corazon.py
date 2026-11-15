"""IA cardiaca para la pulsera: aprende de latidos reales de PhysioNet convertidos a la señal que da una
Xiaomi Smart Band (pulso a 1 Hz, suavizado, sin RR) y se valida con PPG real (MIMIC PERform AF).

Modelos (cada uno con su ventana):
  fa      fibrilación/aleteo auricular contra ritmo sin FA                      120 s
  ectopia extrasístoles frecuentes (≥ 10 % latidos prematuros: bigeminia…)       120 s
  taqui   taquicardia ventricular / supraventricular en episodio                 60 s
  ic      patrón de insuficiencia cardiaca contra corazón sano                   120 s
  ms      patrón previo a muerte súbita (Holter > 30 min antes del paro)         120 s  (experimental)

Validación cruzada agrupada por persona.  Uso: .venv/bin/python -m src.training.train_corazon
Datos: ~/TemrariosSpace/datasets/Corazon (descargar.sh)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import wfdb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.config import PROJECT_ROOT
from src.corazon_senal import VARIABLES, latidos_a_banda, variables

DATOS = Path.home() / "TemrariosSpace/datasets/Corazon"
SALIDA = PROJECT_ROOT / "models/corazon"
LATIDOS = set("NLRBAaJSVrFejnE/fQ?")
ECTOPICOS = set("VAaJSEF")
FA_RITMOS = {"AFIB", "AFL"}
TAQUI_RITMOS = {"VT", "SVTA", "VFL", "VF"}
POR_CLASE = 250
rng = np.random.default_rng(7)

FUENTES = {  # db: (extensión de latidos, descripción)
    "afdb": ("qrs", "MIT-BIH Atrial Fibrillation (25 personas, 10 h)"),
    "ltafdb": ("atr", "Long Term AF (84 personas, 24 h)"),
    "mitdb": ("atr", "MIT-BIH Arrhythmia (47 personas)"),
    "svdb": ("atr", "MIT-BIH Supraventricular Arrhythmia (78 registros)"),
    "incartdb": ("atr", "St Petersburg INCART (32 pacientes coronarios)"),
    "nsrdb": ("atr", "MIT-BIH Normal Sinus Rhythm (18 personas)"),
    "nsr2db": ("ecg", "Normal Sinus Rhythm RR (54 personas)"),
    "chfdb": ("ecg", "BIDMC Congestive Heart Failure (15 personas, NYHA III-IV)"),
    "chf2db": ("ecg", "CHF RR Interval (29 personas, NYHA I-III)"),
    "sddb": ("atr", "Sudden Cardiac Death Holter (23 personas con paro cardiaco)"),
}
# registros que son la misma persona (mitdb 201/202; INCART: 75 registros de 32 pacientes, se agrupan por pares)
def persona(db: str, rec: str) -> str:
    if db == "mitdb" and rec in ("201", "202"):
        return "mitdb/201"
    if db == "incartdb":
        return f"incartdb/{(int(rec[1:]) - 1) // 2}"
    return f"{db}/{rec}"


def _hms(s: str) -> float:
    p = [float(x) for x in s.split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(p)))


def leer(db: str, rec: str) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, str]], float | None]:
    """Tiempos de latido (s), si cada latido es ectópico, ritmos (ini, fin, etiqueta) y hora del paro (sddb)."""
    base = str(DATOS / db / rec)
    h = wfdb.rdheader(base)
    fs = h.fs
    ext = FUENTES[db][0]
    a = wfdb.rdann(base, ext)
    sym = np.array(a.symbol)
    es = np.isin(sym, list(LATIDOS))
    t = a.sample[es] / fs
    ect = np.isin(sym[es], list(ECTOPICOS))
    if ext == "qrs":
        ect = np.zeros(len(t), bool)  # afdb: latidos sin clasificar
    ritmos = []
    if (DATOS / db / f"{rec}.atr").exists():
        r = a if ext == "atr" else wfdb.rdann(base, "atr")
        marcas = [(s / fs, n.strip("(\x00 ")) for s, n in zip(r.sample, r.aux_note) if n and n.startswith("(")]
        fin_reg = (h.sig_len or int(t[-1] * fs) + 1) / fs
        for i, (s, n) in enumerate(marcas):
            ritmos.append((s, marcas[i + 1][0] if i + 1 < len(marcas) else fin_reg, n))
    paro = None
    for c in h.comments or []:
        m = re.search(r"vfon:\s*([\d:]+)", c)
        if m:
            paro = _hms(m.group(1))
    orden = np.argsort(t)
    return t[orden], ect[orden], ritmos, paro


def fraccion(ritmos, ini: float, fin: float, cuales: set[str]) -> float:
    tot = sum(max(0.0, min(b, fin) - max(a, ini)) for a, b, n in ritmos if n in cuales)
    return tot / (fin - ini)


def ventanas(db: str, rec: str) -> list[dict]:
    t, ect, ritmos, paro = leer(db, rec)
    if len(t) < 300:
        return []
    if paro is not None:  # solo lo de antes: hasta 30 min previos al paro
        corte = paro - 1800
        t, ect = t[t < corte], ect[t < corte]
        if len(t) < 300:
            return []
    seg, serie = latidos_a_banda(t, rng, suave_s=float(rng.choice([3, 4, 5, 6, 8])))
    filas = []
    cupo: dict[tuple, int] = {}
    for largo in (120, 60):
        n = len(serie) // largo
        for k in rng.permutation(n):
            x = serie[k * largo:(k + 1) * largo]
            if np.isnan(x).mean() > 0.1:
                continue
            ini = seg[k * largo]
            fa = fraccion(ritmos, ini, ini + largo, FA_RITMOS) if ritmos else 0.0
            tq = fraccion(ritmos, ini, ini + largo, TAQUI_RITMOS) if ritmos else 0.0
            dentro = (t >= ini - 5) & (t < ini + largo)
            pe = float(ect[dentro].mean()) if dentro.sum() > 20 else np.nan
            # clase aproximada para repartir el cupo y no llenar todo con ritmo normal
            clase = (largo, "fa" if fa >= 0.8 else "tq" if tq >= 0.3 else "ec" if pe >= 0.1 else "n")
            if cupo.get(clase, 0) >= (POR_CLASE if clase[1] != "n" else (POR_CLASE if largo == 120 else 60)):
                continue
            v = variables(x)
            if not v:
                continue
            cupo[clase] = cupo.get(clase, 0) + 1
            filas.append(v | {"db": db, "persona": persona(db, rec), "largo": largo, "fa": fa, "tq": tq, "pe": pe})
    return filas


def registros(db: str) -> list[str]:
    return sorted(p.stem for p in (DATOS / db).glob("*.hea"))


def _modelo() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def evaluar(df: pd.DataFrame, nombre: str) -> dict:
    X, Y, G = df[VARIABLES].values, df["y"].values, df["persona"].values
    pred = np.zeros(len(df))
    solo = np.zeros(len(df))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in cv.split(X, Y, G):
        pred[te] = _modelo().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]
        m = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(df[["media"]].values[tr], Y[tr])
        solo[te] = m.predict_proba(df[["media"]].values[te])[:, 1]
    pos, neg = np.sort(pred[Y == 1]), np.sort(pred[Y == 0])
    # "vigilar": sensibilidad 90 % sin pasar de 10 % de falsas alarmas; "alto": 1 % de falsas alarmas
    u_vig = float(max(pos[int(0.10 * len(pos))], neg[int(0.90 * len(neg))]))
    u_alto = float(max(neg[int(0.99 * len(neg))], u_vig))
    def se(u): return float((pred[Y == 1] >= u).mean()), float((pred[Y == 0] < u).mean())
    s1, e1 = se(u_vig)
    s2, e2 = se(u_alto)
    dp = pd.DataFrame({"g": G, "y": Y, "p": pred}).groupby("g").agg(y=("y", "mean"), p=("p", "median"))
    dp = dp[(dp.y == 0) | (dp.y == 1)]
    auc_p = roc_auc_score(dp.y, dp.p) if dp.y.nunique() == 2 and len(dp) > 10 else None
    res = {"auc_ventana": round(roc_auc_score(Y, pred), 3), "auc_persona": None if auc_p is None else round(auc_p, 3),
           "auc_solo_pulso_medio": round(roc_auc_score(Y, solo), 3),
           "umbral_vigilar": round(u_vig, 4), "sens_vigilar": round(s1, 3), "esp_vigilar": round(e1, 3),
           "umbral_alto": round(u_alto, 4), "sens_alto": round(s2, 3), "esp_alto": round(e2, 3),
           "ventanas": int(len(df)), "positivas": int(Y.sum()), "personas": int(df.persona.nunique()),
           "personas_positivas": int(df[df.y == 1].persona.nunique())}
    print(nombre, json.dumps(res), flush=True)
    return res


def tareas(df: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, int]]:
    con_ritmo = ["afdb", "ltafdb", "mitdb", "svdb", "nsrdb", "nsr2db"]
    sanos = ["nsrdb", "nsr2db"]
    w120, w60 = df[df.largo == 120], df[df.largo == 60]
    t = {}
    d = w120[w120.db.isin(con_ritmo) & ((w120.fa >= 0.8) | (w120.fa == 0))].copy()
    d["y"] = (d.fa >= 0.8).astype(int); t["fa"] = (d, 120)
    d = w120[(w120.db != "afdb") & (w120.fa == 0) & (w120.tq == 0) & ((w120.pe >= 0.1) | (w120.pe < 0.01))].copy()
    d["y"] = (d.pe >= 0.1).astype(int); t["ectopia"] = (d, 120)
    d = w60[w60.db.isin(["ltafdb", "mitdb", "svdb", "nsrdb", "nsr2db"]) & (w60.fa == 0) & ((w60.tq >= 0.3) | (w60.tq == 0))].copy()
    d["y"] = (d.tq >= 0.3).astype(int); t["taqui"] = (d, 60)
    d = w120[w120.db.isin(["chfdb", "chf2db"] + sanos)].copy()
    d["y"] = d.db.isin(["chfdb", "chf2db"]).astype(int); t["ic"] = (d, 120)
    # muerte súbita contra sanos Y contra cardiópatas sin paro (para no aprender solo "está enfermo")
    d = w120[w120.db.isin(["sddb", "chf2db"] + sanos) & (w120.fa == 0)].copy()
    d["y"] = (d.db == "sddb").astype(int); t["ms"] = (d, 120)
    return t


def main() -> None:
    filas = []
    for db in FUENTES:
        for r in registros(db):
            try:
                filas += ventanas(db, r)
            except Exception as e:
                print(f"  {db}/{r}: {e}")
        print(f"{db}: {sum(f['db'] == db for f in filas)} ventanas", flush=True)
    df = pd.DataFrame(filas)

    SALIDA.mkdir(parents=True, exist_ok=True)
    resultados: dict = {}
    for clave, (d, largo) in tareas(df).items():
        resultados[clave] = evaluar(d, clave) | {"ventana_s": largo}
        m = _modelo().fit(d[VARIABLES].values, d["y"].values)
        joblib.dump({"modelo": m, "variables": VARIABLES, "ventana_s": largo}, SALIDA / f"{clave}.joblib")
    resultados["fuentes"] = {k: v[1] for k, v in FUENTES.items()}
    resultados["senal"] = "pulso a 1 Hz suavizado 3-8 s y redondeado, como Xiaomi Smart Band por BLE"
    previo = SALIDA / "resultados.json"
    if previo.exists() and "validacion_ppg" in (p := json.loads(previo.read_text())):
        resultados["validacion_ppg"] = p["validacion_ppg"]
    previo.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print("guardado en", SALIDA)


if __name__ == "__main__":
    main()
