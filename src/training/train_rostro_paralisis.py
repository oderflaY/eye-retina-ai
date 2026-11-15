"""Detector de parálisis facial por asimetría (rasgos de MediaPipe → clasificador).

    .venv-etiquetas/bin/python src/data/rasgos_rostro.py ...   # antes: rasgos.csv
    python -m src.training.train_rostro_paralisis

Evaluación honesta:
* Validación cruzada por PERSONA (StratifiedGroupKFold, 7 particiones × 5 repeticiones): una
  persona nunca está en entrenamiento y prueba a la vez. Se reporta por cuadro y por persona
  (promedio de sus cuadros, que es lo que hace la app con varias fotos).
* Falsas alarmas en caras sanas que el modelo no vio (FFHQ, fotos de estudio).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT

YAW_MAX = 25.0


def columnas(df: pd.DataFrame, con_movimiento: bool) -> list[str]:
    c = [k for k in df.columns if k.startswith(("bs_", "geo_"))]
    if con_movimiento:
        c += [k for k in df.columns if k.startswith("bsm_")]
    return c


def candidatos() -> dict:
    return {
        "logistica": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced")),
        "bosque": RandomForestClassifier(500, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1),
        "extra": ExtraTreesClassifier(500, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, class_weight="balanced",
                                              random_state=0),
    }


def cv_por_persona(modelo, X, y, g, rep: int = 5) -> np.ndarray:
    oof = np.zeros(len(y))
    for r in range(rep):
        cv = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=r)
        for tr, te in cv.split(X, y, g):
            oof[te] += clone(modelo).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof / rep


def metricas(y, p, u=0.5) -> dict:
    pred = p >= u
    return {"auc": round(float(roc_auc_score(y, p)), 4), "sens": round(float(pred[y == 1].mean()), 4),
            "esp": round(float((~pred[y == 0]).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rasgos", type=Path, default=PROJECT_ROOT / "data/processed/rostro/rasgos.csv")
    ap.add_argument("--salida", type=Path, default=PROJECT_ROOT / "models/rostro")
    ap.add_argument("--reporte", type=Path, default=PROJECT_ROOT / "reports/rostro")
    a = ap.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)
    a.reporte.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.rasgos)
    df = df[df.yaw.abs() <= YAW_MAX].reset_index(drop=True)
    pal = df[df.fuente == "palsynet"].reset_index(drop=True)
    ffhq = df[df.fuente == "ffhq"].reset_index(drop=True)
    print(f"PalsyNet: {len(pal)} cuadros, {pal.grupo.nunique()} personas ({pal[pal.clase == 1].grupo.nunique()} con parálisis)"
          f" · FFHQ: {len(ffhq)} caras sanas")

    res = {}
    todo = pd.concat([pal, ffhq], ignore_index=True)
    es_ffhq = (todo.fuente == "ffhq").to_numpy()
    for mov in (False, True):
        cols = columnas(df, mov)
        for con_ffhq in (False, True):
            base = todo if con_ffhq else pal
            X, y, g = base[cols].to_numpy(), base.clase.to_numpy(), base.grupo.to_numpy()
            for nombre, m in candidatos().items():
                oof = cv_por_persona(m, X, y, g)
                sel = ~es_ffhq[: len(base)] if con_ffhq else np.ones(len(base), bool)
                per = pd.DataFrame({"g": g[sel], "y": y[sel], "p": oof[sel]}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
                if con_ffhq:   # FFHQ fuera de partición
                    pf = oof[es_ffhq]
                else:          # modelo entrenado en todo PalsyNet, probado en FFHQ
                    pf = clone(m).fit(X, y).predict_proba(ffhq[cols].to_numpy())[:, 1]
                clave = f"{nombre}{'+mov' if mov else ''}{'+ffhq' if con_ffhq else ''}"
                res[clave] = {"cuadro": metricas(y[sel], oof[sel]), "persona": metricas(per.y.to_numpy(), per.p.to_numpy()),
                              "ffhq_falsas_alarmas": round(float((pf >= 0.5).mean()), 4)}
                print(f"{clave:22s} cuadro {res[clave]['cuadro']} | persona {res[clave]['persona']} | "
                      f"FFHQ falsas alarmas {res[clave]['ffhq_falsas_alarmas']:.1%}")

    # Ganador: AUC por persona, penalizando falsas alarmas en caras sanas
    gana = max(res, key=lambda k: res[k]["persona"]["auc"] - res[k]["ffhq_falsas_alarmas"])
    mov = "+mov" in gana
    cols = columnas(df, mov)
    base = todo if "+ffhq" in gana else pal
    final = clone(candidatos()[gana.split("+")[0]]).fit(base[cols].to_numpy(), base.clase.to_numpy())
    imp = None
    if hasattr(final, "feature_importances_"):
        imp = dict(sorted(zip(cols, map(float, final.feature_importances_)), key=lambda kv: -kv[1])[:10])
    info = {"ganador": gana, "caracteristicas": cols, "yaw_max": YAW_MAX, **res[gana],
            "n_personas": int(pal.grupo.nunique()), "n_cuadros": int(len(pal)),
            "datos": "PalsyNet (CC BY 4.0) + FFHQ como control sano", "importancia": imp}
    joblib.dump({"modelo": final, "caracteristicas": cols, "validacion": info}, a.salida / "paralisis.joblib")
    (a.reporte / "paralisis.json").write_text(json.dumps({"comparacion": res, "final": info}, indent=2, ensure_ascii=False))
    print("GANADOR", json.dumps(info, indent=1, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
