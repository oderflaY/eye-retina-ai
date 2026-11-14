"""Entrena el clasificador de palidez de conjuntiva (anemia sí/no).

    python -m src.training.train_anemia [--datos ~/TemrariosSpace/datasets/Anemia/eyes]

Datos: Eyes-defy-anemia (Dimauro et al., IEEE Dataport 2022; copia en HuggingFace
``Yahaira/anemia-eyes``): 218 fotos de conjuntiva con etiqueta Anemia / NoAnemia.

Con tan pocas fotos, una CNN memoriza; se usa el camino clínico: segmentar la conjuntiva,
corregir la luz con el blanco del ojo y medir el color (``inference_anemia``). Se comparan
regresión logística y bosque aleatorio con validación cruzada repetida (10× 5 particiones) y
se guarda el mejor. El umbral se elige sobre predicciones fuera de partición para priorizar
la sensibilidad (que no se escape una anemia): sensibilidad ≥ 0.80.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT
from src.deployment.inference_anemia import CARACTERISTICAS, SegmentadorConjuntiva, extraer

SENS_OBJETIVO = 0.80


def leer_datos(raiz: Path) -> pd.DataFrame:
    filas = []
    for p in sorted(raiz.rglob("*.jpg")) + sorted(raiz.rglob("*.png")):
        clase = p.parent.name
        if clase in ("Anemia", "NoAnemia"):
            filas.append({"ruta": str(p), "anemia": int(clase == "Anemia"), "origen": p.parent.parent.name})
    return pd.DataFrame(filas)


def medir_todo(df: pd.DataFrame, seg: SegmentadorConjuntiva, figuras: Path) -> pd.DataFrame:
    figuras.mkdir(parents=True, exist_ok=True)
    filas = []
    for i, r in df.iterrows():
        bgr = cv2.imread(r.ruta)
        e = extraer(seg, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        fila = {"ruta": r.ruta, "anemia": r.anemia, "valida": e["valida"],
                "fraccion_conjuntiva": e["fraccion_conjuntiva"], "pixeles_blanco": e.get("pixeles_blanco", 0)}
        if e["valida"]:
            fila.update(e["medidas"])
        filas.append(fila)
        if i < 24:  # muestra para revisar la segmentación a ojo
            vis = cv2.cvtColor(e["imagen"], cv2.COLOR_RGB2BGR)
            cont, _ = cv2.findContours(e["mascara"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cont, -1, (0, 255, 0), 2)
            cv2.putText(vis, "ANEMIA" if r.anemia else "normal", (8, 28), 0, 0.9, (0, 255, 255), 2)
            cv2.imwrite(str(figuras / f"seg_{i:03d}.jpg"), vis)
    return pd.DataFrame(filas)


def candidatos() -> dict:
    return {
        "logistica": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000,
                                                                        class_weight="balanced")),
        "bosque": RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced", random_state=0,
                                         n_jobs=-1),
    }


def validar(modelo, X: np.ndarray, y: np.ndarray, rep: int = 10) -> tuple[float, float, np.ndarray]:
    """AUC media ± desviación en validación cruzada repetida y probabilidades fuera de partición
    (promediadas entre repeticiones)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=rep, random_state=42)
    aucs, oof, cuenta = [], np.zeros(len(y)), np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m = clone(modelo).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        oof[te] += p
        cuenta[te] += 1
    return float(np.mean(aucs)), float(np.std(aucs)), oof / cuenta


def umbral_sensible(y: np.ndarray, p: np.ndarray, sens: float = SENS_OBJETIVO) -> dict:
    """El umbral más alto que mantiene la sensibilidad objetivo (menos falsas alarmas)."""
    mejor = None
    for t in np.unique(p)[::-1]:
        pred = p >= t
        s = pred[y == 1].mean()
        if s >= sens:
            mejor = t
            break
    t = float(mejor if mejor is not None else np.median(p))
    pred = p >= t
    return {"umbral": round(t, 4), "sensibilidad": round(float(pred[y == 1].mean()), 4),
            "especificidad": round(float((~pred[y == 0]).mean()), 4),
            "exactitud": round(float((pred == y).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=Path.home() / "TemrariosSpace/datasets/Anemia/eyes")
    ap.add_argument("--seg", type=Path, default=PROJECT_ROOT / "models/conjuntiva/best.keras")
    ap.add_argument("--salida", type=Path, default=PROJECT_ROOT / "models/anemia")
    ap.add_argument("--reporte", type=Path, default=PROJECT_ROOT / "reports/anemia")
    a = ap.parse_args()
    t0 = time.time()
    a.salida.mkdir(parents=True, exist_ok=True)
    a.reporte.mkdir(parents=True, exist_ok=True)

    df = leer_datos(a.datos.expanduser())
    print(f"{len(df)} fotos ({df.anemia.sum()} con anemia)")
    med = medir_todo(df, SegmentadorConjuntiva(a.seg), a.reporte / "segmentacion")
    med.to_csv(a.reporte / "medidas.csv", index=False)
    ok = med[med.valida].reset_index(drop=True)
    print(f"conjuntiva encontrada en {len(ok)}/{len(med)}; sin referencia de blanco: "
          f"{int((ok.pixeles_blanco < 150).sum())}")
    X, y = ok[list(CARACTERISTICAS)].to_numpy(), ok.anemia.to_numpy()

    # Cada medida por separado: ¿cuánto separa por sí sola? (AUC; < 0.5 = más baja con anemia)
    individual = {k: round(float(roc_auc_score(y, ok[k])), 4) for k in CARACTERISTICAS}
    print("AUC por medida:", individual)

    res = {}
    for nombre, modelo in candidatos().items():
        auc, sd, oof = validar(modelo, X, y)
        res[nombre] = {"auc_cv": round(auc, 4), "auc_sd": round(sd, 4), **umbral_sensible(y, oof),
                       "_oof": oof}
        print(f"{nombre}: AUC {auc:.3f} ± {sd:.3f} | {umbral_sensible(y, oof)}")
    gana = max(res, key=lambda k: res[k]["auc_cv"])
    final = candidatos()[gana].fit(X, y)
    val = {k: v for k, v in res[gana].items() if not k.startswith("_")}
    val.update(modelo=gana, n=int(len(y)), n_anemia=int(y.sum()), datos="Eyes-defy-anemia (Dimauro 2022)",
               metodo="validación cruzada estratificada 10×5")
    joblib.dump({"modelo": final, "umbral": val["umbral"], "caracteristicas": list(CARACTERISTICAS),
                 "validacion": val}, a.salida / "clasificador.joblib")
    reporte = {"ganador": gana, "validacion": val,
               "comparacion": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()},
               "auc_por_medida": individual, "fotos": int(len(med)), "con_conjuntiva": int(len(ok)),
               "minutos": round((time.time() - t0) / 60, 1)}
    (a.reporte / "resultados.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False))
    (a.salida / "config.json").write_text(json.dumps(val, indent=2, ensure_ascii=False))
    print(json.dumps(reporte, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
