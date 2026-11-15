"""¿Se puede detectar la anemia por la PIEL de la cara? Experimento con etiquetas reales.

    python -m src.training.experimento_anemia_piel

No hay datasets públicos de caras con hemoglobina. Pero las fotos de Eyes-defy-anemia (218
personas, anemia confirmada por análisis de sangre) traen piel alrededor del ojo: párpado,
ojera, mejilla. Se quita la conjuntiva (U-Net), el blanco del ojo, el iris y el aparato, y se mide
solo la piel. Se comparan con validación cruzada repetida (5 × 20):

* ``piel``: solo color de piel de la cara  → ¿la cara sola sirve?
* ``conjuntiva``: el modelo actual del ojo  → referencia
* ``conjuntiva+piel``: ambas de la misma foto → ¿la piel le suma algo al ojo?

Y una prueba de permutación (etiquetas barajadas) para saber si la AUC de la piel es mejor
que el azar o solo suerte.
"""

from __future__ import annotations

import json
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

DATOS = Path.home() / "TemrariosSpace/datasets/Anemia/eyes"
PIEL_MIN = 400
PIEL = ["piel_L", "piel_a", "piel_b", "piel_rojo", "piel_sat", "pielw_L", "pielw_a", "pielw_b", "pielw_rojo"]


def mascara_piel(img: np.ndarray, conj: np.ndarray, util: np.ndarray) -> np.ndarray:
    """Piel: rango de piel en YCrCb, sin conjuntiva (dilatada), sin blanco del ojo, sin sombras."""
    u8 = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
    ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    piel = (ycc[..., 1] >= 133) & (ycc[..., 1] <= 180) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 135)
    piel &= hsv[..., 2] >= 50          # sin sombras profundas
    piel &= hsv[..., 1] >= 30          # sin blanco del ojo ni el aro blanco del aparato
    lejos = cv2.dilate(conj.astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
    piel &= lejos & util.astype(bool)
    piel = cv2.morphologyEx(piel.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    return piel


def medir_piel(img: np.ndarray, piel: np.ndarray, blanco: np.ndarray) -> dict[str, float]:
    rgb = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
    px = rgb[piel]
    pw = np.clip(px / np.clip(np.asarray(blanco, np.float32), 1e-3, None), 0, 1)  # luz corregida
    lab = cv2.cvtColor(px[None], cv2.COLOR_RGB2Lab)[0]
    labw = cv2.cvtColor(pw[None].astype(np.float32), cv2.COLOR_RGB2Lab)[0]
    hsv = cv2.cvtColor(px[None], cv2.COLOR_RGB2HSV)[0]
    rojo = lambda p: float(np.median((p[:, 0] - p[:, 1]) / (p[:, 0] + p[:, 1] + 1e-6)))  # noqa: E731
    return {
        "piel_L": float(np.median(lab[:, 0])), "piel_a": float(np.median(lab[:, 1])), "piel_b": float(np.median(lab[:, 2])),
        "piel_rojo": rojo(px), "piel_sat": float(np.median(hsv[:, 1])),
        "pielw_L": float(np.median(labw[:, 0])), "pielw_a": float(np.median(labw[:, 1])),
        "pielw_b": float(np.median(labw[:, 2])), "pielw_rojo": rojo(pw),
    }


def cv_auc(modelo, X, y, rep=20, semilla=0) -> tuple[float, float, np.ndarray]:
    aucs, oof, n = [], np.zeros(len(y)), np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=rep, random_state=semilla).split(X, y):
        p = clone(modelo).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        oof[te] += p
        n[te] += 1
    return float(np.mean(aucs)), float(np.std(aucs)), oof / n


def main() -> None:
    seg = SegmentadorConjuntiva(PROJECT_ROOT / "models/conjuntiva/best.keras")
    filas, ejemplos = [], []
    for p in sorted(DATOS.rglob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png") or p.parent.name not in ("Anemia", "NoAnemia"):
            continue
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        e = extraer(seg, rgb)
        if not e["valida"]:
            continue
        img = e["imagen"]
        _, util = __import__("src.deployment.inference_anemia", fromlist=["letterbox"]).letterbox(rgb)
        piel = mascara_piel(img, e["mascara"], util)
        fila = {"ruta": str(p), "anemia": int(p.parent.name == "Anemia"), "n_piel": int(piel.sum()),
                **{k: e["medidas"][k] for k in CARACTERISTICAS}}
        if piel.sum() >= PIEL_MIN:
            fila.update(medir_piel(img, piel, np.array(e["blanco"])))
        filas.append(fila)
        if len(ejemplos) < 8 and piel.sum() >= PIEL_MIN:
            vis = img.copy() if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
            vis[piel] = (0.5 * vis[piel] + [0, 110, 0]).astype(np.uint8)
            vis[e["mascara"].astype(bool)] = (0.5 * vis[e["mascara"].astype(bool)] + [110, 0, 110]).astype(np.uint8)
            ejemplos.append(cv2.resize(vis, (200, 200)))
    df = pd.DataFrame(filas)
    rep = PROJECT_ROOT / "reports/anemia_piel"
    rep.mkdir(parents=True, exist_ok=True)
    df.to_csv(rep / "medidas.csv", index=False)
    if ejemplos:
        cv2.imwrite(str(rep / "mascaras.jpg"), cv2.cvtColor(np.hstack(ejemplos), cv2.COLOR_RGB2BGR))

    con = df.dropna(subset=PIEL).reset_index(drop=True)
    y = con.anemia.to_numpy()
    print(f"Fotos válidas: {len(df)} · con piel suficiente: {len(con)} ({int(y.sum())} anemia)")
    logi = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"))
    bosque = RandomForestClassifier(400, min_samples_leaf=4, class_weight="balanced", random_state=0, n_jobs=-1)
    conj = list(CARACTERISTICAS)
    res = {}
    for nombre, cols in (("piel", PIEL), ("conjuntiva", conj), ("conjuntiva+piel", conj + PIEL)):
        for mn, m in (("logistica", logi), ("bosque", bosque)):
            auc, sd, oof = cv_auc(m, con[cols].to_numpy(), y)
            res[f"{nombre}/{mn}"] = {"auc": round(auc, 4), "sd": round(sd, 4)}
            print(f"{nombre:18s} {mn:9s} AUC {auc:.3f} ± {sd:.3f}")

    # Prueba de permutación para la piel sola (¿mejor que el azar?)
    mejor_piel = max(("logistica", "bosque"), key=lambda m: res[f"piel/{m}"]["auc"])
    real = res[f"piel/{mejor_piel}"]["auc"]
    rng = np.random.default_rng(0)
    m = logi if mejor_piel == "logistica" else bosque
    nulas = [cv_auc(m, con[PIEL].to_numpy(), rng.permutation(y), rep=2, semilla=i)[0] for i in range(60)]
    pval = (1 + sum(n >= real for n in nulas)) / (1 + len(nulas))
    # Cada medida sola (AUC univariada; 0.5 = no sirve)
    uni = {k: round(float(max(roc_auc_score(y, con[k]), 1 - roc_auc_score(y, con[k]))), 3) for k in PIEL}
    resumen = {"n": len(con), "n_anemia": int(y.sum()), "n_sin_piel": int(len(df) - len(con)), "modelos": res,
               "piel_permutacion": {"auc_real": real, "auc_azar_media": round(float(np.mean(nulas)), 4),
                                    "auc_azar_p95": round(float(np.percentile(nulas, 95)), 4), "p_valor": round(pval, 4)},
               "piel_auc_por_medida": uni}
    (rep / "resultados.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps({k: resumen[k] for k in ("piel_permutacion", "piel_auc_por_medida")}, indent=1))


if __name__ == "__main__":
    main()
