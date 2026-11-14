"""Experimento: ¿qué le baja el error al clasificador de anemia con las mismas 218 fotos?

Compara, con validación cruzada estratificada repetida (10 × 5, mismas particiones para todos):

* ``color``: las 12 medidas de color de la conjuntiva (modelo actual).
* ``cnn_<red>``: características de una red preentrenada (ImageNet) sobre el recorte de la
  conjuntiva (fuera de la máscara en gris neutro) → PCA → regresión logística.
* combinaciones color + CNN y un ensamble que promedia probabilidades.

Paso 1 (GPU, una vez): ``--extraer`` guarda recortes y vectores en data/processed/anemia_feats.
Paso 2 (CPU): compara y escribe reports/anemia_exp/resultados.json.

    python -m src.training.experimento_anemia --extraer
    python -m src.training.experimento_anemia
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import PROJECT_ROOT

FEATS = PROJECT_ROOT / "data/processed/anemia_feats"
REDES = ("efficientnetv2b0", "resnet50", "convnext_tiny")
LADO = 224


def recorte_conjuntiva(rgb01: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Caja de la conjuntiva con margen; fuera de la máscara, gris neutro (no aporta color)."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = int(0.08 * max(y1 - y0, x1 - x0))
    y0, x0 = max(0, y0 - m), max(0, x0 - m)
    y1, x1 = min(mask.shape[0], y1 + m), min(mask.shape[1], x1 + m)
    img = rgb01[y0:y1, x0:x1].copy()
    img[~mask[y0:y1, x0:x1]] = 0.5
    h, w = img.shape[:2]
    lado = max(h, w)
    lienzo = np.full((lado, lado, 3), 0.5, np.float32)
    lienzo[(lado - h) // 2 : (lado - h) // 2 + h, (lado - w) // 2 : (lado - w) // 2 + w] = img
    return cv2.resize(lienzo, (LADO, LADO), interpolation=cv2.INTER_AREA)


def extraer() -> None:
    import tensorflow as tf

    from src.deployment.inference_anemia import SegmentadorConjuntiva, extraer as extraer_color

    med = pd.read_csv(PROJECT_ROOT / "reports/anemia/medidas.csv")
    seg = SegmentadorConjuntiva(PROJECT_ROOT / "models/conjuntiva/best.keras")
    FEATS.mkdir(parents=True, exist_ok=True)
    recortes, validos = [], []
    for r in med.itertuples():
        e = extraer_color(seg, cv2.cvtColor(cv2.imread(r.ruta), cv2.COLOR_BGR2RGB))
        ok = bool(e["valida"]) and e["mascara"].sum() > 200
        validos.append(ok)
        if not ok:
            recortes.append(np.full((LADO, LADO, 3), 0.5, np.float32))
            continue
        rgb = e["imagen"].astype(np.float32) / 255.0
        blanco = np.asarray(e["blanco"], np.float32)  # mismo balance de blancos que las medidas de color
        rgb = np.clip(rgb * (blanco.mean() / np.clip(blanco, 0.05, None)), 0, 1)
        recortes.append(recorte_conjuntiva(rgb, e["mascara"].astype(bool)))
    X = np.stack(recortes)
    np.save(FEATS / "recortes.npy", (X * 255).astype(np.uint8))
    np.save(FEATS / "validos.npy", np.array(validos))
    vis = np.vstack([np.hstack(list((X[i : i + 8] * 255).astype(np.uint8))) for i in range(0, 32, 8)])
    cv2.imwrite(str(FEATS / "muestra_recortes.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    apps = tf.keras.applications
    construir = {
        "efficientnetv2b0": (apps.EfficientNetV2B0, lambda x: x * 255.0),  # preprocesa por dentro
        "resnet50": (apps.ResNet50, lambda x: apps.resnet50.preprocess_input(x * 255.0)),
        "convnext_tiny": (apps.ConvNeXtTiny, lambda x: x * 255.0),
    }
    for nombre, (cls, prep) in construir.items():
        red = cls(include_top=False, weights="imagenet", pooling="avg", input_shape=(LADO, LADO, 3))
        vec = []
        for espejo in (False, True):  # TTA: promedio con el espejo
            Xi = X[:, :, ::-1] if espejo else X
            vec.append(red.predict(prep(Xi.astype(np.float32)), batch_size=16, verbose=0))
        np.save(FEATS / f"{nombre}.npy", np.mean(vec, 0))
        print(nombre, np.mean(vec, 0).shape)
        del red
        tf.keras.backend.clear_session()


def cv_auc(modelo, X: np.ndarray, y: np.ndarray, rep: int = 10) -> tuple[float, float, np.ndarray]:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=rep, random_state=42)
    aucs, oof, n = [], np.zeros(len(y)), np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m = clone(modelo).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        oof[te] += p
        n[te] += 1
    return float(np.mean(aucs)), float(np.std(aucs)), oof / n


def resumen(y: np.ndarray, p: np.ndarray) -> dict:
    """Sensibilidad/especificidad con sens ≥ 0.80 y con el umbral de Youden."""
    out = {}
    for t in np.unique(p)[::-1]:
        if (p >= t)[y == 1].mean() >= 0.80:
            out["sens80"] = {"sens": round(float((p >= t)[y == 1].mean()), 3),
                             "esp": round(float((p < t)[y == 0].mean()), 3)}
            break
    j = [((p >= t)[y == 1].mean() + (p < t)[y == 0].mean() - 1, t) for t in np.unique(p)]
    t = max(j)[1]
    out["youden"] = {"sens": round(float((p >= t)[y == 1].mean()), 3), "esp": round(float((p < t)[y == 0].mean()), 3)}
    return out


def comparar() -> None:
    from src.deployment.inference_anemia import CARACTERISTICAS

    med = pd.read_csv(PROJECT_ROOT / "reports/anemia/medidas.csv")
    ok = med.valida.to_numpy() & np.load(FEATS / "validos.npy")
    y = med.anemia.to_numpy()[ok]
    color = med.loc[ok, list(CARACTERISTICAS)].to_numpy()
    emb = {r: np.load(FEATS / f"{r}.npy")[ok] for r in REDES if (FEATS / f"{r}.npy").exists()}
    print(f"{ok.sum()} fotos ({y.sum()} con anemia)")

    def logit(c: float, pca: int | None = None):
        pasos = [StandardScaler()] + ([PCA(pca, random_state=0)] if pca else [])
        return make_pipeline(*pasos, LogisticRegression(C=c, max_iter=5000, class_weight="balanced"))

    pruebas: dict[str, tuple] = {
        "color_bosque": (RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced",
                                                random_state=0, n_jobs=-1), color),
        "color_logistica": (logit(0.3), color),
        "color_svm": (make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, class_weight="balanced",
                                                          random_state=0)), color),
    }
    for r, E in emb.items():
        for k in (16, 32):
            for c in (0.01, 0.1):
                pruebas[f"{r}_pca{k}_C{c}"] = (logit(c, k), E)
                pruebas[f"color+{r}_pca{k}_C{c}"] = (
                    make_pipeline(StandardScaler(), _PCAEnBloque(color.shape[1], k),
                                  LogisticRegression(C=c, max_iter=5000, class_weight="balanced")),
                    np.hstack([color, E]))

    res, oofs = {}, {}
    for nombre, (m, X) in pruebas.items():
        auc, sd, oof = cv_auc(m, X, y)
        res[nombre] = {"auc": round(auc, 4), "sd": round(sd, 4), **resumen(y, oof)}
        oofs[nombre] = oof
        print(f"{nombre:40s} AUC {auc:.3f} ± {sd:.3f}  {res[nombre]}")

    # Ensamble: promedio de rangos del mejor de color + los 2 mejores con CNN
    mejores_cnn = sorted([k for k in res if not k.startswith("color_")], key=lambda k: -res[k]["auc"])[:2]
    combo = ["color_bosque"] + mejores_cnn
    rank = np.mean([pd.Series(oofs[k]).rank(pct=True).to_numpy() for k in combo], 0)
    res["ensamble(" + "+".join(combo) + ")"] = {"auc_oof": round(float(roc_auc_score(y, rank)), 4), **resumen(y, rank),
                                                "nota": "AUC sobre predicciones fuera de partición promediadas"}
    for k in combo:
        res[k]["auc_oof"] = round(float(roc_auc_score(y, oofs[k])), 4)
    sal = PROJECT_ROOT / "reports/anemia_exp"
    sal.mkdir(parents=True, exist_ok=True)
    (sal / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    top = sorted(res.items(), key=lambda kv: -kv[1].get("auc", kv[1].get("auc_oof", 0)))[:8]
    print("\nMEJORES:")
    for k, v in top:
        print(f"  {k:45s} {v}")


class _PCAEnBloque:
    """Escala ya aplicada: deja las primeras ``n_color`` columnas y reduce el resto con PCA."""

    def __init__(self, n_color: int, k: int) -> None:
        self.n_color, self.k = n_color, k

    def fit(self, X, y=None):
        self.pca = PCA(self.k, random_state=0).fit(X[:, self.n_color :])
        return self

    def transform(self, X):
        return np.hstack([X[:, : self.n_color], self.pca.transform(X[:, self.n_color :])])

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_params(self, deep=True):
        return {"n_color": self.n_color, "k": self.k}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraer", action="store_true")
    a = ap.parse_args()
    extraer() if a.extraer else comparar()
