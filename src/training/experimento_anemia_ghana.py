"""Anemia por palma de la mano y uñas (datasets de Ghana, CC BY 4.0) con control de confusiones.

    python -m src.training.experimento_anemia_ghana

Datos: Mendeley ccr8cm22vz (palma) y 2xx4j3kjg2 (uñas); niños de 6–59 meses, anemia por Hb de
laboratorio. Cada niño viene 12 veces (copias rotadas/volteadas, el color es idéntico), así que se
usa UNA imagen por niño: si no, la misma persona caería en entrenamiento y prueba.

Controles para no confundir "la foto" con "la anemia":
* ``molestia``: solo tamaño del recorte, forma, área útil y nitidez. Si esto predice la anemia, las
  fotos de cada grupo se tomaron distinto y cualquier AUC está inflada.
* ``color residualizado``: el color después de quitarle lo que explican esas condiciones.
* Prueba de permutación (etiquetas barajadas) y validación cruzada 5 × 20.
* Lotes de nombre de archivo (Anemic- / AnemicP- / Anemic-Pa-): si un lote tiene color distinto,
  podría ser otra sesión de fotos.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT

GHANA = Path.home() / "TemrariosSpace/datasets/Anemia/ghana"
CONJUNTOS = {"palma": GHANA / "palma_x/Palm", "unas": GHANA / "unas_x/Fingernails"}
COLOR = ["L", "a", "b", "L_p25", "a_p75", "rojo", "sat", "a_sobre_L", "b_sobre_a", "chroma"]
MOLESTIA = ["ancho", "alto", "aspecto", "area_util", "nitidez"]


def medir(ruta: Path) -> dict[str, float] | None:
    im = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
    return None if im is None else medir_imagen(im)


def medir_imagen(im: np.ndarray) -> dict[str, float] | None:
    """``im`` en BGR o BGRA (alfa = zona útil), como lo lee OpenCV."""
    alfa = im[..., 3] > 127 if im.ndim == 3 and im.shape[2] == 4 else np.ones(im.shape[:2], bool)
    bgr = im[..., :3]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat_ok = (hsv[..., 2] > 0.12) & (hsv[..., 2] < 0.98)          # sin sombras ni brillos
    m = alfa & sat_ok
    if m.sum() < 80:
        return None
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[m]
    px = rgb[m]
    L, a, b = (float(np.median(lab[:, i])) for i in range(3))
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "L": L, "a": a, "b": b,
        "L_p25": float(np.percentile(lab[:, 0], 25)), "a_p75": float(np.percentile(lab[:, 1], 75)),
        "rojo": float(np.median((px[:, 0] - px[:, 1]) / (px[:, 0] + px[:, 1] + 1e-6))),
        "sat": float(np.median(hsv[..., 1][m])), "a_sobre_L": a / max(L, 1), "b_sobre_a": b / max(abs(a), 0.5),
        "chroma": float(np.hypot(a, b)),
        "ancho": float(max(im.shape[:2])), "alto": float(min(im.shape[:2])),
        "aspecto": max(im.shape[:2]) / min(im.shape[:2]), "area_util": float(alfa.mean()),
        "nitidez": float(np.log1p(cv2.Laplacian(gris, cv2.CV_64F)[alfa].var())),
    }


def tabla() -> pd.DataFrame:
    filas = []
    for conj, carpeta in CONJUNTOS.items():
        vistos = set()
        for p in sorted(carpeta.glob("*.png")):
            base = re.sub(r" \(\d+\)", "", p.stem)
            clase = 0 if base.lower().startswith("non") else 1
            num = re.findall(r"\d+", base)
            if not num:
                continue
            nino = int(num[0][:3]) if len(num[0]) > 3 else int(num[0])  # "Anmeic-fn-0011" → 1 (sic, 4 dígitos)
            clave = (clase, nino)
            if clave in vistos:
                continue
            r = medir(p)
            if r is None:
                continue
            vistos.add(clave)
            filas.append({"conjunto": conj, "anemia": clase, "nino": nino, "lote": re.match(r"\D+", base).group(0).lower(),
                          "archivo": p.name, **r})
    return pd.DataFrame(filas)


def cv_auc(modelo, X, y, rep=20, semilla=0) -> float:
    aucs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=rep, random_state=semilla).split(X, y):
        aucs.append(roc_auc_score(y[te], clone(modelo).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def residualizar(df: pd.DataFrame, cols: list[str], por: list[str]) -> np.ndarray:
    """Quita a cada medida de color lo que explican las condiciones de la foto (ajuste dentro de CV
    sería más estricto; aquí la regresión no usa la etiqueta, así que no filtra información de y)."""
    Z = StandardScaler().fit_transform(df[por])
    return np.column_stack([df[c] - LinearRegression().fit(Z, df[c]).predict(Z) for c in cols])


def main() -> None:
    df = tabla()
    rep = PROJECT_ROOT / "reports/anemia_ghana"
    rep.mkdir(parents=True, exist_ok=True)
    df.to_csv(rep / "medidas.csv", index=False)
    logi = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"))
    bosque = RandomForestClassifier(400, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1)
    res: dict = {}
    rng = np.random.default_rng(0)

    def probar(nombre, X, y, perm=False):
        r = {mn: cv_auc(m, X, y) for mn, m in (("logistica", logi), ("bosque", bosque))}
        mejor = max(r, key=lambda k: r[k][0])
        out = {"auc": round(r[mejor][0], 4), "sd": round(r[mejor][1], 4), "modelo": mejor}
        if perm:
            m = logi if mejor == "logistica" else bosque
            nulas = [cv_auc(m, X, rng.permutation(y), rep=1, semilla=i)[0] for i in range(40)]
            out["p_valor"] = round((1 + sum(n >= out["auc"] for n in nulas)) / 41, 4)
            out["azar_p95"] = round(float(np.percentile(nulas, 95)), 4)
        res[nombre] = out
        print(f"{nombre:38s} AUC {out['auc']:.3f} ± {out['sd']:.3f} ({mejor})" + (f" p={out['p_valor']}" if perm else ""))

    for conj in CONJUNTOS:
        d = df[df.conjunto == conj].reset_index(drop=True)
        y = d.anemia.to_numpy()
        print(f"\n== {conj}: {len(d)} niños ({y.sum()} con anemia)")
        res[f"{conj}/n"] = {"ninos": int(len(d)), "anemia": int(y.sum())}
        probar(f"{conj}/molestia (tamaño, forma, nitidez)", d[MOLESTIA].to_numpy(), y)
        probar(f"{conj}/color", d[COLOR].to_numpy(), y, perm=True)
        probar(f"{conj}/color sin efecto de la foto", residualizar(d, COLOR, MOLESTIA), y, perm=True)
        # ¿El lote del nombre (otra sesión) explica algo? Solo dentro de los anémicos.
        an = d[d.anemia == 1]
        lotes = an.lote.value_counts()
        print("   lotes anémicos:", lotes.to_dict())
        res[f"{conj}/lotes"] = {k: int(v) for k, v in lotes.items()}
        res[f"{conj}/auc_por_medida"] = {c: round(max(roc_auc_score(y, d[c]), 1 - roc_auc_score(y, d[c])), 3)
                                          for c in COLOR + MOLESTIA}
        # Solo el lote mayoritario de anémicos vs sanos: si el AUC se cae, el lote era parte del truco.
        mayor = lotes.index[0]
        s = d[(d.anemia == 0) | (d.lote == mayor)].reset_index(drop=True)
        probar(f"{conj}/color solo lote '{mayor}'", residualizar(s, COLOR, MOLESTIA), s.anemia.to_numpy())

    # Palma + uñas del mismo niño
    pa, un = (df[df.conjunto == c].set_index(["anemia", "nino"]) for c in CONJUNTOS)
    comun = pa.index.intersection(un.index)
    if len(comun) > 50:
        X = np.hstack([residualizar(pa.loc[comun].reset_index(), COLOR, MOLESTIA),
                       residualizar(un.loc[comun].reset_index(), COLOR, MOLESTIA)])
        y = np.array([i[0] for i in comun])
        print(f"\n== palma + uñas del mismo niño: {len(comun)}")
        probar("palma+uñas sin efecto de la foto", X, y, perm=True)

    (rep / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
