"""Anemia por conjuntiva palpebral con la HEMOGLOBINA real (Eyes-Defy-Anemia original, Kaggle).

    python -m src.training.train_anemia_hb extraer   # rasgos + embeddings → reports/anemia_hb/rasgos.parquet
    python -m src.training.train_anemia_hb evaluar   # validación por persona, por país, externo Bangladesh

Datos: ~/TemrariosSpace/datasets/Anemia/kaggle_eyesdefy/dataset anemia/{India,Italy}/<n>/
  * foto original (.jpg) + recorte A MANO de la conjuntiva palpebral (*_palpebral.png; fuera = transparente o blanco)
  * India.xlsx / Italy.xlsx: Number, Hgb (g/dL), Gender, Age
Anemia = criterio OMS: Hb < 12 (mujer) / < 13 (hombre).

Tres familias de rasgos, todas con 1 foto = 1 persona (no hay fuga):
  mano  : color dentro del recorte a mano, balance de blancos con la esclera de la foto original
  app   : la MISMA tubería que usa la app (U-Net models/conjuntiva → medir_color)
  cnn   : embedding de EfficientNetV2-B0 (ImageNet, congelado) del recorte a mano, 224 px
La validación compara solo dentro de cada país (India tiene más anemia y piel más oscura: el país confunde).
Externo: Bangladesh (Kaggle nahiyan1402; 104 personas, % R/G/B de la conjuntiva + Hb).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

D = Path.home() / "TemrariosSpace/datasets/Anemia/kaggle_eyesdefy/dataset anemia"
BANG = Path.home() / "TemrariosSpace/datasets/Anemia/kaggle_bangladesh/Anemia_Dataset.csv"
OUT = PROJECT_ROOT / "reports/anemia_hb"


def leer(f: Path) -> np.ndarray | None:
    x = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    if x is None:
        return None
    if x.dtype == np.uint16:
        x = (x / 257).astype(np.uint8)
    return x


def recorte_mano(f: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """RGB float 0..1 y máscara de la conjuntiva del PNG recortado a mano."""
    x = leer(f)
    if x is None or x.ndim != 3:
        return None
    if x.shape[2] == 4:
        a = x[..., 3] > 0
        rgb = cv2.cvtColor(x[..., :3], cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        a = np.ones(x.shape[:2], bool)
    blanco_puro = (rgb.min(2) >= 250)
    negro_puro = (rgb.max(2) <= 3)
    m = a & ~blanco_puro & ~negro_puro
    if m.sum() < 200:
        return None
    return rgb.astype(np.float32) / 255.0, m


def blanco_global(rgb: np.ndarray) -> np.ndarray:
    """Referencia de blanco de la foto completa (esclera/aro): píxeles claros y poco saturados."""
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255
    s, v = hsv[..., 1], hsv[..., 2]
    c = (s < 0.6) & (v > 0.35) & (v < 0.995)
    if c.sum() < 100:
        return np.ones(3, np.float32) * float(rgb.mean())
    cl = c & (v >= np.percentile(v[c], 75))
    sel = cl & (s <= np.percentile(s[cl], 50))
    return np.median(rgb[sel], axis=0).astype(np.float32)


def tabla() -> pd.DataFrame:
    filas = []
    for pais in ("India", "Italy"):
        t = pd.read_excel(D / pais / f"{pais}.xlsx")
        for _, r in t.iterrows():
            hb = pd.to_numeric(str(r.Hgb).replace(",", ".").strip(), errors="coerce")
            if pd.isna(hb) or pd.isna(pd.to_numeric(r.Number, errors="coerce")):
                continue
            n = int(r.Number)
            carp = D / pais / str(n)
            if not carp.exists():
                continue
            fotos = [f for f in carp.iterdir() if f.suffix.lower() in (".jpg", ".jpeg") and "palpebral" not in f.name]
            mascaras = [f for f in carp.iterdir() if f.name.endswith("_palpebral.png") and "forniceal" not in f.name]
            sexo = str(r.Gender).strip().upper()[:1]
            filas.append({"persona": f"{pais}_{n}", "pais": pais, "hb": float(hb), "mujer": int(sexo == "F"),
                          "edad": float(pd.to_numeric(str(r.Age).replace(",", "."), errors="coerce")),
                          "foto": str(fotos[0]) if fotos else None, "mascara": str(mascaras[0]) if mascaras else None})
    d = pd.DataFrame(filas)
    d["anemia"] = ((d.mujer == 1) & (d.hb < 12) | (d.mujer == 0) & (d.hb < 13)).astype(int)
    return d


def extraer() -> None:
    import tensorflow as tf
    from src.deployment.inference_anemia import SegmentadorConjuntiva, extraer as extraer_app, medir_color

    d = tabla()
    print(d.groupby("pais").agg(n=("hb", "size"), anemia=("anemia", "mean"), hb=("hb", "mean")).round(2))
    seg = SegmentadorConjuntiva(PROJECT_ROOT / "models/conjuntiva/best.keras")
    base = tf.keras.applications.EfficientNetV2B0(include_top=False, pooling="avg", weights="imagenet")
    filas, emb = [], []
    for i, r in d.iterrows():
        fila = {"persona": r.persona}
        foto = leer(Path(r.foto)) if isinstance(r.foto, str) else None
        rgb_foto = cv2.cvtColor(foto, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 if foto is not None else None
        # app: misma tubería que en producción
        if foto is not None:
            e = extraer_app(seg, cv2.cvtColor(foto, cv2.COLOR_BGR2RGB))
            fila["app_valida"] = int(e["valida"])
            if e["valida"]:
                fila.update({f"app_{k}": v for k, v in e["medidas"].items()})
        # mano
        rm = recorte_mano(Path(r.mascara)) if isinstance(r.mascara, str) else None
        v = np.full(1280, np.nan, np.float32)
        if rm is not None:
            rgb, m = rm
            bl = blanco_global(rgb_foto) if rgb_foto is not None else np.ones(3, np.float32)
            fila.update({f"mano_{k}": val for k, val in medir_color(rgb, m, bl).items()})
            sin = medir_color(rgb, m, np.ones(3, np.float32))  # sin balance de blancos
            fila.update({f"cruda_{k}": val for k, val in sin.items()})
            px = rgb[m]
            s = px.sum(0)
            fila.update(pct_r=100 * s[0] / s.sum(), pct_g=100 * s[1] / s.sum(), pct_b=100 * s[2] / s.sum())
            # recorte cuadrado alrededor de la conjuntiva, fondo gris neutro → EfficientNet
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            c = rgb.copy()
            c[~m] = 0.5
            c = c[y0:y1, x0:x1]
            lado = max(c.shape[:2])
            lienzo = np.full((lado, lado, 3), 0.5, np.float32)
            oy, ox = (lado - c.shape[0]) // 2, (lado - c.shape[1]) // 2
            lienzo[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
            x = cv2.resize(lienzo, (224, 224), interpolation=cv2.INTER_AREA)[None] * 255
            v = base(x, training=False).numpy()[0]
            vf = base(x[:, :, ::-1], training=False).numpy()[0]
            v = (v + vf) / 2
        emb.append(v)
        filas.append(fila)
        if i % 20 == 0:
            print(i, r.persona, "app" if fila.get("app_valida") else "-", "mano" if rm is not None else "-", flush=True)
    r = d.merge(pd.DataFrame(filas), on="persona")
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_parquet(OUT / "rasgos.parquet")
    np.save(OUT / "embeddings.npy", np.array(emb))
    print("listo", len(r), "mano", r.mano_eritema.notna().sum(), "app", r.app_eritema.notna().sum())


def evaluar() -> None:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    from src.deployment.inference_anemia import CARACTERISTICAS

    r = pd.read_parquet(OUT / "rasgos.parquet")
    E = np.load(OUT / "embeddings.npy")
    r["pais_india"] = (r.pais == "India").astype(int)
    conjuntos = {
        "solo_pais_sexo_edad": ["pais_india", "mujer", "edad"],
        "mano_color": [f"mano_{k}" for k in CARACTERISTICAS],
        "mano_color_crudo": [f"cruda_{k}" for k in CARACTERISTICAS],
        "pct_rgb": ["pct_r", "pct_g", "pct_b"],
        "app_unet_color": [f"app_{k}" for k in CARACTERISTICAS],
        "mano_color+sexo+edad": [f"mano_{k}" for k in CARACTERISTICAS] + ["mujer", "edad"],
        "cnn_embedding": "E",
        "cnn+mano_color": "E+",
    }
    modelos = {
        "logistica": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced")),
        "bosque": lambda: RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1),
    }
    res: dict = {"n": {p: {"personas": int((r.pais == p).sum()), "anemia": int(r[r.pais == p].anemia.sum())} for p in ("India", "Italy")}}
    REP = 10

    def matriz(cols, sub):
        if cols == "E":
            return E[sub.index]
        if cols == "E+":
            return np.hstack([E[sub.index], sub[[f"mano_{k}" for k in CARACTERISTICAS]].to_numpy()])
        return sub[cols].to_numpy(float)

    for nombre, cols in conjuntos.items():
        need = [f"mano_{k}" for k in CARACTERISTICAS] if cols in ("E", "E+") else cols
        sub = r[r[need].notna().all(1)] if isinstance(need, list) else r
        X = matriz(cols, sub)
        y = sub.anemia.to_numpy()
        estrato = sub.pais_india.to_numpy() * 2 + y
        for mn, mk in modelos.items():
            if cols in ("E", "E+") and mn == "bosque":
                continue
            fab = mk
            if cols in ("E", "E+"):
                fab = lambda: make_pipeline(StandardScaler(), PCA(32, random_state=0),
                                            LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
            aucs, aucs_in, aucs_it = [], [], []
            for rep in range(REP):
                p = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, estrato):
                    p[te] = fab().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                aucs.append(roc_auc_score(y, p))
                ind = (sub.pais == "India").to_numpy()
                aucs_in.append(roc_auc_score(y[ind], p[ind]))
                aucs_it.append(roc_auc_score(y[~ind], p[~ind]))
            dentro = (np.mean(aucs_in) * ind.sum() + np.mean(aucs_it) * (~ind).sum()) / len(y)
            res[f"{nombre}|{mn}"] = {"n": len(y), "auc_global": round(float(np.mean(aucs)), 3),
                                     "auc_india": round(float(np.mean(aucs_in)), 3), "auc_italia": round(float(np.mean(aucs_it)), 3),
                                     "auc_dentro_pais": round(float(dentro), 3)}
            print(f"{nombre:24s} {mn:9s} n={len(y)} global {np.mean(aucs):.3f}  India {np.mean(aucs_in):.3f}  Italia {np.mean(aucs_it):.3f}  dentro {dentro:.3f}", flush=True)

    # Hemoglobina (g/dL): error contra predecir la media de su país
    for nombre, cols in [("mano_color+pais+sexo", [f"mano_{k}" for k in CARACTERISTICAS] + ["pais_india", "mujer"]),
                         ("solo_pais_sexo", ["pais_india", "mujer"])]:
        sub = r[r[cols].notna().all(1)]
        X, hb = sub[cols].to_numpy(float), sub.hb.to_numpy()
        maes = []
        for rep in range(REP):
            p = np.zeros(len(hb))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, sub.pais_india * 2 + sub.anemia):
                p[te] = make_pipeline(StandardScaler(), RidgeCV(np.logspace(-2, 3, 20))).fit(X[tr], hb[tr]).predict(X[te])
            maes.append(np.abs(p - hb).mean())
        res[f"hb_mae|{nombre}"] = round(float(np.mean(maes)), 3)
        print(f"Hb MAE {nombre}: {np.mean(maes):.2f} g/dL", flush=True)

    # Externo Bangladesh: entrenar con % RGB en Eyes-Defy, probar allá
    if BANG.exists():
        b = pd.read_csv(BANG)
        b.columns = [c.strip() for c in b.columns]
        b["mujer"] = (b.Sex.str.strip().str.upper() == "F").astype(int)
        b["anemia"] = ((b.mujer == 1) & (b.Hb < 12) | (b.mujer == 0) & (b.Hb < 13)).astype(int)
        Xb = b[["%Red Pixel", "%Green pixel", "%Blue pixel"]].to_numpy(float)
        sub = r[r.pct_r.notna()]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"))
        m.fit(sub[["pct_r", "pct_g", "pct_b"]].to_numpy(), sub.anemia)
        auc_ext = roc_auc_score(b.anemia, m.predict_proba(Xb)[:, 1])
        auc_rojo = roc_auc_score(b.anemia, -b["%Red Pixel"])
        res["bangladesh"] = {"n": len(b), "anemia": int(b.anemia.sum()), "auc_modelo_eyesdefy": round(float(auc_ext), 3),
                             "auc_solo_pct_rojo": round(float(auc_rojo), 3)}
        print("Bangladesh externo:", res["bangladesh"])
    json.dump(res, open(OUT / "resultados.json", "w"), indent=1, ensure_ascii=False)


def extraer_app_crudo() -> None:
    """La tubería de la app (U-Net) pero SIN balance de blancos, + % RGB dentro de la máscara."""
    from src.deployment.inference_anemia import SegmentadorConjuntiva, letterbox, mayor_componente, medir_color

    r = pd.read_parquet(OUT / "rasgos.parquet")
    seg = SegmentadorConjuntiva(PROJECT_ROOT / "models/conjuntiva/best.keras")
    nuevas = []
    for i, x in r.iterrows():
        f = {"persona": x.persona}
        foto = leer(Path(x.foto)) if isinstance(x.foto, str) else None
        if foto is not None:
            img, util = letterbox(cv2.cvtColor(foto, cv2.COLOR_BGR2RGB))
            rgb = img.astype(np.float32) / 255
            conj = mayor_componente((seg(rgb) * util) >= seg.umbral)
            if conj.sum() >= 300:
                f.update({f"appc_{k}": v for k, v in medir_color(rgb, conj, np.ones(3, np.float32)).items()})
                sm = rgb[conj].sum(0)
                f.update(app_pct_r=100 * sm[0] / sm.sum(), app_pct_g=100 * sm[1] / sm.sum(), app_pct_b=100 * sm[2] / sm.sum())
        nuevas.append(f)
        if i % 40 == 0:
            print(i, flush=True)
    r = r.drop(columns=[c for c in r.columns if c.startswith(("appc_", "app_pct"))]).merge(pd.DataFrame(nuevas), on="persona")
    r.to_parquet(OUT / "rasgos.parquet")
    print("listo app crudo", r.appc_eritema.notna().sum())


def evaluar2() -> None:
    """Combinaciones finales; lo que importa es superar a sexo+edad DENTRO de cada país."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.deployment.inference_anemia import CARACTERISTICAS

    r = pd.read_parquet(OUT / "rasgos.parquet")
    r["pais_india"] = (r.pais == "India").astype(int)
    C = list(CARACTERISTICAS)
    conj = {
        "sexo+edad": ["mujer", "edad"],
        "mano_crudo": [f"cruda_{k}" for k in C],
        "mano_crudo+pct": [f"cruda_{k}" for k in C] + ["pct_r", "pct_g", "pct_b"],
        "mano_crudo+pct+sexo+edad": [f"cruda_{k}" for k in C] + ["pct_r", "pct_g", "pct_b", "mujer", "edad"],
        "app_balance(actual)": [f"app_{k}" for k in C],
        "app_crudo": [f"appc_{k}" for k in C],
        "app_crudo+pct": [f"appc_{k}" for k in C] + ["app_pct_r", "app_pct_g", "app_pct_b"],
        "app_crudo+pct+sexo+edad": [f"appc_{k}" for k in C] + ["app_pct_r", "app_pct_g", "app_pct_b", "mujer", "edad"],
    }
    modelos = {
        "logistica": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced")),
        "bosque": lambda: RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1),
    }
    res = {}
    for nombre, cols in conj.items():
        sub = r[r[cols].notna().all(1)].reset_index(drop=True)
        X, y = sub[cols].to_numpy(float), sub.anemia.to_numpy()
        ind = (sub.pais == "India").to_numpy()
        for mn, fab in modelos.items():
            ps, ai, at = [], [], []
            for rep in range(15):
                p = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, sub.pais_india * 2 + y):
                    # se entrena con ambos países pero el país NO es un rasgo
                    p[te] = fab().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                ps.append(p); ai.append(roc_auc_score(y[ind], p[ind])); at.append(roc_auc_score(y[~ind], p[~ind]))
            dentro = (np.mean(ai) * ind.sum() + np.mean(at) * (~ind).sum()) / len(y)
            res[f"{nombre}|{mn}"] = {"n": len(y), "india": round(float(np.mean(ai)), 3), "italia": round(float(np.mean(at)), 3),
                                     "dentro_pais": round(float(dentro), 3), "ic_india": [round(float(np.percentile(ai, 5)), 3), round(float(np.percentile(ai, 95)), 3)]}
            print(f"{nombre:28s} {mn:9s} India {np.mean(ai):.3f}  Italia {np.mean(at):.3f}  dentro {dentro:.3f}", flush=True)
    json.dump(res, open(OUT / "resultados2.json", "w"), indent=1, ensure_ascii=False)




def guardar() -> None:
    """Modelo final para producción: tubería de la app SIN balance de blancos + regresión logística."""
    import joblib
    import shutil
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.deployment.inference_anemia import CARACTERISTICAS

    r = pd.read_parquet(OUT / "rasgos.parquet")
    cols = [f"appc_{k}" for k in CARACTERISTICAS]
    r = r[r[cols].notna().all(1)].reset_index(drop=True)
    X, y = r[cols].to_numpy(float), r.anemia.to_numpy()
    ind = (r.pais == "India").to_numpy()
    fab = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"))  # noqa: E731
    oofs, dentro, ai, at = [], [], [], []
    for rep in range(20):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, ind * 2 + y):
            p[te] = fab().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oofs.append(p)
        a1, a2 = roc_auc_score(y[ind], p[ind]), roc_auc_score(y[~ind], p[~ind])
        ai.append(a1); at.append(a2); dentro.append((a1 * ind.sum() + a2 * (~ind).sum()) / len(y))
    p = np.mean(oofs, 0)
    # peso por (país, clase): cada país cuenta 50/50 → la prevalencia por país no ayuda
    w = np.ones(len(y))
    for g in (ind, ~ind):
        for c in (0, 1):
            m = g & (y == c)
            w[m] = 0.5 * g.sum() / max(m.sum(), 1)

    def sens_esp(bajo: float, alto: float) -> tuple[float, float, float]:
        pos, neg = p >= alto, p < bajo
        dado = pos | neg
        s = np.sum(w[pos & (y == 1)]) / np.sum(w[(y == 1) & dado])
        e = np.sum(w[neg & (y == 0)]) / np.sum(w[(y == 0) & dado])
        return float(s), float(e), float(dado.mean())

    umbral = 0.5
    s0, e0, _ = sens_esp(umbral, umbral)
    zona = (0.35, 0.65)
    s, e, cob = sens_esp(*zona)
    modelo = fab().fit(X, y)
    v = {
        "auc_cv": round(float(np.mean(dentro)), 4), "auc_sd": round(float(np.std(dentro)), 4),
        "auc_india": round(float(np.mean(ai)), 4), "auc_italia": round(float(np.mean(at)), 4),
        "umbral": umbral, "sin_zona_gris": {"sensibilidad": round(s0, 4), "especificidad": round(e0, 4), "exactitud": round((s0 + e0) / 2, 4)},
        "zona_gris": list(zona), "cobertura": round(cob, 4), "sensibilidad": round(s, 4), "especificidad": round(e, 4),
        "exactitud": round((s + e) / 2, 4), "modelo": "logistica", "n": int(len(y)), "n_anemia": int(y.sum()),
        "balance_blancos": False,
        "datos": "Eyes-Defy-Anemia original (Dimauro 2022) con Hb de laboratorio; anemia OMS por sexo",
        "metodo": "AUC dentro de cada país (India/Italia), 5×20 por persona; sens/esp ponderadas 50/50 por país",
        "nota": "Sin balance de blancos: en este dataset (luz estandarizada) corregir con la esclera empeoraba (0.74 → 0.82). "
                "Con luz ambiente no validado.",
    }
    dst = PROJECT_ROOT / "models/anemia"
    resp = PROJECT_ROOT / "models/anemia_respaldo_v2"
    if not resp.exists():
        shutil.copytree(dst, resp)
    joblib.dump({"modelo": modelo, "umbral": umbral, "zona_gris": list(zona), "caracteristicas": list(CARACTERISTICAS),
                 "balance_blancos": False, "validacion": v}, dst / "clasificador.joblib")
    (dst / "config.json").write_text(json.dumps(v, indent=2, ensure_ascii=False))
    print(json.dumps(v, indent=2, ensure_ascii=False))


GHANA_C = Path.home() / "TemrariosSpace/datasets/Anemia/ghana_conj_completo"


def ghana() -> None:
    """Validación EXTERNA con Ghana (Mendeley nt7r8hv2pz, CC BY 4.0: niños 6-59 meses, otra cámara,
    conjuntiva ya recortada con alfa, etiqueta anémico/no según Hb de laboratorio)."""
    import re
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.deployment.inference_anemia import CARACTERISTICAS, medir_color

    filas = []
    for f in sorted(GHANA_C.rglob("*.png")):
        if re.search(r" \(\d+\)", f.name):
            continue  # copias aumentadas (giros/espejos): mismo color
        x = leer(f)
        if x is None or x.ndim != 3 or x.shape[2] != 4:
            continue
        rgb = cv2.cvtColor(x[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        m = (x[..., 3] > 127) & (rgb.max(2) > 0.03)
        if m.sum() < 80:
            continue
        clase = 0 if f.name.lower().startswith("non") else 1
        num = re.search(r"(\d+)", f.stem).group(1).lstrip("0")[:3]
        pref = re.sub(r"[-_ ]?\d.*", "", f.stem).lower()
        fila = {"archivo": f.name, "persona": f"{pref}_{num}", "anemia": clase}
        fila.update({f"cruda_{k}": v for k, v in medir_color(rgb, m, np.ones(3, np.float32)).items()})
        filas.append(fila)
    g = pd.DataFrame(filas)
    print("Ghana:", len(g), "fotos,", g.persona.nunique(), "niños; anemia", g.groupby("persona").anemia.first().mean().round(2))
    cols = [f"cruda_{k}" for k in CARACTERISTICAS]
    fab = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"))  # noqa: E731
    r = pd.read_parquet(OUT / "rasgos.parquet")
    r = r[r[cols].notna().all(1)]
    res = {"ghana": {"fotos": len(g), "ninos": int(g.persona.nunique()), "anemia": int(g.groupby("persona").anemia.first().sum())}}
    # 1) externo: entrenar en Eyes-Defy (adultos, aditamento), probar en Ghana (niños, otra cámara)
    m = fab().fit(r[cols].to_numpy(float), r.anemia)
    p = m.predict_proba(g[cols].to_numpy(float))[:, 1]
    pp = pd.DataFrame({"p": p, "y": g.anemia, "g": g.persona}).groupby("g").mean()
    res["externo_eyesdefy_a_ghana"] = round(float(roc_auc_score(pp.y, pp.p)), 3)
    # 2) interno Ghana (agrupado por niño) y 3) entrenar con ambos
    X, y, grp = g[cols].to_numpy(float), g.anemia.to_numpy(), g.persona.to_numpy()
    ai, am = [], []
    for rep in range(10):
        pi, pm = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, grp):
            pi[te] = fab().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            Xa = np.vstack([X[tr], r[cols].to_numpy(float)]); ya = np.r_[y[tr], r.anemia.to_numpy()]
            pm[te] = fab().fit(Xa, ya).predict_proba(X[te])[:, 1]
        for lst, pr in ((ai, pi), (am, pm)):
            q = pd.DataFrame({"p": pr, "y": y, "g": grp}).groupby("g").mean()
            lst.append(roc_auc_score(q.y, q.p))
    res["interno_ghana"] = round(float(np.mean(ai)), 3)
    res["ghana_con_eyesdefy_en_entreno"] = round(float(np.mean(am)), 3)
    print(json.dumps(res, indent=1))
    g.to_parquet(OUT / "ghana.parquet")
    json.dump(res, open(OUT / "resultados_ghana.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    {"extraer": extraer, "evaluar": evaluar, "extraer_app_crudo": extraer_app_crudo, "evaluar2": evaluar2, "guardar": guardar, "ghana": ghana}[sys.argv[1]]()
