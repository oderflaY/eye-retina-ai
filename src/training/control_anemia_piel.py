"""Controles de confusión del experimento de anemia por piel (ver docs/ROSTRO.md).

Corre: .venv/bin/python -c "import sys;sys.path.insert(0,'.');exec(open('src/training/control_anemia_piel.py').read())"

Resultado (2026-11-14): piel sin efecto de la foto AUC 0.54 (azar); conjuntiva sin efecto 0.74.
"""
import cv2, numpy as np, pandas as pd
from pathlib import Path
import src.training.experimento_anemia_piel as E
from src.deployment.inference_anemia import SegmentadorConjuntiva, extraer, letterbox, CARACTERISTICAS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.config import PROJECT_ROOT
seg = SegmentadorConjuntiva(PROJECT_ROOT / "models/conjuntiva/best.keras")
cache = []
for p in sorted(E.DATOS.rglob("*")):
    if p.suffix.lower() not in (".jpg",".png",".jpeg") or p.parent.name not in ("Anemia","NoAnemia"): continue
    rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB); e = extraer(seg, rgb)
    if not e["valida"]: continue
    _, util = letterbox(rgb)
    u8 = e["imagen"] if e["imagen"].dtype == np.uint8 else (e["imagen"]*255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    # "ojo" = conjuntiva + blanco del ojo (claro y poco saturado) + iris oscuro cercano → casco convexo
    blanco = (hsv[...,2] > 150) & (hsv[...,1] < 70) & util.astype(bool)
    ojo = (e["mascara"].astype(bool) | blanco).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ojo)
    grandes = np.isin(lab, [i for i in range(1, n) if st[i, 4] > 400])
    pts = np.argwhere(grandes)[:, ::-1].astype(np.int32)
    casco = np.zeros_like(ojo)
    if len(pts) > 3: cv2.fillConvexPoly(casco, cv2.convexHull(pts), 1)
    cache.append((p.parent.name == "Anemia", u8, e, util, casco))
print("fotos", len(cache))
logi = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"))
for modo, d in [("conj", 25), ("conj", 45), ("conj", 70), ("conj", 100), ("casco", 15), ("casco", 35), ("casco", 60)]:
    filas = []
    for an, u8, e, util, casco in cache:
        base = e["mascara"] if modo == "conj" else casco
        lejos = cv2.dilate(base.astype(np.uint8), np.ones((d, d), np.uint8)) == 0
        piel = E.mascara_piel(u8, np.zeros_like(e["mascara"]), util) & lejos
        if piel.sum() < E.PIEL_MIN: continue
        f = E.medir_piel(u8, piel, np.array(e["blanco"])); f["y"] = int(an); f["L"] = f["piel_L"]
        f.update({k: e["medidas"][k] for k in CARACTERISTICAS})
        filas.append(f)
    df = pd.DataFrame(filas); y = df.y.to_numpy()
    a_piel = E.cv_auc(logi, df[E.PIEL].to_numpy(), y, rep=10)[0]
    a_conj = E.cv_auc(logi, df[list(CARACTERISTICAS)].to_numpy(), y, rep=10)[0]
    a_ambas = E.cv_auc(logi, df[list(CARACTERISTICAS)+E.PIEL].to_numpy(), y, rep=10)[0]
    print(f"excluir {modo:5s} +{d:3d}px: n={len(df):3d} ({y.sum()} anemia) | piel {a_piel:.3f} | conj {a_conj:.3f} | conj+piel {a_ambas:.3f}")

print("---- controles ----")
from sklearn.metrics import roc_auc_score
filas=[]
for an, u8, e, util, casco in cache:
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV); h = u8.shape[0]
    aro = (hsv[...,1] < 40) & (hsv[...,2] > 170) & util.astype(bool); aro[: int(h*0.75)] = False
    lejos = cv2.dilate(casco, np.ones((35,35),np.uint8)) == 0
    piel = E.mascara_piel(u8, np.zeros_like(e["mascara"]), util) & lejos
    f = {"y": int(an), "n_aro": int(aro.sum())}
    if aro.sum() > 200:
        px = u8[aro].astype(np.float32)/255; lab = cv2.cvtColor(px[None], cv2.COLOR_RGB2Lab)[0]
        f.update(aro_L=np.median(lab[:,0]), aro_a=np.median(lab[:,1]), aro_b=np.median(lab[:,2]))
    b = np.array(e["blanco"], np.float32); lb = cv2.cvtColor(b[None,None], cv2.COLOR_RGB2Lab)[0,0]
    f.update(blanco_L=lb[0], blanco_a=lb[1], blanco_b=lb[2], brillo=float(hsv[...,2][util.astype(bool)].mean()))
    if piel.sum() >= E.PIEL_MIN: f.update(E.medir_piel(u8, piel, b))
    filas.append(f)
d = pd.DataFrame(filas)
def auc_set(cols, nombre):
    s = d.dropna(subset=cols); y = s.y.to_numpy()
    print(f"{nombre:38s} n={len(s):3d} AUC {E.cv_auc(logi, s[cols].to_numpy(), y, rep=10)[0]:.3f}")
auc_set(["aro_L","aro_a","aro_b"], "aro de plástico del aparato (NO es persona)")
auc_set(["blanco_L","blanco_a","blanco_b"], "blanco del ojo (referencia de luz)")
auc_set(["brillo"], "brillo general de la foto")
auc_set(["piel_L"], "tono de piel (claro/oscuro)")
auc_set(E.PIEL, "piel lejos del ojo (todas)")
# piel ajustada por condiciones de foto: ¿agrega algo sobre el aro + blanco + brillo?
ctrl = ["aro_L","aro_a","aro_b","blanco_L","blanco_a","blanco_b","brillo"]
auc_set(ctrl, "solo condiciones de foto (aro+blanco+brillo)")
auc_set(ctrl + E.PIEL, "condiciones + piel")
# por tono: dentro de tercios de tono de piel
s = d.dropna(subset=E.PIEL).copy(); s["tercio"] = pd.qcut(s.piel_L, 3, labels=False)
for t in range(3):
    q = s[s.tercio == t]
    if q.y.nunique() == 2: print(f"tercio de tono {t}: n={len(q)} anemia={q.y.mean():.0%}  AUC(piel_a sola)={max(roc_auc_score(q.y,q.piel_a),1-roc_auc_score(q.y,q.piel_a)):.3f}")

print("---- ¿la conjuntiva agrega algo real? ----")
for i,(an,u8,e,util,casco) in enumerate(cache):
    for k in CARACTERISTICAS: d.loc[i,k]=e["medidas"][k]
C=list(CARACTERISTICAS)
auc_set(C, "conjuntiva sola (mismas 152)") if False else None
s=d.dropna(subset=ctrl); y=s.y.to_numpy()
for cols,n in ((ctrl,"condiciones de foto"),(C,"conjuntiva"),(ctrl+C,"condiciones + conjuntiva"),(ctrl+C+E.PIEL,"condiciones + conj + piel")):
    q=s.dropna(subset=cols); print(f"{n:30s} n={len(q)} AUC {E.cv_auc(logi,q[cols].to_numpy(),q.y.to_numpy(),rep=10)[0]:.3f}")
# residualizar: quitar a la conjuntiva lo que explican las condiciones (regresión lineal), y probar el resto
from sklearn.linear_model import LinearRegression
X=s[ctrl].to_numpy(); R=np.column_stack([s[c].to_numpy()-LinearRegression().fit(X,s[c].to_numpy()).predict(X) for c in C])
print(f"{'conjuntiva SIN efecto de la foto':30s} n={len(s)} AUC {E.cv_auc(logi,R,y,rep=10)[0]:.3f}")
s2=s.dropna(subset=E.PIEL); X2=s2[ctrl].to_numpy()
R2=np.column_stack([s2[c].to_numpy()-LinearRegression().fit(X2,s2[c].to_numpy()).predict(X2) for c in E.PIEL])
print(f"{'piel SIN efecto de la foto':30s} n={len(s2)} AUC {E.cv_auc(logi,R2,s2.y.to_numpy(),rep=10)[0]:.3f}")
