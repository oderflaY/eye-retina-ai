"""Pulso y calidad de señal PPG con la cámara del celular (dedo + flash) — BUT PPG; SpO2 exploratorio — MTHS.

Datos (en ~/TemrariosSpace/datasets/PPG):
  * BUT PPG 2.0.0 (PhysioNet, CC BY 4.0): 3 888 grabaciones de 10 s a 30 Hz (Xiaomi Mi9 / Huawei P20 Pro), 50 personas,
    pulso de referencia por ECG y calidad anotada (1 buena / 0 mala). Canal rojo (en las viejas viene invertido).
  * MTHS (Farvardin et al. 2022; CC BY-NC-ND 4.0): 62 personas, iPhone 5s, RGB 30 Hz + pulso y SpO2 de oxímetro a 1 Hz.
    Solo se usa para EVALUAR (pulso externo) y explorar SpO2; nada entrenado con MTHS se empaqueta en la app.

Qué hace:
  1. Evalúa el estimador ACTUAL de la app (copia fiel de EstimadorPpg.kt) contra el ECG.
  2. Estimador nuevo: pasa-banda 0.7–3.5 Hz + espectro + picos con verificación de armónicos.
  3. Detector de calidad (regresión logística con 10 rasgos calculables en el teléfono), validación agrupada por persona.
  4. MTHS: validación externa del pulso y SpO2 (ratio de ratios rojo/azul) por persona.
Salida: models/ppg/calidad.json (coeficientes para Kotlin), reports/ppg/resultados.json

Uso: .venv-rag/bin/python src/training/train_ppg.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sg
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
BUT = Path.home() / "TemrariosSpace/datasets/PPG/butppg/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0"
MTHS = Path.home() / "TemrariosSpace/datasets/PPG/mths/MTHS/Data"
FS = 30.0
SALIDA_MODELO = RAIZ / "models/ppg"
SALIDA_REPORTE = RAIZ / "reports/ppg"


# ── carga ────────────────────────────────────────────────────────────────────
def cargar_but() -> pd.DataFrame:
    import wfdb
    q = pd.read_csv(BUT / "quality-hr-ann.csv", encoding="utf-8-sig")
    s = pd.read_csv(BUT / "subject-info.csv", encoding="utf-8-sig")
    df = q.merge(s[["ID", "Ear/finger", "Motion"]], on="ID")
    rojos = []
    for rid in df.ID:
        r = wfdb.rdrecord(str(BUT / f"{rid}/{rid}_PPG"))
        x = r.p_signal
        rojo = -x.reshape(-1) if x.shape[0] == 1 else x[:, 0]  # viejas: 1×300 rojo invertido; nuevas: 300×RGB
        rojos.append(rojo.astype(float))
    df["rojo"] = rojos
    df["persona"] = df.ID // 1000
    df["dedo"] = df["Ear/finger"] == 1
    return df


# ── estimador actual de la app (EstimadorPpg.kt, sin cambios) ─────────────────
def estimador_app(x: np.ndarray, fs: float = FS):
    n = len(x)
    if n < 60:
        return None
    k = max(int(fs), 3)
    sin_tend = np.array([x[i] - x[max(0, i - k // 2):min(n - 1, i + k // 2) + 1].mean() for i in range(n)])
    s = np.array([sin_tend[max(0, i - 1):min(n, i + 2)].mean() for i in range(n)])
    y = -s
    amp = y.max() - y.min()
    if amp <= 0:
        return None
    min_dist = max(int(fs * 0.33), 2)
    picos: list[int] = []
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > amp * 0.15:
            if not picos or i - picos[-1] >= min_dist:
                picos.append(i)
            elif y[i] > y[picos[-1]]:
                picos[-1] = i
    if len(picos) < 4:
        return None
    iv = [(b - a) * 1000 / fs for a, b in zip(picos, picos[1:])]
    iv = [v for v in iv if 330 <= v <= 1500]
    if len(iv) < 3:
        return None
    med = sorted(iv)[len(iv) // 2]
    return 60000 / med


# ── estimador nuevo + rasgos de calidad ──────────────────────────────────────
# Butterworth 2.º orden 0.7–3.5 Hz a 30 Hz, aplicado ida y vuelta (mismos coeficientes en Kotlin)
B, A = sg.butter(2, [0.7, 3.5], btype="band", fs=FS)


def filtrar(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    return sg.filtfilt(B, A, x, padlen=min(3 * max(len(A), len(B)), len(x) - 1))


def espectro(y: np.ndarray, fs: float = FS, nfft: int = 1024):
    w = np.hanning(len(y))
    p = np.abs(np.fft.rfft(y * w, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    return f, p


def picos_de(y: np.ndarray, fs: float, hr_guia: float) -> list[int]:
    """Máximos locales con prominencia ≥ 0.3·desv. estándar, separados ≥ 0.6 periodos de la guía espectral.
    Misma lógica que find_peaks(distance, prominence) de scipy, escrita a mano para copiarla igual en Kotlin."""
    n = len(y)
    cand = [i for i in range(1, n - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1]]
    umbral = 0.3 * float(np.std(y))
    def prominencia(i: int) -> float:
        izq, j = y[i], i - 1
        while j >= 0 and y[j] <= y[i]:
            izq = min(izq, y[j]); j -= 1
        der, j = y[i], i + 1
        while j < n and y[j] <= y[i]:
            der = min(der, y[j]); j += 1
        return y[i] - max(izq, der)
    cand = [i for i in cand if prominencia(i) >= umbral]
    dist = max(2, int(fs * 60 / hr_guia * 0.6))
    elegidos: list[int] = []
    for i in sorted(cand, key=lambda k: -y[k]):  # el más alto gana dentro de la distancia mínima
        if all(abs(i - j) >= dist for j in elegidos):
            elegidos.append(i)
    return sorted(elegidos)


def analizar(x: np.ndarray, fs: float = FS) -> dict:
    """Pulso + 10 rasgos de calidad de una ventana de señal roja cruda."""
    y = -filtrar(x)  # la sangre absorbe: el latido es un valle del rojo
    f, p = espectro(y, fs)
    banda = (f >= 0.7) & (f <= 3.5)
    fb, pb = f[banda], p[banda]
    i0 = int(np.argmax(pb))
    f0 = fb[i0]
    hr_fft = f0 * 60
    pk = picos_de(y, fs, hr_fft)
    iv = np.diff(pk) / fs * 1000
    iv = iv[(iv >= 330) & (iv <= 1500)]
    hr_picos = 60000 / np.median(iv) if len(iv) >= 2 else np.nan
    # concentración de energía en la fundamental y su armónico
    cerca = (np.abs(fb - f0) <= 0.15) | (np.abs(fb - 2 * f0) <= 0.15)
    snr = pb[cerca].sum() / max(pb.sum(), 1e-12)
    # autocorrelación al periodo
    yc = (y - y.mean()) / (y.std() + 1e-9)
    lag = int(round(fs * 60 / hr_fft))
    ac = float(np.mean(yc[:-lag] * yc[lag:])) if 0 < lag < len(yc) - 5 else 0.0
    dc = float(np.mean(x))
    # parecido entre latidos (correlación con la plantilla promedio)
    L = int(fs * 60 / hr_fft)
    segs = [y[q - L // 2:q + L // 2] for q in pk if q - L // 2 >= 0 and q + L // 2 <= len(y)]
    if len(segs) >= 3:
        S = np.array(segs)
        tpl = S.mean(0)
        plantilla = float(np.mean([np.corrcoef(z, tpl)[0, 1] for z in S]))
    else:
        plantilla = 0.0
    # entropía espectral en la banda y energía fuera de banda (movimiento, deriva) sobre la señal sin filtrar
    fx, px = espectro(x - x.mean(), fs)
    bx = (fx >= 0.7) & (fx <= 3.5)
    pn = px[bx] / max(px[bx].sum(), 1e-12)
    entropia = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
    fuera = px[((fx > 0.1) & (fx < 0.7)) | (fx > 3.5)].sum() / max(px[bx].sum(), 1e-12)
    rasgos = {
        "snr": float(snr),
        "autocorr": ac,
        "acuerdo": float(abs(hr_fft - hr_picos) / hr_fft) if np.isfinite(hr_picos) else 1.0,
        "cv_intervalos": float(iv.std() / iv.mean()) if len(iv) >= 3 else 1.0,
        "perfusion": float(np.log10(max((np.percentile(y, 95) - np.percentile(y, 5)) / max(abs(dc), 1e-6) * 100, 1e-4))),
        "asimetria": float(skew(y)),
        "curtosis": float(kurtosis(y)),
        "deriva": float(np.log10(max(abs(np.polyfit(np.arange(len(x)) / fs, x, 1)[0]) / max(abs(dc), 1e-6) * 100, 1e-5))),
        "rojo_dc": float(dc / 255.0),
        "n_picos_ok": float(len(iv) / max(len(x) / fs * hr_fft / 60, 1)),
        "plantilla": plantilla,
        "entropia": entropia,
        "fuera_banda": float(np.log10(fuera + 1e-6)),
    }
    rasgos = {k: (float(np.nan_to_num(v, nan=0.0, posinf=10.0, neginf=-10.0))) for k, v in rasgos.items()}
    hr = hr_picos if np.isfinite(hr_picos) else hr_fft
    return {"hr": float(hr), "hr_fft": float(hr_fft), **rasgos}


RASGOS = ["snr", "autocorr", "acuerdo", "cv_intervalos", "perfusion", "asimetria", "curtosis", "deriva", "rojo_dc", "n_picos_ok",
          "plantilla", "entropia", "fuera_banda"]


def metricas_hr(pred: np.ndarray, ref: np.ndarray) -> dict:
    ok = np.isfinite(pred)
    e = np.abs(pred[ok] - ref[ok])
    return {"n": int(len(ref)), "con_lectura": round(float(ok.mean()), 3), "mae": round(float(e.mean()), 2) if len(e) else None,
            "dentro_5lpm": round(float((e <= 5).mean()), 3) if len(e) else None,
            "dentro_10lpm": round(float((e <= 10).mean()), 3) if len(e) else None}


# ── MTHS ─────────────────────────────────────────────────────────────────────
def ventanas_mths(seg: int = 10, paso: int = 5) -> pd.DataFrame:
    filas = []
    for sf in sorted(MTHS.glob("signal_*.npy")):
        pid = int(sf.stem.split("_")[1])
        s = np.load(sf)
        lab = np.load(MTHS / f"label_{pid}.npy")
        if s.shape[0] == 3 and s.shape[1] != 3:
            s = s.T
        for t0 in range(0, int(len(s) / FS) - seg, paso):
            a, b = int(t0 * FS), int((t0 + seg) * FS)
            l = lab[t0:t0 + seg]
            if len(l) < seg:
                break
            r, g, bl = s[a:b, 0], s[a:b, 1], s[a:b, 2]
            # AC/DC de cada canal (para el ratio de ratios del SpO2)
            acdc = []
            for c in (r, g, bl):
                yc = filtrar(c)
                acdc.append((np.percentile(yc, 95) - np.percentile(yc, 5)) / max(c.mean(), 1e-6))
            fila = {"persona": pid, "hr_ref": float(np.nanmedian(l[:, 0])), "spo2_ref": float(np.nanmedian(l[:, 1])),
                    "ac_r": acdc[0], "ac_g": acdc[1], "ac_b": acdc[2], "dc_r": r.mean(), "dc_g": g.mean(), "dc_b": bl.mean()}
            fila.update({f"q_{k}": v for k, v in analizar(r).items()})
            fila["hr_app"] = estimador_app(r) or np.nan
            filas.append(fila)
    return pd.DataFrame(filas)


def main() -> None:
    SALIDA_MODELO.mkdir(parents=True, exist_ok=True)
    SALIDA_REPORTE.mkdir(parents=True, exist_ok=True)
    df = cargar_but()
    print(f"BUT PPG: {len(df)} grabaciones, {df.persona.nunique()} personas, buena calidad {int(df.Quality.sum())}")

    df["hr_app"] = [estimador_app(x) or np.nan for x in df.rojo]
    an = pd.DataFrame([analizar(x) for x in df.rojo])
    df = pd.concat([df, an], axis=1)

    res: dict = {"datos": {"grabaciones": len(df), "personas": int(df.persona.nunique()), "buenas": int(df.Quality.sum()),
                           "dedo": int(df.dedo.sum())}}
    # 1-2. pulso: app actual vs nuevo (en grabaciones de buena calidad, dedo / todas)
    res["pulso"] = {}
    for nombre, m in {"buenas_dedo": (df.Quality == 1) & df.dedo, "buenas_todas": df.Quality == 1, "todas_dedo": df.dedo}.items():
        d = df[m]
        res["pulso"][nombre] = {"app_actual": metricas_hr(d.hr_app.to_numpy(float), d.HR.to_numpy(float)),
                                "nuevo": metricas_hr(d.hr.to_numpy(float), d.HR.to_numpy(float))}
    print(json.dumps(res["pulso"], indent=1))

    # 3. calidad: regresión logística, validación agrupada por persona (5 pliegues)
    X = df[RASGOS].to_numpy(float)
    # objetivo = el pulso que saca el estimador quedó a ±5 lpm del ECG (lo que le importa a la persona)
    yq = ((df.hr - df.HR).abs() <= 5).to_numpy(int)
    g = df.persona.to_numpy()
    prob = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(X, yq, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        m.fit(X[tr], yq[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]
    df["p_calidad"] = prob
    auc = roc_auc_score(yq, prob)
    auc_dedo = roc_auc_score(yq[df.dedo], prob[df.dedo])
    auc_etiqueta_but = roc_auc_score(df.Quality, prob)
    # umbral: el que maximiza la precisión del pulso aceptado manteniendo ≥ 70 % de las buenas
    umbrales = {}
    # umbral = el que deja pasar el mejor 30 % de las ventanas del dedo (la app sigue midiendo hasta que una pase)
    u30 = float(np.quantile(prob[df.dedo], 0.7))
    for u in [0.3, 0.4, 0.5, 0.6, round(u30, 3)]:
        acep = df[df.dedo & (df.p_calidad >= u)]
        rech = df[df.dedo & (df.p_calidad < u)]
        umbrales[str(u)] = {"aceptadas": len(acep), "sens_buenas": round(float((acep.Quality == 1).sum() / max((df.dedo & (df.Quality == 1)).sum(), 1)), 3),
                            "pulso_aceptadas": metricas_hr(acep.hr.to_numpy(float), acep.HR.to_numpy(float)),
                            "pulso_rechazadas": metricas_hr(rech.hr.to_numpy(float), rech.HR.to_numpy(float))}
    res["calidad"] = {"objetivo": "pulso a ±5 lpm del ECG", "auc_todas": round(float(auc), 3), "auc_dedo": round(float(auc_dedo), 3),
                      "auc_vs_etiqueta_humana": round(float(auc_etiqueta_but), 3), "umbral_app": round(u30, 3), "umbrales_dedo": umbrales}
    print(json.dumps(res["calidad"], indent=1))

    # modelo final con todo BUT PPG → coeficientes para Kotlin
    final = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")).fit(X, yq)
    sc, lr = final.named_steps["standardscaler"], final.named_steps["logisticregression"]
    modelo = {"rasgos": RASGOS, "media": sc.mean_.round(6).tolist(), "escala": sc.scale_.round(6).tolist(),
              "coef": lr.coef_[0].round(6).tolist(), "intercepto": round(float(lr.intercept_[0]), 6),
              "umbral": round(u30, 4), "filtro_b": B.tolist(), "filtro_a": A.tolist(),
              "filtro_zi": sg.lfilter_zi(B, A).tolist(), "padlen": 15, "fs": FS, "nfft": 1024,
              "auc_cv_persona": round(float(auc), 3), "fuente": "BUT PPG 2.0.0 (PhysioNet, CC BY 4.0)"}
    json.dump(modelo, open(SALIDA_MODELO / "calidad.json", "w"), indent=1)

    # 4. MTHS: externo (otro teléfono, otras personas)
    if MTHS.exists():
        mt = ventanas_mths()
        mt["p_calidad"] = final.predict_proba(mt[[f"q_{k}" for k in RASGOS]].to_numpy(float))[:, 1]
        acep = mt[mt.p_calidad >= u30]
        res["mths_pulso"] = {"ventanas": len(mt), "personas": int(mt.persona.nunique()),
                             "app_actual": metricas_hr(mt.hr_app.to_numpy(float), mt.hr_ref.to_numpy(float)),
                             "nuevo": metricas_hr(mt.q_hr.to_numpy(float), mt.hr_ref.to_numpy(float)),
                             "nuevo_calidad_ok": metricas_hr(acep.q_hr.to_numpy(float), acep.hr_ref.to_numpy(float)),
                             "fraccion_aceptada": round(float(len(acep) / len(mt)), 3)}
        # SpO2 por ratio de ratios, por persona (GroupKFold); baseline = media del entrenamiento
        Xs = np.column_stack([mt.ac_r / mt.ac_b, mt.ac_r / mt.ac_g, mt.dc_r / mt.dc_b, mt.dc_g / mt.dc_r])
        ys = mt.spo2_ref.to_numpy(float)
        ok = np.isfinite(Xs).all(1) & np.isfinite(ys)
        Xs, ys, gs = Xs[ok], ys[ok], mt.persona.to_numpy()[ok]
        pred, base = np.zeros(len(ys)), np.zeros(len(ys))
        for tr, te in GroupKFold(5).split(Xs, ys, gs):
            pred[te] = LinearRegression().fit(Xs[tr], ys[tr]).predict(Xs[te])
            base[te] = ys[tr].mean()
        # por persona (promedio de sus ventanas)
        pp = pd.DataFrame({"g": gs, "y": ys, "p": pred, "b": base}).groupby("g").mean()
        res["mths_spo2"] = {"rango_spo2": [float(ys.min()), float(ys.max())], "sd_spo2": round(float(ys.std()), 2),
                            "mae_modelo_ventana": round(float(np.abs(pred - ys).mean()), 2),
                            "mae_media_ventana": round(float(np.abs(base - ys).mean()), 2),
                            "mae_modelo_persona": round(float(np.abs(pp.p - pp.y).mean()), 2),
                            "mae_media_persona": round(float(np.abs(pp.b - pp.y).mean()), 2),
                            "corr_persona": round(float(np.corrcoef(pp.p, pp.y)[0, 1]), 3),
                            "personas_bajo_94": int((pp.y < 94).sum())}
        print(json.dumps({k: res[k] for k in ("mths_pulso", "mths_spo2")}, indent=1))
    json.dump(res, open(SALIDA_REPORTE / "resultados.json", "w"), indent=1, ensure_ascii=False)
    # 8 señales reales de BUT PPG (CC BY 4.0) con lo que debe calcular Kotlin
    ej = df[df.dedo].groupby("Quality").head(4)
    vect = []
    for _, r in ej.iterrows():
        a = analizar(r.rojo)
        z = (np.array([a[k] for k in RASGOS]) - sc.mean_) / sc.scale_
        vect.append({"id": int(r.ID), "rojo": np.round(r.rojo, 4).tolist(), "hr": a["hr"], "hr_fft": a["hr_fft"],
                     "rasgos": {k: a[k] for k in RASGOS},
                     "p": float(1 / (1 + np.exp(-(z @ lr.coef_[0] + lr.intercept_[0])))), "hr_ecg": int(r.HR)})
    json.dump(vect, open(SALIDA_MODELO / "vectores_prueba.json", "w"))


if __name__ == "__main__":
    main()
