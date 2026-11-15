"""Color de referencia de encías sanas (y prueba del medidor de color) con el dataset de gingivitis.

Datos: Dental IntraOral Gingivitis (Mendeley 3253gj88rr, CC BY 4.0), reducido en el VPS →
~/TemrariosSpace/datasets/Encias/reducido/gingivitis (fotos + etiquetas.csv).
Cada diente trae una caja y un ``Class_ID`` = grado de gingivitis de su encía + 2 (deducido de los
textos de los periodoncistas; se verifica abajo). Grado 0 = encía sana.

No hay fotos con TODA la encía sana (grado 0 solo en 309 dientes sueltos), así que se mide por
diente: la franja de encía en el cuello de cada diente (arriba en el maxilar, abajo en la
mandíbula) − dientes (models/dientes) − sombra, con los dientes de la foto como blanco. Resultado: models/encia/referencia.json.

    python -m src.data.referencia_encias
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from src.deployment.inference_anemia import medir_color
from src.deployment.inference_encia import SegmentadorDientes, blanco_dientes, mascara_encia

DATOS = Path.home() / "TemrariosSpace/datasets/Encias/reducido/gingivitis"
SALIDA = Path("models/encia")


def caja(c, W, H):
    cx, cy, w, h = c
    return int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)


def verificar_escala(filas) -> dict:
    """Compara Class_ID con los grados que escriben los periodoncistas en el texto."""
    pares = Counter()
    for f in filas[:300]:
        et = json.loads(f["labels"])
        grados_texto = {int(g) for g in re.findall(r"gingivitis (\d)", f["caption"])}
        clases = {t["Class_ID"] for t in et.get("Teeth", [])}
        if len(grados_texto) == 1 and len(clases) == 1:
            pares[(clases.pop(), grados_texto.pop())] += 1
    return {f"clase {c} ↔ grado {g}": n for (c, g), n in pares.most_common(10)}


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    filas = list(csv.DictReader(open(DATOS / "etiquetas.csv")))
    escala = verificar_escala(filas)
    print("escala (clase ↔ grado del texto):", escala)
    seg = SegmentadorDientes(Path("models/dientes/best.keras"))

    medidas = []
    for i, f in enumerate(filas):
        et = json.loads(f["labels"])
        dientes = et.get("Teeth", [])
        if not dientes:
            continue
        bgr = cv2.imread(str(DATOS / f["archivo"]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W = rgb.shape[:2]
        mask_dientes = seg(rgb)
        blanco, n_blanco = blanco_dientes(rgb, mask_dientes)
        maxilar = et.get("Maxilla_BB_Coords")
        for t in dientes:
            x0, y0, x1, y1 = caja(t["Bounding_Box_Coordinates"], W, H)
            h = max(4, y1 - y0)
            arriba = maxilar is not None and abs(t["Bounding_Box_Coordinates"][1] - maxilar[1]) < maxilar[3] / 2
            # franja de encía en el cuello del diente: hacia arriba en el maxilar, hacia abajo en la mandíbula
            ya, yb = (y0 - int(0.45 * h), y0 + int(0.25 * h)) if arriba else (y1 - int(0.25 * h), y1 + int(0.45 * h))
            region = np.zeros((H, W), bool)
            region[max(0, ya):min(H, yb), max(0, x0):min(W, x1)] = True
            encia = mascara_encia(rgb, region, mask_dientes)
            if encia.sum() < 60:
                continue
            c = medir_color(rgb, encia, blanco)
            medidas.append({"archivo": f["archivo"], "grado": t["Class_ID"] - 2, "arriba": bool(arriba),
                            "eritema": c["eritema"], "a_media": c["a_media"], "px": int(encia.sum()),
                            "corregido": n_blanco >= 200})
        if i % 100 == 0:
            print(i, len(medidas), flush=True)

    # La app mide la encía de ARRIBA: la referencia usa solo franjas del maxilar.
    medidas = [x for x in medidas if x["arriba"]]
    g = np.array([x["grado"] for x in medidas])
    eri = np.array([x["eritema"] for x in medidas])
    sanas = eri[g == 0]
    inflamadas = eri[g >= 3]
    sel = (g == 0) | (g >= 3)
    auc = float(roc_auc_score((g[sel] >= 3).astype(int), eri[sel]))
    corr = float(np.corrcoef(g, eri)[0, 1])
    por_grado = {int(k): {"n": int((g == k).sum()), "eritema_mediana": round(float(np.median(eri[g == k])), 4)}
                 for k in sorted(set(g.tolist()))}
    ref = {
        "sanas_eritema": np.round(np.sort(sanas), 5).tolist(),
        "umbral_inflamacion": float(np.percentile(sanas, 95)),
        "pct_palidez": 0.10,
        "pct_gris": 0.25,
        "n_franjas": len(medidas), "n_sanas": int(len(sanas)), "n_inflamadas_g3_4": int(len(inflamadas)),
        "por_grado": por_grado,
        "validacion_medidor": {
            "auc_sana_vs_inflamada_g3_por_eritema": round(auc, 3),
            "correlacion_grado_eritema": round(corr, 3),
            "nota": "Prueba que el medidor ve el enrojecimiento. NO valida la anemia (no hay hemoglobina).",
        },
        "escala_verificada": escala,
        "datos": "Mendeley 3253gj88rr (CC BY 4.0) + dientes: Dentalai (MIT)",
    }
    (SALIDA / "referencia.json").write_text(json.dumps(ref, indent=2, ensure_ascii=False))
    (SALIDA / "medidas.json").write_text(json.dumps(medidas, indent=1))
    print(json.dumps({k: v for k, v in ref.items() if k != "sanas_eritema"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
