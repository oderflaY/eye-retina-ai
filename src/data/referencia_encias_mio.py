"""Reentrena la referencia de encía sana sumando MIO (My Intraoral Images) y valida el medidor.

Datos: MIO, Zenodo 22047269 (CC BY 4.0) → ~/TemrariosSpace/datasets/Anemia/encia/mio_extraido/
{SANO, GINGIVITIS}: fotos frontales de dentista con separadores, dientes juntos.

Por foto se mide la encía de ARRIBA como en la app: franja justo encima del borde superior de los
dientes superiores (U-Net models/dientes) − dientes − sombra/brillo, con los dientes como blanco.

Salida:
- models/encia/referencia.json: ``sanas_eritema`` = franjas sanas de Mendeley (261) + fotos sanas de
  MIO; se guarda copia de la versión anterior en referencia_v1.json.
- models/encia/medidas_mio.json y reports/encia/mio.json (AUC sana vs gingivitis por foto).

NO hay hemoglobina en estos datos: esto mejora "cómo se ve una encía sana", no valida anemia.

    python -m src.data.referencia_encias_mio
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from src.deployment.inference_anemia import medir_color
from src.deployment.inference_encia import SegmentadorDientes, blanco_dientes, mascara_encia

DATOS = Path.home() / "TemrariosSpace/datasets/Anemia/encia/mio_extraido"
SALIDA = Path("models/encia")
REPORTE = Path("reports/encia")
ANCHO = 1024
EXT = {".jpg", ".jpeg", ".png"}


def franja_superior(dientes: np.ndarray, alto_frac: float = 0.09) -> np.ndarray:
    """Franja encima del primer diente de cada columna (solo columnas con diente en la mitad de arriba
    de la foto: así no se toma la encía de abajo cuando faltan dientes superiores)."""
    h, w = dientes.shape
    cols = dientes.any(0)
    tope = np.where(cols, dientes.argmax(0), -1)
    cols &= (tope > 0) & (tope < 0.6 * h)
    alto = int(alto_frac * h)
    filas = np.arange(h)[:, None]
    return cols[None, :] & (filas < tope[None, :] - 2) & (filas >= tope[None, :] - alto)


def medir(ruta: Path, seg: SegmentadorDientes) -> dict | None:
    bgr = cv2.imread(str(ruta))
    if bgr is None:
        return None
    H, W = bgr.shape[:2]
    bgr = cv2.resize(bgr, (ANCHO, int(H * ANCHO / W)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    dientes = seg(rgb)
    # quitar ruido: componente(s) grandes
    n, lab, st, _ = cv2.connectedComponentsWithStats(dientes.astype(np.uint8))
    grandes = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 0.002 * dientes.size]
    dientes = np.isin(lab, grandes)
    if dientes.sum() < 500:
        return {"archivo": ruta.name, "valida": False, "motivo": "sin_dientes"}
    region = franja_superior(dientes)
    encia = mascara_encia(rgb, region, dientes)
    blanco, n_blanco = blanco_dientes(rgb, dientes)
    if encia.sum() < 300:
        return {"archivo": ruta.name, "valida": False, "motivo": "sin_encia", "px": int(encia.sum())}
    c = medir_color(rgb, encia, blanco)
    return {"archivo": ruta.name, "valida": True, "eritema": c["eritema"], "a_media": c["a_media"],
            "px": int(encia.sum()), "corregido": n_blanco >= 200}


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    REPORTE.mkdir(parents=True, exist_ok=True)
    seg = SegmentadorDientes(Path("models/dientes/best.keras"))
    medidas = []
    for clase in ("SANO", "GINGIVITIS"):
        fs = sorted(p for p in (DATOS / clase).iterdir() if p.suffix.lower() in EXT)
        for i, f in enumerate(fs):
            m = medir(f, seg)
            if m:
                m["clase"] = clase
                medidas.append(m)
            if i % 50 == 0:
                print(clase, i, len(fs), flush=True)
    (SALIDA / "medidas_mio.json").write_text(json.dumps(medidas, indent=1))

    ok = [m for m in medidas if m["valida"]]
    y = np.array([m["clase"] == "GINGIVITIS" for m in ok], int)
    eri = np.array([m["eritema"] for m in ok])
    a = np.array([m["a_media"] for m in ok])
    sanas_mio = eri[y == 0]

    ref_path = SALIDA / "referencia.json"
    ref = json.loads(ref_path.read_text())
    if not (SALIDA / "referencia_v1.json").exists():
        shutil.copy(ref_path, SALIDA / "referencia_v1.json")
        sanas_mendeley = np.array(ref["sanas_eritema"])
    else:
        sanas_mendeley = np.array(json.loads((SALIDA / "referencia_v1.json").read_text())["sanas_eritema"])

    # Percentil de la gingivitis MIO con la referencia vieja y la nueva (cuánto "inflamación" detecta)
    def pct(x, sanas):
        return np.array([(sanas < v).mean() for v in x])

    sanas_nueva = np.sort(np.concatenate([sanas_mendeley, sanas_mio]))
    umbral_nuevo = float(np.percentile(sanas_nueva, 95))
    reporte = {
        "fotos": {"sano": int((np.array([m["clase"] for m in medidas]) == "SANO").sum()),
                  "gingivitis": int((np.array([m["clase"] for m in medidas]) == "GINGIVITIS").sum())},
        "validas": {"sano": int((y == 0).sum()), "gingivitis": int((y == 1).sum())},
        "descartadas": {k: sum(1 for m in medidas if not m["valida"] and m["motivo"] == k)
                        for k in ("sin_dientes", "sin_encia")},
        "auc_sana_vs_gingivitis_eritema": round(float(roc_auc_score(y, eri)), 3),
        "auc_sana_vs_gingivitis_a": round(float(roc_auc_score(y, a)), 3),
        "eritema_mediana": {"mio_sano": round(float(np.median(sanas_mio)), 4),
                            "mio_gingivitis": round(float(np.median(eri[y == 1])), 4),
                            "mendeley_sano_franjas": round(float(np.median(sanas_mendeley)), 4)},
        "p10_sanas": {"mendeley": round(float(np.percentile(sanas_mendeley, 10)), 4),
                      "mio": round(float(np.percentile(sanas_mio, 10)), 4),
                      "combinada": round(float(np.percentile(sanas_nueva, 10)), 4)},
        "gingivitis_mio_marcada_inflamada": {
            "referencia_v1": round(float((eri[y == 1] >= ref["umbral_inflamacion"]).mean()), 3),
            "referencia_v2": round(float((eri[y == 1] >= umbral_nuevo).mean()), 3)},
        "sanas_mio_marcadas_palidas_v2": round(float((pct(sanas_mio, sanas_nueva) <= ref["pct_palidez"]).mean()), 3),
        "nota": "Sin hemoglobina: valida que el medidor ve el color de la encía, no la anemia.",
    }
    ref.update({
        "sanas_eritema": np.round(sanas_nueva, 5).tolist(),
        "umbral_inflamacion": umbral_nuevo,
        "n_sanas": int(len(sanas_nueva)),
        "n_sanas_mendeley_franjas": int(len(sanas_mendeley)),
        "n_sanas_mio_fotos": int(len(sanas_mio)),
        "validacion_mio": reporte,
        "datos": "Mendeley 3253gj88rr (CC BY 4.0) + MIO Zenodo 22047269 (CC BY 4.0) + dientes: Dentalai (MIT)",
        "version": 2,
    })
    ref_path.write_text(json.dumps(ref, indent=2, ensure_ascii=False))
    (REPORTE / "mio.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False))
    print(json.dumps(reporte, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
