"""Señal de pulso tal como la manda una pulsera (Xiaomi Smart Band por BLE): un valor en lpm cada ~1 s,
suavizado y redondeado, sin intervalos latido a latido.

Lo comparten el entrenamiento (src/training/train_corazon.py, que la simula a partir de los latidos
anotados de PhysioNet) y la inferencia (src/deployment/inference_corazon.py, que la recibe de la app),
para que las variables se calculen exactamente igual en los dos lados.
"""
from __future__ import annotations

import numpy as np

VENTANA_S = 120  # 2 minutos: lo que dura una lectura de la app
MIN_PUNTOS = 60

VARIABLES = [
    "media", "de", "cv", "minimo", "maximo", "rango", "p10", "p90",
    "dif_abs", "dif_de", "dif_3", "dif_6", "dif_10", "rmssd", "giros", "ac1", "ac5", "ac10",
    "pend", "baja_freq", "alta_freq", "entropia",
]


def latidos_a_banda(t_latidos: np.ndarray, rng: np.random.Generator | None = None,
                    suave_s: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Latidos (segundos) → serie a 1 Hz como la de la pulsera.

    Cada segundo la pulsera reporta el pulso medio de los últimos `suave_s` segundos. Se agrega ruido
    pequeño de medición óptica y se redondea a entero. Huecos (> 3 s sin latido) quedan como NaN.
    """
    rr = np.diff(t_latidos)
    t = t_latidos[1:]
    ok = (rr > 0.25) & (rr < 3.0)
    hr = np.where(ok, 60.0 / np.clip(rr, 1e-3, None), np.nan)
    seg = np.arange(np.ceil(t[0] + suave_s), np.floor(t[-1]))
    if len(seg) == 0:
        return seg, seg
    # suma acumulada para promediar por ventana deslizante en O(n)
    validos = np.where(np.isnan(hr), 0.0, hr)
    cs = np.concatenate([[0.0], np.cumsum(validos)])
    cn = np.concatenate([[0], np.cumsum(~np.isnan(hr))])
    fin = np.searchsorted(t, seg, side="right")
    ini = np.searchsorted(t, seg - suave_s, side="left")
    n = cn[fin] - cn[ini]
    with np.errstate(invalid="ignore", divide="ignore"):
        serie = (cs[fin] - cs[ini]) / n
    serie[n < max(2, suave_s * 0.4)] = np.nan
    if rng is not None:
        serie = serie + rng.normal(0, 0.8, size=serie.shape)
    return seg, np.round(serie)


def variables(serie: np.ndarray) -> dict[str, float] | None:
    """Variables de una ventana a 1 Hz (lpm). None si hay muy pocos puntos válidos."""
    x = np.asarray(serie, dtype=float)
    x = x[~np.isnan(x)]
    x = x[(x >= 30) & (x <= 230)]
    if len(x) < MIN_PUNTOS:
        return None
    d = np.diff(x)
    ad = np.abs(d)
    m = float(x.mean())
    xc = x - m
    var = float((xc ** 2).mean()) or 1e-6

    def ac(k: int) -> float:
        return float((xc[:-k] * xc[k:]).mean() / var) if len(x) > k + 5 else 0.0

    giros = float(np.mean((d[1:] * d[:-1]) < 0)) if len(d) > 2 else 0.0
    pend = float(np.polyfit(np.arange(len(x)), x, 1)[0])
    esp = np.abs(np.fft.rfft(xc)) ** 2
    f = np.fft.rfftfreq(len(xc), d=1.0)
    tot = float(esp[1:].sum()) or 1e-6
    baja = float(esp[(f >= 0.01) & (f < 0.05)].sum() / tot)
    alta = float(esp[(f >= 0.15)].sum() / tot)
    # entropía de los cambios cuantizados (qué tan impredecible es el siguiente salto)
    q = np.clip(np.round(d), -8, 8)
    _, cuenta = np.unique(q, return_counts=True)
    p = cuenta / cuenta.sum()
    entropia = float(-(p * np.log2(p)).sum())
    return {
        "media": m, "de": float(x.std()), "cv": float(x.std() / m), "minimo": float(x.min()), "maximo": float(x.max()),
        "rango": float(x.max() - x.min()), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
        "dif_abs": float(ad.mean()), "dif_de": float(d.std()), "dif_3": float(np.mean(ad >= 3)),
        "dif_6": float(np.mean(ad >= 6)), "dif_10": float(np.mean(ad >= 10)), "rmssd": float(np.sqrt((d ** 2).mean())),
        "giros": giros, "ac1": ac(1), "ac5": ac(5), "ac10": ac(10), "pend": pend,
        "baja_freq": baja, "alta_freq": alta, "entropia": entropia,
    }
