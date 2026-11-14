"""Detección de palidez de la conjuntiva (signo de anemia) en una foto del ojo.

La persona jala su párpado inferior hacia abajo; la conjuntiva palpebral (la parte interna,
rosada/roja) se vuelve pálida cuando la hemoglobina es baja. Es el signo clínico clásico.

Pasos:
1. U-Net (``models/conjuntiva/best.keras``) → máscara de la conjuntiva.
2. Balance de blancos con la esclerótica (lo blanco del ojo) como referencia: cada teléfono
   y cada foco dan otro color, y sin esto el rojo medido no es comparable.
3. Medidas de color dentro de la conjuntiva (índice de eritema, a* de CIELAB, cromaticidad…).
4. Clasificador pequeño (``models/anemia/clasificador.joblib``) → probabilidad de palidez.

Es una herramienta de tamizaje/investigación: la anemia se confirma con una biometría
hemática (hemoglobina en sangre).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LADO = 512
CARACTERISTICAS = (
    "eritema", "eritema_p75", "a_media", "a_p25", "a_p75", "l_media", "b_media",
    "crom_r", "crom_g", "rojo_alto", "saturacion", "r_sobre_g",
)


def letterbox(img: np.ndarray, lado: int = LADO) -> tuple[np.ndarray, np.ndarray]:
    """Cuadra sin deformar (relleno negro); devuelve la imagen y la máscara del área útil."""
    h, w = img.shape[:2]
    s = lado / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    t, l = (lado - nh) // 2, (lado - nw) // 2
    out = cv2.copyMakeBorder(img, t, lado - nh - t, l, lado - nw - l, cv2.BORDER_CONSTANT, value=0)
    util = np.zeros((lado, lado), bool)
    util[t : t + nh, l : l + nw] = True
    return out, util


def mayor_componente(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def blanco_de_referencia(rgb: np.ndarray, conj: np.ndarray, util: np.ndarray) -> tuple[np.ndarray, int]:
    """Color de la esclerótica: píxeles claros y poco saturados fuera de la conjuntiva.

    Returns:
        (color RGB float en [0, 1], número de píxeles usados); con < 150 píxeles no es fiable.
    """
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    fuera = util & ~cv2.dilate(conj.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    s, v = hsv[..., 1], hsv[..., 2]
    # Umbrales relativos a la propia foto: con un foco cálido el blanco del ojo sale saturado
    # (S≈0.4), así que no sirve un límite fijo bajo. Se toman los más claros y, de esos, los
    # menos saturados (la piel es más saturada que la esclerótica bajo la misma luz).
    cand = fuera & (s < 0.60) & (v > 0.35) & (v < 0.995)  # sin sombras ni reflejos quemados
    if cand.sum() < 150:
        return np.ones(3, np.float32), int(cand.sum())
    claros = cand & (v >= np.percentile(v[cand], 75))
    sel = claros & (s <= np.percentile(s[claros], 50))
    return np.median(rgb[sel], axis=0).astype(np.float32), int(sel.sum())


def medir_color(rgb: np.ndarray, conj: np.ndarray, blanco: np.ndarray) -> dict[str, float]:
    """Medidas de rojez dentro de la conjuntiva, con el balance de blancos aplicado."""
    gan = blanco.mean() / np.clip(blanco, 0.05, None)
    corr = np.clip(rgb * gan, 0, 1)
    px = corr[conj]
    # sin brillos especulares ni sombras profundas
    v = px.max(1)
    px = px[(v > 0.08) & (v < 0.98)]
    if len(px) < 50:
        px = corr[conj]
    eps = 1e-3
    r, g, b = px[:, 0] + eps, px[:, 1] + eps, px[:, 2] + eps
    eritema = np.log(r) - np.log(g)
    lab = cv2.cvtColor(px[None].astype(np.float32), cv2.COLOR_RGB2Lab)[0]  # L 0..100, a/b ≈ -127..127
    hsv = cv2.cvtColor(px[None].astype(np.float32), cv2.COLOR_RGB2HSV)[0]  # H 0..360, S 0..1
    h, s = hsv[:, 0], hsv[:, 1]
    rojo = ((h < 20) | (h > 340)) & (s > 0.35)
    suma = r + g + b
    return {
        "eritema": float(eritema.mean()),
        "eritema_p75": float(np.percentile(eritema, 75)),
        "a_media": float(lab[:, 1].mean()),
        "a_p25": float(np.percentile(lab[:, 1], 25)),
        "a_p75": float(np.percentile(lab[:, 1], 75)),
        "l_media": float(lab[:, 0].mean()),
        "b_media": float(lab[:, 2].mean()),
        "crom_r": float((r / suma).mean()),
        "crom_g": float((g / suma).mean()),
        "rojo_alto": float(rojo.mean()),
        "saturacion": float(s.mean()),
        "r_sobre_g": float(np.median(r / g)),
    }


class SegmentadorConjuntiva:
    def __init__(self, model_path: str | Path) -> None:
        import tensorflow as tf

        import src.models.unet  # noqa: F401  registra CaffePreprocess y ResizeLike para load_model

        self.tf = tf
        self.model = tf.keras.models.load_model(model_path, compile=False)
        pp = Path(model_path).with_name("preprocess.json")
        self.umbral = float(json.loads(pp.read_text()).get("threshold", 0.5)) if pp.exists() else 0.5

    def __call__(self, rgb01: np.ndarray) -> np.ndarray:
        """Probabilidad de conjuntiva (LADO×LADO) con TTA de espejo."""
        tf = self.tf
        x = tf.constant(rgb01[None].astype(np.float32))
        p = tf.sigmoid(tf.cast(self.model(x, training=False), tf.float32)).numpy()[0, ..., 0]
        q = tf.sigmoid(tf.cast(self.model(x[:, :, ::-1], training=False), tf.float32)).numpy()[0, :, ::-1, 0]
        return (p + q) / 2


def extraer(seg: SegmentadorConjuntiva, image_rgb: np.ndarray) -> dict[str, Any]:
    """Segmenta y mide. ``image_rgb`` uint8 de cualquier tamaño."""
    img, util = letterbox(image_rgb)
    rgb = img.astype(np.float32) / 255.0
    prob = seg(rgb) * util
    conj = mayor_componente(prob >= seg.umbral)
    frac = float(conj.sum() / max(util.sum(), 1))
    out: dict[str, Any] = {"fraccion_conjuntiva": round(frac, 4), "mascara": conj, "imagen": img}
    if conj.sum() < 300:
        out["valida"] = False
        return out
    blanco, n_blanco = blanco_de_referencia(rgb, conj, util)
    out.update(valida=True, blanco=blanco.round(4).tolist(), pixeles_blanco=n_blanco,
               medidas=medir_color(rgb, conj, blanco))
    return out


class AnalizadorAnemia:
    """Segmentador + clasificador de palidez. ``predict(rgb)`` → dict serializable."""

    def __init__(self, seg_path: str | Path, clf_path: str | Path) -> None:
        import joblib

        self.seg = SegmentadorConjuntiva(seg_path)
        paquete = joblib.load(clf_path)
        self.clf = paquete["modelo"]
        self.umbral = float(paquete["umbral"])
        self.zona = tuple(paquete.get("zona_gris", (self.umbral, self.umbral)))
        self.cv = paquete.get("validacion", {})
        self.nombre = Path(clf_path).parent.name

    def predict(self, image_rgb: np.ndarray) -> dict[str, Any]:
        e = extraer(self.seg, image_rgb)
        base = {"fraccion_conjuntiva": e["fraccion_conjuntiva"], "model": self.nombre,
                "aviso": "Tamizaje de investigación: la anemia solo se confirma con un análisis de sangre "
                         "(hemoglobina)."}
        if not e["valida"]:
            return {**base, "valida": False, "resultado": "sin_conjuntiva",
                    "titulo": "No se ve la parte interna del párpado",
                    "descripcion": "Jala tu párpado inferior hacia abajo con un dedo y mira hacia arriba, "
                                   "para que se vea bien la parte rosada de adentro."}
        x = np.array([[e["medidas"][k] for k in CARACTERISTICAS]])
        p = float(self.clf.predict_proba(x)[0, 1])
        bajo, alto = self.zona
        sin_ref = e["pixeles_blanco"] < 150
        if p >= alto:
            resultado, titulo, desc = "palidez", "Conjuntiva pálida: posible anemia", (
                "El interior del párpado se ve más pálido de lo normal. Conviene hacer una biometría "
                "hemática para medir la hemoglobina.")
        elif p < bajo:
            resultado, titulo, desc = "sin_palidez", "Color de conjuntiva normal", (
                "El interior del párpado tiene buena coloración; no se ven signos de anemia.")
        else:
            # Zona gris: aquí el modelo se equivoca casi la mitad de las veces; mejor no adivinar.
            resultado, titulo, desc = "no_concluyente", "Resultado no concluyente", (
                "El color quedó en un punto intermedio y no se puede decir con confianza. Repite la foto "
                "con luz blanca de día, sin flash, jalando bien el párpado. Si vuelve a salir así, "
                "hazte una biometría hemática.")
        res = {
            **base,
            "valida": True,
            "resultado": resultado,
            "probabilidad_anemia": round(p, 4),
            "umbral": round(self.umbral, 4),
            "zona_gris": [round(bajo, 4), round(alto, 4)],
            "titulo": titulo,
            "descripcion": desc,
            "medidas": {k: round(v, 4) for k, v in e["medidas"].items()},
            "balance_blancos": not sin_ref,
            "validacion": self.cv,
        }
        if sin_ref:
            res["advertencia_luz"] = ("No se vio suficiente blanco del ojo para corregir la luz; el resultado "
                                      "es menos fiable. Toma la foto con luz blanca y que se vea el ojo completo.")
        return res
