"""API REST (FastAPI) para los modelos de Eye-Retina AI.

Arranque::

    uvicorn src.deployment.api:app --reload --port 8000

Variables de entorno:

* ``EYE_MODEL_PATH``  – detector de un ojo (por defecto ``models/eye_best.h5``).
* ``GRID_MODEL_PATH`` – detector multi-ojo de rejilla (opcional).
* ``RETINA_MODEL_PATH`` – U-Net de vasos (por defecto ``models/retina/best.keras``).

Endpoints:

* ``GET  /``                  – health check.
* ``POST /predict/eye``       – caja, landmarks y confianza de un ojo.
* ``POST /predict/eyes``      – todos los ojos de un rostro (detector de rejilla).
* ``POST /predict/retina``    – segmentación de vasos (densidad + máscara PNG opcional).
* ``POST /predict/disease``   – Fase 2 (501).
* ``POST /captures/eye``      – guarda un recorte de ojo de la app OjoScan (con consentimiento)
  en ``CAPTURES_DIR`` (por defecto ``data/captures/eyes``) como dato de entrenamiento.
* ``GET  /docs``              – Swagger automático.

Los errores se devuelven como ``{"error": {"code": ..., "message": ...}}``.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT
from src.data.io import decode_image_bytes

log = logging.getLogger("eye-retina-api")
MAX_BYTES = 15 * 1024 * 1024
VERSION = "0.3.0"

app = FastAPI(
    title="Eye-Retina AI",
    version=VERSION,
    description="Detección de ojos y análisis de retina. Herramienta de investigación: "
    "no sustituye el diagnóstico de un oftalmólogo.",
)


# --------------------------------------------------------------------------- #
# Esquemas
# --------------------------------------------------------------------------- #
class Landmark(BaseModel):
    """Un punto del ojo."""

    name: str
    x: float
    y: float


class EyePrediction(BaseModel):
    """Respuesta de ``/predict/eye``."""

    bbox: list[float] = Field(..., description="x, y, ancho, alto en píxeles de la imagen original")
    landmarks: list[Landmark]
    confidence: float = Field(..., description="probabilidad de que haya un ojo (0-1)")
    is_eye: bool
    image_size: list[int] = Field(..., description="[ancho, alto]")
    model: str


class ErrorBody(BaseModel):
    """Detalle de error."""

    code: int
    message: str


# --------------------------------------------------------------------------- #
# Manejo de errores estructurado
# --------------------------------------------------------------------------- #
@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    """Errores esperados (400, 404, 413, 501, 503)."""
    return JSONResponse(
        status_code=exc.status_code, content={"error": {"code": exc.status_code, "message": str(exc.detail)}}
    )


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    """Errores no previstos: se registran y se responde 500 sin filtrar detalles internos."""
    log.exception("Error no controlado: %s", exc)
    return JSONResponse(
        status_code=500, content={"error": {"code": 500, "message": "Error interno del servidor"}}
    )


# --------------------------------------------------------------------------- #
# Modelos (carga perezosa)
# --------------------------------------------------------------------------- #
def _model_path(var: str, default: str) -> Path:
    p = Path(os.environ.get(var, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=1)
def get_eye_detector() -> Any:
    """Detector de un ojo (se carga una vez por proceso)."""
    from src.deployment.inference_eye import EyeDetector

    path = _model_path("EYE_MODEL_PATH", "models/eye_best.h5")
    if not path.exists():
        raise HTTPException(
            503, f"Modelo no disponible ({path.name}). Pendiente de entrenamiento o define EYE_MODEL_PATH."
        )
    return EyeDetector(path)


@lru_cache(maxsize=1)
def get_eye_analyzer() -> Any:
    """Analizador de ojo externo (normal / enrojecimiento / pterigión / catarata)."""
    from src.deployment.inference_ojo_externo import AnalizadorOjoExterno

    path = _model_path("EYE_ANALYSIS_MODEL_PATH", "models/ojo_externo/best.keras")
    if not path.exists():
        raise HTTPException(503, f"Modelo no disponible ({path.name}). Entrénalo con train_ojo_externo.py.")
    return AnalizadorOjoExterno(path)


@lru_cache(maxsize=1)
def get_grid_detector() -> Any:
    """Detector multi-ojo de rejilla (opcional)."""
    from src.deployment.predictor_grid import GridEyeDetector

    path = _model_path("GRID_MODEL_PATH", "models/eye_grid/eye_grid_mobilenetv2_224/best.keras")
    if not path.exists():
        raise HTTPException(503, f"Detector multi-ojo no disponible ({path.name}).")
    return GridEyeDetector(path)


@lru_cache(maxsize=1)
def get_retina_segmenter() -> Any:
    """U-Net de vasos de retina (se carga una vez por proceso)."""
    from src.deployment.inference_retina import RetinaSegmenter

    path = _model_path("RETINA_MODEL_PATH", "models/retina/best.keras")
    if not path.exists():
        raise HTTPException(503, f"Modelo de retina no disponible ({path.name}). Entrénalo con train_retina.py.")
    return RetinaSegmenter(path)


def png_base64(arr: Any) -> str:
    """Codifica un arreglo uint8 (gris o RGB) como PNG en base64."""
    import base64

    import cv2

    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise HTTPException(500, "No se pudo codificar la máscara")
    return base64.b64encode(buf.tobytes()).decode("ascii")


async def read_upload(file: UploadFile) -> Any:
    """Lee y decodifica la imagen subida (400 si no es válida)."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "El archivo está vacío")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Imagen demasiado grande (máx. 15 MB)")
    try:
        return decode_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(400, f"No es una imagen válida: {exc}") from exc


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_rostro_analyzer() -> Any:
    """Rostro (parálisis facial, ictericia) + ojos de la misma foto. Necesita servicio_rostro en :8001."""
    from src.deployment.inference_rostro import AnalizadorRostro

    carpeta = _model_path("ROSTRO_MODEL_DIR", "models/rostro")
    for n in ("paralisis.joblib", "ictericia.joblib"):
        if not (carpeta / n).exists():
            raise HTTPException(503, f"Modelo de rostro no disponible ({n}).")
    return AnalizadorRostro(carpeta, analizador_ojo=get_eye_analyzer)


@lru_cache(maxsize=1)
def get_palma_analyzer() -> Any:
    """Palidez de la palma de la mano. Necesita servicio_rostro (manos) en :8001."""
    from src.deployment.inference_palma import AnalizadorPalma

    ruta = _model_path("PALMA_MODEL_PATH", "models/palma/clasificador.joblib")
    if not ruta.exists():
        raise HTTPException(503, "Modelo de palma no disponible. Entrénalo con train_anemia_palma.py.")
    return AnalizadorPalma(ruta)


@lru_cache(maxsize=1)
def get_anemia_analyzer() -> Any:
    """Palidez de conjuntiva: U-Net de conjuntiva + clasificador de color."""
    from src.deployment.inference_anemia import AnalizadorAnemia

    seg = _model_path("CONJUNTIVA_MODEL_PATH", "models/conjuntiva/best.keras")
    clf = _model_path("ANEMIA_MODEL_PATH", "models/anemia/clasificador.joblib")
    for p in (seg, clf):
        if not p.exists():
            raise HTTPException(503, f"Modelo de anemia no disponible ({p.name}). Entrénalo con train_anemia.py.")
    return AnalizadorAnemia(seg, clf)


@lru_cache(maxsize=1)
def get_encia_analyzer() -> Any:
    """Palidez de la encía (experimental). Necesita servicio_rostro (boca) en :8001."""
    from src.deployment.inference_encia import AnalizadorEncia

    dientes = _model_path("DIENTES_MODEL_DIR", "models/dientes")
    ref = _model_path("ENCIA_REFERENCIA", "models/encia/referencia.json")
    if not (dientes / "best.keras").exists() or not ref.exists():
        raise HTTPException(503, "Modelo no disponible: encía. Entrena train_dientes y referencia_encias.")
    return AnalizadorEncia(dientes, ref)


@lru_cache(maxsize=1)
def get_perfil_analyzer() -> Any:
    """Perfil médico con los datos que escribe el paciente (modelos de ~/Deep_Learning)."""
    from src.deployment.inference_perfil import AnalizadorPerfil

    carpeta = _model_path("PERFIL_MODEL_DIR", "models/perfil")
    if not (carpeta / "resultados.json").exists():
        raise HTTPException(503, "Modelo no disponible: perfil. Entrénalo con python -m src.training.train_perfil.")
    return AnalizadorPerfil(carpeta)


@app.get("/")
def health() -> dict[str, Any]:
    """Health check: estado del servicio y disponibilidad de los modelos."""
    eye = _model_path("EYE_MODEL_PATH", "models/eye_best.h5")
    retina = _model_path("RETINA_MODEL_PATH", "models/retina/best.keras")
    analisis = _model_path("EYE_ANALYSIS_MODEL_PATH", "models/ojo_externo/best.keras")
    anemia = _model_path("ANEMIA_MODEL_PATH", "models/anemia/clasificador.joblib")
    conj = _model_path("CONJUNTIVA_MODEL_PATH", "models/conjuntiva/best.keras")
    return {
        "status": "ok",
        "version": VERSION,
        "models": {
            "eye": {"path": eye.name, "available": eye.exists()},
            "retina": {"path": retina.name, "available": retina.exists()},
            "eye_analysis": {"path": analisis.name, "available": analisis.exists()},
            "anemia": {"path": anemia.name, "available": anemia.exists() and conj.exists()},
            "encia": {"path": "dientes+referencia", "available": (_model_path("DIENTES_MODEL_DIR", "models/dientes") / "best.keras").exists()
                      and _model_path("ENCIA_REFERENCIA", "models/encia/referencia.json").exists()},
            "perfil": {"path": "perfil", "available": (_model_path("PERFIL_MODEL_DIR", "models/perfil") / "resultados.json").exists()},
            "corazon": {"path": "corazon", "available": (PROJECT_ROOT / "models/corazon/fa.joblib").exists()},
        },
    }


@app.post(
    "/predict/eye",
    response_model=EyePrediction,
    responses={400: {"model": ErrorBody}, 500: {"model": ErrorBody}},
)
async def predict_eye(
    file: UploadFile = File(..., description="imagen JPG/PNG/TIFF"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="umbral de confianza"),
) -> EyePrediction:
    """Localiza un ojo: caja, 6 landmarks y confianza."""
    image = await read_upload(file)
    det = get_eye_detector()
    pred = det.predict(image)
    pred["is_eye"] = pred["confidence"] >= threshold
    h, w = image.shape[:2]
    return EyePrediction(**pred, image_size=[w, h], model=det.model_path.name)


@app.post("/predict/eye/analysis", responses={400: {"model": ErrorBody}, 503: {"model": ErrorBody}})
async def predict_eye_analysis(file: UploadFile = File(..., description="foto del ojo (recorte de OjoScan)")) -> dict[str, Any]:
    """Analiza la foto externa del ojo: hallazgo principal y probabilidad de cada clase."""
    image = await read_upload(file)
    return get_eye_analyzer().predict(image)


@app.post("/predict/rostro", responses={400: {"model": ErrorBody}, 503: {"model": ErrorBody}})
async def predict_rostro(file: UploadFile = File(..., description="foto de frente de toda la cara")) -> dict[str, Any]:
    """Asimetría facial (parálisis) e ictericia, cruzadas con el análisis de los ojos de la misma foto."""
    image = await read_upload(file)
    return get_rostro_analyzer().predict(image)


@app.post("/predict/anemia", responses={400: {"model": ErrorBody}, 503: {"model": ErrorBody}})
async def predict_anemia(
    file: UploadFile = File(..., description="foto del ojo con el párpado inferior jalado hacia abajo"),
) -> dict[str, Any]:
    """Tamizaje de anemia por palidez de la conjuntiva palpebral."""
    image = await read_upload(file)
    return get_anemia_analyzer().predict(image)


@app.post("/predict/palma", responses={400: {"model": ErrorBody}, 503: {"model": ErrorBody}})
async def predict_palma(
    file: UploadFile = File(..., description="foto de la palma de la mano abierta"),
) -> dict[str, Any]:
    """Tamizaje de anemia por palidez de la palma (misma respuesta que /predict/anemia)."""
    image = await read_upload(file)
    return get_palma_analyzer().predict(image)


@app.post("/predict/encia", responses={400: {"model": ErrorBody}, 503: {"model": ErrorBody}})
async def predict_encia(file: UploadFile = File(..., description="foto de frente sonriendo con la encía de arriba visible")) -> dict[str, Any]:
    """Anemia por palidez de la encía (EXPERIMENTAL, sin validar con hemoglobina). Misma forma que /predict/anemia."""
    datos = await file.read()
    try:
        image = decode_image_bytes(datos)
    except ValueError:
        raise HTTPException(400, "El archivo no es una imagen válida")
    return get_encia_analyzer().predict(datos, image)


@app.post("/predict/perfil", responses={400: {"model": ErrorBody}, 503: {"model": ErrorBody}})
async def predict_perfil(file: UploadFile = File(..., description="datos.json con edad, medidas, presión, síntomas")) -> dict[str, Any]:
    """Perfil médico. Los datos llegan como archivo JSON para pasar por el intermediario del VPS."""
    import json

    try:
        datos = json.loads((await file.read()).decode("utf-8"))
        assert isinstance(datos, dict)
    except Exception:
        raise HTTPException(400, "datos.json no es un objeto JSON válido")
    return get_perfil_analyzer().predict(datos)


@lru_cache(maxsize=1)
def get_corazon_analyzer() -> Any:
    from src.deployment.inference_corazon import AnalizadorCorazon
    carpeta = PROJECT_ROOT / "models/corazon"
    if not (carpeta / "fa.joblib").exists():
        raise HTTPException(503, "Modelo cardiaco no disponible: corre src/training/train_corazon.py")
    return AnalizadorCorazon(carpeta)


@app.post("/predict/corazon", responses={400: {"model": ErrorBody}, 503: {"model": ErrorBody}})
async def predict_corazon(file: UploadFile = File(..., description="corazon.json: serie de pulso a 1 Hz, historial, edad, síntomas")) -> dict[str, Any]:
    """Vigilancia cardiaca con la pulsera: IA de ritmo + comparación con tu historial + alertas de prevención."""
    import json

    try:
        datos = json.loads((await file.read()).decode("utf-8"))
        assert isinstance(datos, dict)
    except Exception:
        raise HTTPException(400, "corazon.json no es un objeto JSON válido")
    return get_corazon_analyzer().predict(datos)


@app.post("/predict/eyes")
async def predict_eyes(
    file: UploadFile = File(...), threshold: float = Query(0.5, ge=0.0, le=1.0)
) -> dict[str, Any]:
    """Detecta todos los ojos de un rostro con el detector de rejilla."""
    image = await read_upload(file)
    det = get_grid_detector()
    det.score_threshold = threshold
    dets = det.predict(image)
    return {"num_eyes": len(dets), "detections": dets}


class RetinaPrediction(BaseModel):
    """Respuesta de ``/predict/retina``."""

    vessel_density: float = Field(..., description="fracción del campo de visión ocupada por vasos (0-1)")
    vessel_pixels: int
    fov_pixels: int
    threshold: float
    image_size: list[int] = Field(..., description="[ancho, alto]")
    model: str
    mask_png: str | None = Field(None, description="máscara binaria PNG en base64 (si include=mask|both)")
    overlay_png: str | None = Field(None, description="vasos sobre la imagen, PNG en base64 (si include=overlay|both)")


@app.post(
    "/predict/retina",
    response_model=RetinaPrediction,
    responses={400: {"model": ErrorBody}, 503: {"model": ErrorBody}},
)
async def predict_retina(
    file: UploadFile = File(..., description="fondo de ojo JPG/PNG/TIFF"),
    threshold: float | None = Query(None, ge=0.0, le=1.0, description="umbral; por defecto el elegido en validación"),
    include: str = Query("none", pattern="^(none|mask|overlay|both)$", description="imágenes a devolver"),
) -> RetinaPrediction:
    """Segmenta los vasos de un fondo de ojo (U-Net entrenada en FIVES)."""
    image = await read_upload(file)
    seg = get_retina_segmenter()
    out = seg.predict(image, threshold=threshold)
    mask, fov = out["mask"], out["fov"]
    h, w = image.shape[:2]
    return RetinaPrediction(
        vessel_density=round(out["vessel_density"], 5),
        vessel_pixels=int(mask.sum()),
        fov_pixels=int(fov.sum()),
        threshold=out["threshold"],
        image_size=[w, h],
        model=seg.model_path.name,
        mask_png=png_base64(mask.astype("uint8") * 255) if include in ("mask", "both") else None,
        overlay_png=png_base64(seg.overlay(image, mask)) if include in ("overlay", "both") else None,
    )


@app.post("/predict/disease")
async def predict_disease(file: UploadFile = File(...)) -> dict[str, Any]:
    """Clasificación multilabel de enfermedades (Parte B)."""
    raise HTTPException(501, "Clasificación de enfermedades pendiente (Parte B)")


# --------------------------------------------------------------------------- #
# Capturas de la app OjoScan → datos de entrenamiento
# --------------------------------------------------------------------------- #
CAPTURE_LANDMARKS = ("inner_corner", "outer_corner", "upper_lid", "lower_lid", "pupil", "iris")


class CaptureMeta(BaseModel):
    """Metadatos que manda la app junto al recorte (coordenadas en px del recorte)."""

    ojo: str = Field(..., pattern="^(izquierdo|derecho)$")
    caja: list[float] | None = Field(None, min_length=4, max_length=4, description="x, y, ancho, alto")
    puntos: dict[str, list[float]] = Field(default_factory=dict)
    apertura: float | None = None
    consentimiento: bool
    origen: str = "ojoscan"


def captures_dir() -> Path:
    """Carpeta de capturas (formato CSV de docs/FORMATO_ETIQUETAS.md)."""
    p = Path(os.environ.get("CAPTURES_DIR", "data/captures/eyes"))
    return p if p.is_absolute() else PROJECT_ROOT / p


@app.post("/captures/eye", responses={400: {"model": ErrorBody}})
async def capture_eye(
    file: UploadFile = File(..., description="recorte JPG del ojo"),
    meta: str = Form(..., description="JSON con ojo, caja, puntos, apertura y consentimiento"),
) -> dict[str, str]:
    """Guarda el recorte y sus etiquetas automáticas (ML Kit) para entrenar el detector.

    La fila se agrega a ``annotations.csv``; la carpeta se puede usar directo como
    ``data.data_dir`` del detector (``label_format: csv``). Las etiquetas vienen del teléfono:
    conviene revisarlas antes de entrenar.
    """
    import csv
    import uuid
    from datetime import datetime

    import cv2

    try:
        m = CaptureMeta.model_validate_json(meta)
    except ValueError as exc:
        raise HTTPException(400, f"Metadatos inválidos: {exc}") from exc
    if not m.consentimiento:
        raise HTTPException(400, "Falta el consentimiento para guardar la foto")
    image = await read_upload(file)

    root = captures_dir()
    (root / "images").mkdir(parents=True, exist_ok=True)
    cid = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}-{m.ojo}"
    rel = f"images/{cid}.jpg"
    cv2.imwrite(str(root / rel), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

    header = ["image", "x", "y", "w", "h"]
    for n in CAPTURE_LANDMARKS:
        header += [f"{n}_x", f"{n}_y"]
    header += ["eye_side", "eye_open_prob", "source"]
    row: list[Any] = [rel, *(m.caja or ["", "", "", ""])]
    for n in CAPTURE_LANDMARKS:
        pt = m.puntos.get(n)
        row += [pt[0], pt[1]] if pt and len(pt) == 2 else ["", ""]
    row += [m.ojo, "" if m.apertura is None else round(m.apertura, 4), m.origen]

    csv_path = root / "annotations.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        w.writerow(row)
    log.info("Captura guardada: %s", rel)
    return {"id": cid, "image": rel}
