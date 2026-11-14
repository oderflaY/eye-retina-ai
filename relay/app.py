"""Intermediario OjoScan (corre en el VPS).

El teléfono habla con este servicio por HTTPS como si fuera la API de la IA (mismas rutas). La PC
que tiene los modelos NO recibe conexiones: su *worker* (src/deployment/relay_worker.py) se
conecta hacia acá, pide trabajos con long-polling, los ejecuta en su API local y devuelve la
respuesta. Así no hay que abrir puertos en la red de la casa.

* Predicciones (``POST /predict/...``): el teléfono espera la respuesta de la PC (hasta
  ``ESPERA_PREDICCION`` s). Si la PC no está conectada responde 502 de inmediato.
* Capturas (``POST /captures/eye``): se guardan en disco aquí y se responde al momento; la PC las
  recibe cuando se conecte (aunque haya estado apagada). Se reintenta si la PC no confirma.

Rutas del worker (``/worker/*``) protegidas con ``Authorization: Bearer $RELAY_TOKEN``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.datastructures import UploadFile

VERSION = "1.0.0"
TOKEN = os.environ.get("RELAY_TOKEN", "")
DATOS = Path(os.environ.get("DATA_DIR", "/data"))
ESPERA_PREDICCION = float(os.environ.get("ESPERA_PREDICCION", "55"))
PC_VIVA_S = 45.0          # sin pedir trabajo en este tiempo → la PC se considera desconectada
REINTENTO_CAPTURA_S = 120  # captura entregada sin confirmar → vuelve a la cola
MAX_BYTES = 15 * 1024 * 1024

log = logging.getLogger("ojoscan-relay")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = FastAPI(title="OjoScan relay", version=VERSION)

DESCARGAS = DATOS / "descargas"     # OjoScan.apk para instalar sin cable
PENDIENTES = DATOS / "pendientes"   # capturas que la PC aún no confirma
ENTREGADAS = DATOS / "entregadas"   # respaldo de lo que ya llegó a la PC


class Estado:
    def __init__(self) -> None:
        self.cola: asyncio.Queue[str] = asyncio.Queue()
        self.trabajos: dict[str, dict[str, Any]] = {}       # predicciones en memoria
        self.esperas: dict[str, asyncio.Future] = {}
        self.entregando: dict[str, float] = {}              # captura → cuándo se entregó
        self.pc_visto = 0.0
        self.pc_salud: dict[str, Any] = {}

    def pc_conectada(self) -> bool:
        return time.monotonic() - self.pc_visto < PC_VIVA_S


E = Estado()


def error(code: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": msg}}, status_code=code)


@app.exception_handler(HTTPException)
async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
    return error(exc.status_code, str(exc.detail))


def autorizar(authorization: str | None) -> None:
    esperado = f"Bearer {TOKEN}"
    if not TOKEN or not authorization or not secrets.compare_digest(authorization, esperado):
        raise HTTPException(401, "Token del worker inválido")


# --------------------------------------------------------------------------- #
# Arranque: recuperar capturas pendientes del disco
# --------------------------------------------------------------------------- #
@app.on_event("startup")
async def _arranque() -> None:
    PENDIENTES.mkdir(parents=True, exist_ok=True)
    ENTREGADAS.mkdir(parents=True, exist_ok=True)
    for f in sorted(PENDIENTES.glob("*.json")):
        E.cola.put_nowait(f.stem)
    log.info("Capturas pendientes al arrancar: %d", E.cola.qsize())
    asyncio.create_task(_reintentos())


async def _reintentos() -> None:
    while True:
        await asyncio.sleep(30)
        ahora = time.monotonic()
        for cid, t in list(E.entregando.items()):
            if ahora - t > REINTENTO_CAPTURA_S:
                E.entregando.pop(cid, None)
                if (PENDIENTES / f"{cid}.json").exists():
                    log.warning("La PC no confirmó %s, se reintenta", cid)
                    E.cola.put_nowait(cid)


# --------------------------------------------------------------------------- #
# Descarga de la app (abrir https://2.25.68.28:13443/app en el navegador del teléfono)
# --------------------------------------------------------------------------- #
@app.get("/app", response_class=HTMLResponse)
def pagina_app() -> str:
    apk = DESCARGAS / "OjoScan.apk"
    if not apk.exists():
        raise HTTPException(404, "La app aún no está publicada")
    fecha = datetime.fromtimestamp(apk.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
    mb = apk.stat().st_size / 1e6
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>OjoScan</title>
<style>body{{font-family:system-ui,sans-serif;background:#0f1a1a;color:#e8f4f3;margin:0;padding:24px;
max-width:520px;margin:auto}}a.b{{display:block;background:#0e8c8c;color:#fff;text-align:center;padding:18px;
border-radius:14px;font-size:20px;font-weight:700;text-decoration:none;margin:24px 0}}li{{margin:8px 0}}
small{{color:#9bb}}</style></head><body><h1>👁 OjoScan</h1>
<p>Foto guiada del ojo con análisis por IA.</p>
<a class="b" href="/app/OjoScan.apk">Descargar app ({mb:.0f} MB)</a>
<ol><li>Toca el botón y espera la descarga.</li><li>Abre el archivo descargado.</li>
<li>Si Android lo pide, permite <b>instalar apps de esta fuente</b>.</li>
<li>Toca <b>Actualizar</b> o <b>Instalar</b>.</li></ol>
<small>Versión del {fecha}. Herramienta de investigación, no es un diagnóstico médico.</small></body></html>"""


@app.get("/app/OjoScan.apk")
def descargar_apk() -> FileResponse:
    apk = DESCARGAS / "OjoScan.apk"
    if not apk.exists():
        raise HTTPException(404, "La app aún no está publicada")
    return FileResponse(apk, media_type="application/vnd.android.package-archive", filename="OjoScan.apk")


# --------------------------------------------------------------------------- #
# Lado del teléfono
# --------------------------------------------------------------------------- #
@app.get("/")
def salud() -> dict[str, Any]:
    pc = E.pc_conectada()
    return {
        "status": "ok",
        "version": E.pc_salud.get("version", VERSION) if pc else VERSION,
        "relay": VERSION,
        "pc_conectada": pc,
        "models": E.pc_salud.get("models", {}) if pc else {},
        "capturas_pendientes": len(list(PENDIENTES.glob("*.json"))),
    }


async def leer_formulario(request: Request) -> tuple[dict[str, str], list[dict[str, str]]]:
    if int(request.headers.get("content-length") or 0) > MAX_BYTES:
        raise HTTPException(413, "La imagen es demasiado grande")
    form = await request.form()
    campos: dict[str, str] = {}
    archivos: list[dict[str, str]] = []
    for k, v in form.multi_items():
        if isinstance(v, UploadFile):
            datos = await v.read()
            if not datos:
                raise HTTPException(400, "Archivo vacío")
            archivos.append({
                "campo": k, "nombre": v.filename or "imagen.jpg",
                "tipo": v.content_type or "application/octet-stream",
                "b64": base64.b64encode(datos).decode(),
            })
        else:
            campos[k] = str(v)
    if not archivos:
        raise HTTPException(400, "Falta la imagen (campo 'file')")
    return campos, archivos


@app.post("/predict/{resto:path}")
async def predecir(resto: str, request: Request) -> Response:
    if not E.pc_conectada():
        return error(502, "La computadora con la IA está apagada o sin internet. Intenta más tarde.")
    campos, archivos = await leer_formulario(request)
    tid = uuid.uuid4().hex
    E.trabajos[tid] = {
        "id": tid, "metodo": "POST", "ruta": f"/predict/{resto}", "query": request.url.query,
        "campos": campos, "archivos": archivos,
    }
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    E.esperas[tid] = fut
    await E.cola.put(tid)
    try:
        r = await asyncio.wait_for(fut, ESPERA_PREDICCION)
    except asyncio.TimeoutError:
        return error(504, "La computadora con la IA tardó demasiado en responder. Intenta de nuevo.")
    finally:
        E.trabajos.pop(tid, None)
        E.esperas.pop(tid, None)
    return Response(r["cuerpo"], status_code=r["estado"], media_type=r.get("tipo") or "application/json")


@app.post("/captures/eye")
async def capturar(request: Request) -> dict[str, str]:
    campos, archivos = await leer_formulario(request)
    try:
        meta = json.loads(campos.get("meta", ""))
    except ValueError:
        raise HTTPException(400, "Metadatos inválidos") from None
    if meta.get("consentimiento") is not True:
        raise HTTPException(400, "Falta el consentimiento para guardar la foto")
    cid = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    trabajo = {"id": cid, "metodo": "POST", "ruta": "/captures/eye", "query": "",
               "campos": campos, "archivos": archivos}
    tmp = PENDIENTES / f"{cid}.json.tmp"
    tmp.write_text(json.dumps(trabajo))
    tmp.rename(PENDIENTES / f"{cid}.json")
    await E.cola.put(cid)
    log.info("Captura recibida %s (PC %s)", cid, "conectada" if E.pc_conectada() else "desconectada")
    return {"id": cid, "image": "en camino a la PC" if E.pc_conectada() else "guardada; llegará a la PC al conectarse"}


# --------------------------------------------------------------------------- #
# Lado de la PC (worker)
# --------------------------------------------------------------------------- #
@app.get("/worker/siguiente")
async def siguiente(request: Request, espera: float = 25, authorization: str | None = Header(None)) -> Response:
    autorizar(authorization)
    E.pc_visto = time.monotonic()
    fin = time.monotonic() + min(max(espera, 0), 30)
    while True:
        resto = fin - time.monotonic()
        if resto <= 0:
            E.pc_visto = time.monotonic()
            return Response(status_code=204)
        try:
            tid = await asyncio.wait_for(E.cola.get(), min(resto, 2))
        except asyncio.TimeoutError:
            if await request.is_disconnected():  # worker que se fue (reinicio, cambio de red)
                return Response(status_code=204)
            continue
        if await request.is_disconnected():
            # Esta petición es de un worker que ya no existe: el trabajo vuelve a la cola para que
            # lo tome la conexión viva.
            E.cola.put_nowait(tid)
            return Response(status_code=204)
        E.pc_visto = time.monotonic()
        if tid in E.trabajos:                    # predicción que el teléfono sigue esperando
            return JSONResponse(E.trabajos[tid])
        f = PENDIENTES / f"{tid}.json"
        if f.exists():                           # captura guardada en disco
            E.entregando[tid] = time.monotonic()
            return Response(f.read_text(), media_type="application/json")
        # si no, era una predicción que el teléfono ya dejó de esperar: se descarta


@app.post("/worker/resultado/{tid}")
async def resultado(tid: str, request: Request, authorization: str | None = Header(None)) -> dict[str, bool]:
    autorizar(authorization)
    E.pc_visto = time.monotonic()
    r = await request.json()
    fut = E.esperas.get(tid)
    if fut is not None and not fut.done():
        fut.set_result(r)
        return {"ok": True}
    f = PENDIENTES / f"{tid}.json"
    if f.exists():
        E.entregando.pop(tid, None)
        if 200 <= int(r.get("estado", 500)) < 300:
            shutil.move(str(f), ENTREGADAS / f.name)
            log.info("Captura %s guardada en la PC", tid)
        elif int(r.get("estado", 500)) in (400, 413, 422):
            shutil.move(str(f), ENTREGADAS / f"{tid}.rechazada.json")
            log.warning("La PC rechazó la captura %s: %s", tid, r.get("cuerpo", "")[:200])
        else:
            await E.cola.put(tid)                # error temporal: se reintenta
    return {"ok": True}


@app.post("/worker/estado")
async def estado(request: Request, authorization: str | None = Header(None)) -> dict[str, bool]:
    autorizar(authorization)
    E.pc_visto = time.monotonic()
    E.pc_salud = await request.json()
    return {"ok": True}
