"""Worker de la PC: conecta la IA local con el intermediario del VPS (relay/app.py).

Pide trabajos al relay (long-polling, conexión saliente: no hay que abrir puertos), los manda a
la API local (``src/deployment/api.py``) y devuelve la respuesta.

    RELAY_URL=https://2.25.68.28:13443 RELAY_TOKEN=... LOCAL_API=http://127.0.0.1:8000 \
        python -m src.deployment.relay_worker
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time

import requests

RELAY = os.environ.get("RELAY_URL", "https://2.25.68.28:13443").rstrip("/")
TOKEN = os.environ["RELAY_TOKEN"]
LOCAL = os.environ.get("LOCAL_API", "http://127.0.0.1:8000").rstrip("/")
ESTADO_CADA_S = 30
ATASCO_S = 240  # sin avanzar este tiempo (reloj de pared: cuenta suspensiones) → reinicio

_ultimo_avance = time.time()


def _vigilante() -> None:
    """Si el bucle se atora (conexión muerta tras cambiar de red o suspender la laptop), sale con
    error y systemd (Restart=always) lo levanta de nuevo con conexiones nuevas."""
    while True:
        time.sleep(10)
        if time.time() - _ultimo_avance > ATASCO_S:
            log.error("Sin avanzar %.0f s (¿cambio de red o suspensión?): reinicio", time.time() - _ultimo_avance)
            os._exit(3)

log = logging.getLogger("relay-worker")


def salud_local(s: requests.Session) -> dict:
    try:
        return s.get(f"{LOCAL}/", timeout=10).json()
    except requests.RequestException:
        return {"status": "sin_api_local", "models": {}}


def ejecutar(s: requests.Session, t: dict) -> dict:
    """Reproduce la petición del teléfono contra la API local."""
    url = LOCAL + t["ruta"] + (f"?{t['query']}" if t.get("query") else "")
    files = [
        (a["campo"], (a["nombre"], base64.b64decode(a["b64"]), a["tipo"])) for a in t.get("archivos", [])
    ]
    try:
        r = s.request(t.get("metodo", "POST"), url, data=t.get("campos") or {}, files=files, timeout=180)
        return {"estado": r.status_code, "cuerpo": r.text, "tipo": r.headers.get("content-type", "application/json")}
    except requests.RequestException as exc:
        log.error("La API local no respondió: %s", exc)
        return {
            "estado": 502,
            "cuerpo": '{"error":{"code":502,"message":"La IA de la computadora no está encendida."}}',
            "tipo": "application/json",
        }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    relay = requests.Session()
    relay.headers["Authorization"] = f"Bearer {TOKEN}"
    local = requests.Session()
    ultimo_estado = 0.0
    pausa = 1.0
    log.info("Conectando %s → %s", RELAY, LOCAL)
    global _ultimo_avance
    threading.Thread(target=_vigilante, daemon=True).start()
    while True:
        _ultimo_avance = time.time()
        try:
            if time.monotonic() - ultimo_estado > ESTADO_CADA_S:
                relay.post(f"{RELAY}/worker/estado", json=salud_local(local), timeout=15).raise_for_status()
                ultimo_estado = time.monotonic()
            r = relay.get(f"{RELAY}/worker/siguiente", params={"espera": 25}, timeout=40)
            r.raise_for_status()
            pausa = 1.0
            if r.status_code == 204:
                continue
            t = r.json()
            inicio = time.monotonic()
            res = ejecutar(local, t)
            relay.post(f"{RELAY}/worker/resultado/{t['id']}", json=res, timeout=30).raise_for_status()
            log.info("%s %s → %s (%.1f s)", t["ruta"], t["id"], res["estado"], time.monotonic() - inicio)
        except requests.RequestException as exc:
            log.warning("Sin conexión con el relay (%s); reintento en %.0f s", exc, pausa)
            time.sleep(pausa)
            pausa = min(pausa * 2, 30)


if __name__ == "__main__":
    main()
