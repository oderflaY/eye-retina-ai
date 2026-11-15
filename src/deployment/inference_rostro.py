"""Análisis del rostro que complementa al del ojo.

Pide las medidas al servicio local de MediaPipe (``servicio_rostro``, 127.0.0.1:8001), aplica los
modelos entrenados y recorta los dos ojos de la misma foto para pasarlos al analizador de ojo
externo. El resultado cruza ambos: p. ej. parálisis + un ojo que no cierra, o ictericia +
esclerótica amarilla.

Tamizaje, no diagnóstico. Asimetría súbita de la cara = posible infarto cerebral → urgencias.
"""

from __future__ import annotations

import json
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2
import joblib
import numpy as np

SERVICIO = os.environ.get("ROSTRO_SERVICIO", "http://127.0.0.1:8001")
YAW_MAX = 25.0
ZONA_PARALISIS = (0.35, 0.65)
ZONA_ICTERICIA = (0.35, 0.65)

AVISO = ("Tamizaje orientativo hecho con pocos datos públicos; no es diagnóstico. Si la cara se te "
         "desvió de repente, se te cae un lado de la boca, no puedes cerrar un ojo, o hay debilidad en un "
         "brazo o dificultad para hablar, llama a emergencias (911) YA: puede ser un infarto cerebral.")


def _llamar_servicio(jpeg: bytes) -> dict:
    limite = uuid.uuid4().hex
    cuerpo = (f"--{limite}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"cara.jpg\"\r\n"
              f"Content-Type: image/jpeg\r\n\r\n").encode() + jpeg + f"\r\n--{limite}--\r\n".encode()
    req = urllib.request.Request(f"{SERVICIO}/rasgos", data=cuerpo, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={limite}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _veredicto(p: float, zona: tuple[float, float]) -> str:
    return "posible" if p >= zona[1] else ("no_concluyente" if p > zona[0] else "sin_signos")


class AnalizadorRostro:
    def __init__(self, carpeta: Path, analizador_ojo: Callable[[], Any] | None = None) -> None:
        self.paralisis = joblib.load(carpeta / "paralisis.joblib")
        self.ictericia = joblib.load(carpeta / "ictericia.joblib")
        self._ojo = analizador_ojo

    def _recortar_ojo(self, rgb: np.ndarray, caja: list[float]) -> np.ndarray | None:
        h, w = rgb.shape[:2]
        x, y, bw, bh = caja[0] * w, caja[1] * h, caja[2] * w, caja[3] * h
        cx, cy, lado = x + bw / 2, y + bh / 2, max(bw * 2.0, 24)  # como el recorte de OjoScan (2× ancho)
        x0, y0 = int(max(0, cx - lado / 2)), int(max(0, cy - lado / 2))
        x1, y1 = int(min(w, cx + lado / 2)), int(min(h, cy + lado / 2))
        return rgb[y0:y1, x0:x1] if x1 - x0 > 16 and y1 - y0 > 16 else None

    def predict(self, rgb: np.ndarray) -> dict[str, Any]:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        try:
            m = _llamar_servicio(buf.tobytes())
        except Exception as exc:  # servicio caído
            return {"cara": False, "error": f"servicio de rostro no disponible: {exc}"}
        if not m.get("cara"):
            return {"cara": False, "mensaje": "No encontré una cara completa. Toma la foto de frente, con toda la cara en el cuadro."}

        asim, color = m["asimetria"], m.get("color")
        avisos_foto = []
        if m.get("caras", 1) > 1:
            avisos_foto.append("Hay más de una cara; analicé la más clara.")
        de_frente = abs(asim["yaw"]) <= YAW_MAX
        if not de_frente:
            avisos_foto.append("La cara está de lado: la asimetría no se puede medir bien.")

        hallazgos = []
        # Parálisis facial (asimetría)
        cols = self.paralisis["caracteristicas"]
        p_par = float(self.paralisis["modelo"].predict_proba(np.array([[asim[c] for c in cols]]))[0, 1])
        v = self.paralisis["validacion"]
        hallazgos.append({
            "clave": "paralisis_facial", "nombre": "Asimetría / parálisis facial",
            "probabilidad": round(p_par, 4), "resultado": _veredicto(p_par, ZONA_PARALISIS) if de_frente else "no_concluyente",
            "detalle": {"comisuras": round(asim["geo_comisura"], 4), "parpados": round(asim["geo_ojo"], 4),
                        "cejas": round(asim["geo_ceja"], 4)},
            "validacion": {"auc_por_persona": v["persona"]["auc"], "sensibilidad": v["persona"]["sens"],
                           "especificidad": v["persona"]["esp"], "falsas_alarmas_caras_sanas": v["ffhq_falsas_alarmas"],
                           "personas": v["n_personas"]},
        })
        # Ictericia (blanco del ojo amarillo)
        if color:
            cols = self.ictericia["caracteristicas"]
            p_ic = float(self.ictericia["modelo"].predict_proba(np.array([[color[c] for c in cols]]))[0, 1])
            v = self.ictericia["validacion"]
            hallazgos.append({
                "clave": "ictericia", "nombre": "Ictericia (ojos amarillos)",
                "probabilidad": round(p_ic, 4), "resultado": _veredicto(p_ic, ZONA_ICTERICIA),
                "detalle": {"amarillo_esclerotica_b": round(color["esp_b"], 2),
                            "ojo_vs_piel": round(color["esp_b_menos_piel"], 2)},
                "validacion": {"auc": v["auc_cv"], "sensibilidad": v["sensibilidad"], "especificidad": v["especificidad"],
                               "falsas_alarmas_caras_sanas": v["falsas_alarmas_ffhq"], "casos": v["n_ictericia"]},
            })

        # Ojos de la misma foto → analizador de ojo externo
        ojos = {}
        if self._ojo is not None:
            try:
                analizador = self._ojo()
                for lado, caja in m["ojos"].items():
                    rec = self._recortar_ojo(rgb, caja)
                    if rec is not None:
                        r = analizador.predict(rec)
                        ojos[lado] = {"principal": r["principal"], "titulo": r["titulo"], "confianza": r["confianza"]}
            except Exception:  # sin modelo de ojo: el rostro sigue solo
                ojos = {}

        return {"cara": True, "de_frente": de_frente, "avisos_foto": avisos_foto, "hallazgos": hallazgos,
                "ojos": ojos, "combinado": self._combinar(hallazgos, ojos, asim), "caja_cara": m["caja_cara"],
                "aviso": AVISO}

    @staticmethod
    def _combinar(hallazgos: list[dict], ojos: dict, asim: dict) -> list[dict]:
        """Lecturas cruzadas rostro + ojo (lo que ninguno ve solo)."""
        h = {x["clave"]: x for x in hallazgos}
        notas = []
        par = h.get("paralisis_facial")
        if par and par["resultado"] == "posible":
            urg = "alta" if asim["geo_ojo"] > 0.25 else "media"
            notas.append({"nivel": urg, "texto": "Un lado de la cara se mueve menos que el otro"
                          + (" y un párpado cierra distinto" if asim["geo_ojo"] > 0.25 else "")
                          + ". Si empezó de golpe, ve a urgencias YA (descartar infarto cerebral). "
                            "En parálisis de Bell el ojo que no cierra se reseca: lubrícalo y protégelo."})
        ic = h.get("ictericia")
        if ic and ic["resultado"] == "posible":
            notas.append({"nivel": "alta", "texto": "El blanco del ojo se ve amarillo. Puede ser ictericia (hígado, "
                          "vesícula, sangre). Revisión médica y análisis de bilirrubina pronto."})
        rojos = [l for l, o in ojos.items() if o["principal"] == "enrojecimiento"]
        if rojos:
            notas.append({"nivel": "media", "texto": f"Ojo {' y '.join(rojos)} enrojecido en la foto de la cara. "
                          "Para confirmar usa el modo Ojo de cerca."})
        if par and par["resultado"] == "posible" and rojos:
            notas.append({"nivel": "media", "texto": "Asimetría + ojo rojo: un ojo que no cierra bien se irrita. "
                          "Conviene revisión con oftalmología."})
        if not notas:
            notas.append({"nivel": "ok", "texto": "Sin señales en cara ni en ojos con esta foto."})
        return notas
