"""Construye la base de conocimiento de primeros auxilios (español) para el RAG del Diario de AstroPulse.

Fuentes (en ~/TemrariosSpace/datasets/PrimerosAuxilios):
  * belvisk/First-Aid-Dataset (MIT)      — 346 preguntas/respuestas curadas a mano (Mayo, Cruz Roja India, …)
  * i-am-mushfiq/FirstAidQA (CC BY 4.0)  — 5 550 pares SINTÉTICOS (GPT-4o-mini); su README avisa de
                                           respuestas inseguras → se filtran con reglas de seguridad.
  * HUMADEX/spanish_ner_dataset (Apache) — solo para el léxico de síntomas (palabras etiquetadas PROBLEM).
  * MedlinePlus en español (NLM, dominio público; mplus_topics_*.xml) — 1 016 resúmenes de temas de salud
    escritos en español por la Biblioteca Nacional de Medicina. Solo <full-summary> (la enciclopedia A.D.A.M.
    y las fichas de fármacos enlazadas tienen copyright y NO se usan). Cita: «Fuente: MedlinePlus, NLM».

La traducción inglés→español es local (CTranslate2 + modelo OPUS de Argos, CPU int8) y se guarda en caché
(traducciones.jsonl) para poder reanudar.

Uso:  .venv-rag/bin/python src/rag/construir_primeros_auxilios.py
Salida: data/processed/primeros_auxilios/{base.json, lexico_humadex.json, reporte.json}
        mobile/androidApp/src/main/assets/primeros_auxilios.json (lo que lee la app)
"""
from __future__ import annotations

import collections
import csv
import json
import re
import time
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = Path.home() / "TemrariosSpace/datasets/PrimerosAuxilios"
MODELO = DATOS / "modelo_en_es/en_es"
SALIDA = RAIZ / "data/processed/primeros_auxilios"
ASSET = RAIZ / "mobile/androidApp/src/main/assets/primeros_auxilios.json"
CACHE = SALIDA / "traducciones.jsonl"

# ── 1. Carga ────────────────────────────────────────────────────────────────
TYPOS = {"cardicac": "cardiac", "chocking": "choking", "unconcious": "unconscious", "bleding": "bleeding",
         "poisioning": "poisoning", "frist": "first", "brusies": "bruises", "spraind": "sprained"}


def limpiar_en(s: str) -> str:
    s = s.replace("’", "'").replace("“", '"').replace("”", '"').replace(" ", " ")
    for a, b in TYPOS.items():
        s = re.sub(a, b, s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def cargar() -> list[dict]:
    docs = []
    with open(DATOS / "belvisk/merged_data.csv", newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            q, a = limpiar_en(r.get("question") or ""), limpiar_en(r.get("answer") or "")
            if len(q) > 8 and len(a) > 40:  # descarta respuestas cortadas («perform the thrusts…»)
                docs.append({"id": f"b{i}", "fuente": "belvisk", "q": q, "a": a})
    for i, r in enumerate(json.load(open(DATOS / "firstaidqa/firstaidqa_v1.json", encoding="utf-8"))):
        q, a = limpiar_en(r.get("question") or ""), limpiar_en(r.get("answer") or "")
        if len(q) > 8 and len(a) > 40:
            docs.append({"id": f"f{i}", "fuente": "firstaidqa", "q": q, "a": a})
    # duplicados exactos de pregunta: se queda el curado a mano
    vistos, unicos = set(), []
    for d in docs:
        k = re.sub(r"[^a-z ]", "", d["q"].lower())
        if k not in vistos:
            vistos.add(k)
            unicos.append(d)
    return unicos


def _html_a_texto(h: str) -> list[str]:
    """Párrafos de texto plano; las viñetas quedan como «• …»."""
    import html as _html
    h = re.sub(r"<li[^>]*>", "\n• ", h)
    h = re.sub(r"</(p|ul|ol|h\d)>", "\n\n", h)
    h = re.sub(r"<[^>]+>", "", h)
    h = _html.unescape(h).replace("\u00a0", " ")
    bloques = [re.sub(r"[ \t]+", " ", b).strip() for b in re.split(r"\n\s*\n", h)]
    bloques = [re.sub(r"\s*\n\s*", "\n", b) for b in bloques if len(b) > 2]
    unidos: list[str] = []
    for b in bloques:  # «Las causas incluyen:» + su lista = un solo bloque
        if unidos and (unidos[-1].endswith(":") or b.startswith("•")):
            unidos[-1] += "\n" + b
        else:
            unidos.append(b)
    return unidos


def cargar_medlineplus(max_chars: int = 750) -> tuple[list[dict], dict[str, str]]:
    """Temas de salud en español → fragmentos de ~750 caracteres (párrafos completos)."""
    import xml.etree.ElementTree as ET
    xml = sorted((DATOS / "medlineplus").glob("mplus_topics_*.xml"))[-1]
    docs, grupos = [], {}
    for t in ET.parse(xml).getroot().findall("health-topic"):
        if t.get("language") != "Spanish":
            continue
        titulo = t.get("title", "").strip()
        alias = [a.text.strip() for a in t.findall("also-called") if a.text][:4]
        g = t.find("group")
        clave = f"m{g.get('id')}" if g is not None else "general"
        if g is not None:
            grupos[clave] = g.text.strip()
        parrafos = _html_a_texto(t.findtext("full-summary") or "")
        trozo, n = "", 0
        for par in parrafos + [None]:
            if par is None or (trozo and len(trozo) + len(par) > max_chars):
                if len(trozo) > 80:
                    pregunta = titulo + (f" ({', '.join(alias)})" if alias and n == 0 else "")
                    docs.append({"id": f"m{t.get('id')}_{n}", "fuente": "medlineplus", "tema": clave,
                                 "q_es": pregunta, "a_es": trozo.strip(), "url": t.get("url")})
                    n += 1
                trozo = ""
            if par is not None:
                trozo += ("\n" if trozo else "") + par
    return docs, grupos


# ── 2. Filtro de seguridad ──────────────────────────────────────────────────
# Consejos obsoletos o peligrosos que aparecen en material de primeros auxilios generado o viejo.
# Una oración que los recomienda SIN negación («do not», «never», «avoid») descarta el par entero.
PELIGROSOS = {
    "mantequilla/aceite/pasta en quemadura": r"\b(apply|put|rub|spread|use|cover)\w*\b[^.]*\b(butter|toothpaste|grease|egg white|cooking oil)\b",
    "hielo directo en quemadura": r"\b(apply|put|place|use)\w* (ice|an ice pack)\b[^.]*\bburn|\bburn[^.]*\b(apply|put|place)\w* ice\b",
    "provocar el vómito": r"\binduc\w* vomit|\bmake (them|the person|him|her) (vomit|throw up)|\bsyrup of ipecac",
    "succionar/cortar mordedura": r"\b(suck|suction)\w*\b.*\b(venom|poison|bite)|\bcut\b.*\b(bite|wound) to\b|\bincision\b.*\bbite",
    "torniquete en mordedura de serpiente": r"\btourniquet\b.*\bsnake|\bsnake.*\btourniquet",
    "cabeza atrás en sangrado nasal": r"(nose ?bleed|bleeding (from the )?nose)[^.]*\btilt\w* (the|their|your) head back|\btilt\w* (the|their|your) head back[^.]*nose",
    "objeto en la boca en convulsión": r"\b(put|place|insert|force)\w* (something|an object|anything|a spoon|your fingers?|a wallet)\b.*\bmouth",
    "sujetar a quien convulsiona": r"\b(restrain|hold down)\b.*\b(seizure|convuls)",
    "frotar congelación / alcohol": r"\brub\w*\b.*\bfrost|\bfrostbit\w*.*\brub|\b(give|offer)\w* (them |him |her )?alcohol",
    "agua caliente en congelación": r"\bhot water\b.*\bfrost|\bfrost.*\bhot water",
    "torniquete casero / aflojarlo": r"\bloosen\w* the tourniquet",
    "orina en picadura de medusa": r"\burinat\w*\b.*\bjelly|\bjelly.*\burin",
    "sacar objeto clavado": r"\b(remove|pull out)\w* (the )?(impaled|embedded) object",
    "golpes en la espalda a quien tose": r"\bslap\w* (them |him |her )?on the back\b.*\bcough",
}
NEGACION = re.compile(r"\b(not|never|avoid|don'?t|doesn'?t|no longer|instead of|rather than|myth|outdated|harmful|dangerous|"
                      r"without|worsen\w*|can cause|may cause|causing|damage|only if|only be|if advised|unless|whether|"
                      r"increas\w* the risk|misconception)\b", re.I)


def oraciones(texto: str) -> list[str]:
    partes = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", texto)
    return [p.strip() for p in partes if len(p.strip()) > 1]


def inseguro(d: dict) -> str | None:
    for o in oraciones(d["a"]):
        for nombre, patron in PELIGROSOS.items():
            if re.search(patron, o, re.I) and not NEGACION.search(o):
                return nombre
    return None


# ── 3. Temas (para mostrar y para filtrar) ───────────────────────────────────
TEMAS = [  # (clave, nombre, patrón en inglés) — el primero que coincide en la PREGUNTA gana
    ("rcp", "Paro cardiaco / RCP", r"\bcpr\b|cardiac arrest|chest compression|\baed\b|defibrillat|not breathing|no pulse|resuscitat"),
    ("atragantamiento", "Atragantamiento", r"chok|heimlich|abdominal thrust|airway obstruct|swallowed (an|a) object"),
    ("corazon", "Infarto / dolor de pecho", r"heart attack|chest pain|angina|cardiac"),
    ("acv", "Derrame cerebral (ACV)", r"stroke|\bfast\b test|facial droop|slurred"),
    ("hemorragia", "Sangrado y heridas", r"bleed|wound|cut\b|cuts\b|laceration|hemorrhag|amputat|tourniquet|puncture|graze|abrasion"),
    ("quemadura", "Quemaduras", r"burn|scald|blister"),
    ("fractura", "Fracturas, esguinces y golpes", r"fractur|broken bone|sprain|strain|dislocat|splint|sling|bruise|crush"),
    ("cabeza", "Golpe en la cabeza / columna", r"head injur|concussion|skull|spinal|spine|neck injur"),
    ("convulsion", "Convulsiones", r"seizure|convuls|epilep"),
    ("alergia", "Alergia / anafilaxia", r"allerg|anaphyla|epipen|epinephrine|hives|bee sting|wasp"),
    ("respiracion", "Asma y falta de aire", r"asthma|breath|inhaler|hyperventil|wheez"),
    ("envenenamiento", "Envenenamiento y sobredosis", r"poison|overdose|toxic|ingest|swallow|carbon monoxide|fumes|chemical"),
    ("mordedura", "Mordeduras y picaduras", r"bite|sting|snake|spider|scorpion|tick|jellyfish|animal"),
    ("calor", "Golpe de calor / deshidratación", r"heat ?stroke|heat exhaustion|dehydrat|sunburn|heat cramp|hyperthermia"),
    ("frio", "Hipotermia / congelación", r"hypotherm|frostbit|cold"),
    ("ahogamiento", "Ahogamiento", r"drown|water rescue"),
    ("electrico", "Descarga eléctrica", r"electric|lightning|shock from"),
    ("diabetes", "Diabetes / azúcar baja", r"diabet|hypoglyc|blood sugar|insulin|glucose"),
    ("desmayo", "Desmayo / inconsciencia", r"faint|unconscious|unresponsive|recovery position|collapse"),
    ("choque", "Estado de choque", r"\bshock\b"),
    ("ojo", "Ojos", r"\beye"),
    ("nariz", "Nariz / oído / dientes", r"nose|ear\b|ears\b|tooth|teeth|dental"),
    ("parto", "Embarazo y parto", r"pregnan|childbirth|labou?r|deliver"),
    ("bebe", "Bebés y niños", r"infant|baby|child|toddler"),
    ("mental", "Crisis emocional", r"panic|anxiety|suicid|mental|distress"),
]


def tema(d: dict) -> str:
    for texto in (d["q"], d["a"]):
        for clave, _, patron in TEMAS:
            if re.search(patron, texto, re.I):
                return clave
    return "general"


# ── 4. Traducción local ─────────────────────────────────────────────────────
ARREGLOS = [  # (si el inglés contiene, buscar en español, reemplazo)
    (r"chok", r"\bse ahoga\b", "se atraganta"), (r"chok", r"\bahogándose\b", "atragantándose"),
    (r"chok", r"\bahogamiento\b", "atragantamiento"), (r"chok", r"\bahogo\b", "atragantamiento"),
    (r"chok", r"\bahogarse\b", "atragantarse"), (r"chok", r"\bahogada?\b", "atragantada"),
    (r"chok", r"\basfixia\b", "atragantamiento"),
    (r"back blow", r"golpes? (de|en|traseros de) (la )?espalda", "golpes en la espalda"),
    (r"\bCPR\b", r"\bCPR\b", "RCP"), (r"\bAED\b", r"\bAED\b", "DEA"),
    (r"casualty", r"\bbaja\b", "víctima"), (r"casualties", r"\bbajas\b", "víctimas"),
    (r"recovery position", r"posición de recuperación", "posición lateral de seguridad"),
    (r"first aider", r"\b(el |la |un |una )?primer(o|a)? (ayudante|auxiliar|socorrista)", "quien da los primeros auxilios"),
    (r"first aider", r"\b(los |unos )?primeros (ayudantes|auxiliares)", "quienes dan primeros auxilios"),
    (r"seiz", r"\b(incautaciones|embargos|confiscaciones|incautamientos)\b", "convulsiones"),
    (r"seiz", r"(?<![Ss]in )\b(incautación|incautacion|embargo|confiscación|incautamiento)\b", "convulsión"),
    (r"seiz", r"\bgestión del convulsión\b", "manejo de la convulsión"),
    (r"puffer", r"\bpuffer\b", "inhalador"), (r"dislodge", r"\bdesactivar\b", "sacar"), (r"dislodge", r"\bdesalojar\b", "sacar"),
    (r"food poisoning", r"envenenamiento (por|de) (los )?alimentos", "intoxicación alimentaria"),
    (r"thrust", r"\bimpulsos\b", "compresiones"), (r"thrust", r"\bempujes\b", "compresiones"),
    (r"EpiPen|auto-?injector", r"\bEpiPen\b", "autoinyector de epinefrina (EpiPen)"),
    (r"emergency services", r"servicios de emergencia", "servicios de emergencia (911)"),
]


class Traductor:
    def __init__(self):
        import ctranslate2
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(MODELO / "sentencepiece.model"))
        self.tr = ctranslate2.Translator(str(MODELO / "model"), device="cpu", compute_type="int8",
                                         inter_threads=2, intra_threads=4)
        self.cache: dict[str, str] = {}
        if CACHE.exists():
            for linea in CACHE.open(encoding="utf-8"):
                r = json.loads(linea)
                self.cache[r["en"]] = r["es"]

    def traducir(self, frases: list[str], lote: int = 48) -> None:
        faltan = sorted({f for f in frases if f not in self.cache}, key=len)
        print(f"traducir: {len(faltan)} frases nuevas ({len(self.cache)} en caché)", flush=True)
        t0 = time.time()
        with CACHE.open("a", encoding="utf-8") as out:
            for i in range(0, len(faltan), lote):
                bloque = faltan[i:i + lote]
                res = self.tr.translate_batch([self.sp.encode(f, out_type=str) for f in bloque], beam_size=4,
                                              max_decoding_length=256)
                for en, r in zip(bloque, res):
                    es = self.sp.decode(r.hypotheses[0]).strip()
                    self.cache[en] = es
                    out.write(json.dumps({"en": en, "es": es}, ensure_ascii=False) + "\n")
                if (i // lote) % 20 == 0:
                    print(f"  {i + len(bloque)}/{len(faltan)}  {time.time() - t0:.0f} s", flush=True)

    def es(self, texto_en: str) -> str:
        partes = []
        for o in ([texto_en] if texto_en in self.cache else oraciones(texto_en)):
            s = self.cache[o]
            for si_en, buscar, poner in ARREGLOS:
                if re.search(si_en, o, re.I) and not re.search(r"drown", o, re.I):
                    s = re.sub(buscar, poner, s)
            partes.append(s)
        return " ".join(partes)


# ── 5. Léxico de síntomas desde HUMADEX ─────────────────────────────────────
def sin_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn").replace("ñ", "n")


VACIAS = set("""el la los las un una unos unas de del al a en y o u que por para con sin su sus se le les lo me mi
mis te tu es son ser fue era esta este esto estos estas ese esa eso hay muy mas pero como cuando si no sobre entre
tambien puede pueden tener tiene tienen hacer hace otro otra otros otras todo toda todos todas algo alguna algun
usted ustedes ellos ellas nos cual cuales donde porque ya solo puedo estoy tengo debe deberia podria cosa cosas parte
partes tipo tipos causa causas problema problemas condicion condiciones paciente pacientes persona personas medico
medica medicos enfermedad enfermedades sintoma sintomas sintomatico afeccion afecciones caso casos nivel niveles
grave leve severo severa cronico cronica agudo aguda normal anormal alto alta bajo baja""".split())


def lexico_humadex(min_frec: int = 40, min_ratio: float = 0.6) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(DATOS / "humadex/spanish_medical_data.parquet")
    total, prob = collections.Counter(), collections.Counter()
    for frase, tags in zip(df["sentence"], df["tags"]):
        for w, t in zip(frase, tags):
            w = sin_acentos(re.sub(r"[^\wáéíóúüñ]", "", str(w)))
            if len(w) < 4 or w in VACIAS or w.isdigit():
                continue
            total[w] += 1
            if 1 <= int(t) <= 4:  # B/I/E/S-PROBLEM
                prob[w] += 1
    lex = [{"palabra": w, "frec": prob[w], "ratio": round(prob[w] / total[w], 3)}
           for w in prob if prob[w] >= min_frec and prob[w] / total[w] >= min_ratio]
    return sorted(lex, key=lambda x: -x["frec"])


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    docs = cargar()
    descartes = collections.Counter()
    ejemplos: dict[str, str] = {}
    seguros = []
    for d in docs:
        motivo = inseguro(d)
        if motivo:
            descartes[motivo] += 1
            ejemplos.setdefault(motivo, d["a"][:220])
        else:
            d["tema"] = tema(d)
            seguros.append(d)
    print(f"{len(docs)} pares únicos; {len(docs) - len(seguros)} descartados por seguridad: {dict(descartes)}")

    # Procedimientos escritos a mano para huecos importantes en México (la fuente casi no trae alacrán)
    seguros += [{"id": "p1", "fuente": "propio", "tema": "mordedura", "q": "", "a": "",
                 "q_es": "¿Qué hacer si me picó un alacrán (escorpión)?",
                 "a_es": "Mantén la calma y lava la picadura con agua y jabón. Pon frío local (hielo envuelto en un trapo) 10 minutos. "
                         "No cortes, no succiones, no pongas torniquete ni remedios caseros. Ve de inmediato a urgencias (IMSS, ISSSTE "
                         "o centro de salud): existe un antídoto (faboterápico antialacrán) que funciona mejor si se aplica pronto. "
                         "Es URGENTE en niños, adultos mayores, o si hay babeo, inquietud, dificultad para respirar, movimientos "
                         "raros de los ojos o mucha comezón en nariz y garganta: llama al 911."}]
    tr = Traductor()
    tr.traducir([o for d in seguros if d["fuente"] != "propio" for o in [d["q"], *oraciones(d["a"])]])
    # las preguntas son una sola «oración» aunque traigan varios signos
    for d in seguros:
        if d["fuente"] == "propio":
            continue
        q = tr.es(d["q"]).strip()
        if q.startswith("["):  # belvisk trae variantes: ["¿Cómo…?", "¿Qué…?", …] → la primera
            q = re.split(r'"\s*,\s*"', q.strip("[] "))[0]
        d["q_es"] = q.strip('"[] ')
        d["a_es"] = tr.es(d["a"]).strip('"[] ')

    mlp, grupos = cargar_medlineplus()
    print(f"MedlinePlus: {len(mlp)} fragmentos de {len({d['id'].split('_')[0] for d in mlp})} temas")
    seguros += mlp

    lex = lexico_humadex()
    print(f"léxico HUMADEX: {len(lex)} palabras PROBLEM")

    nombres = {c: n for c, n, _ in TEMAS} | {"general": "Primeros auxilios general"} | grupos
    base = {
        "version": time.strftime("%Y-%m-%d"),
        "aviso": "Orientación general de primeros auxilios. No sustituye a un profesional ni al 911.",
        "fuentes": {"belvisk": "First-Aid-Dataset (curado a mano, MIT)",
                    "firstaidqa": "FirstAidQA (sintético, CC BY 4.0, filtrado)",
                    "propio": "Escrito para AstroPulse con guías de IMSS/Secretaría de Salud",
                    "medlineplus": "MedlinePlus, Biblioteca Nacional de Medicina de EE. UU. (dominio público)"},
        "temas": nombres,
        "docs": [{"id": d["id"], "f": d["fuente"][0], "t": d["tema"], "p": d["q_es"], "r": d["a_es"]} for d in seguros],
        "lexico": [x["palabra"] for x in lex],
    }
    json.dump(base, open(SALIDA / "base.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump(lex, open(SALIDA / "lexico_humadex.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    json.dump(base, open(ASSET, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    reporte = {"pares_unicos": len(docs), "seguros": len(seguros), "descartados": dict(descartes),
               "ejemplos_descartados": ejemplos, "por_fuente": collections.Counter(d["fuente"] for d in seguros),
               "por_tema": collections.Counter(d["tema"] for d in seguros), "lexico": len(lex),
               "asset_kb": round(ASSET.stat().st_size / 1024)}
    json.dump(reporte, open(SALIDA / "reporte.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(reporte, ensure_ascii=False, indent=1)[:2500])


if __name__ == "__main__":
    main()
