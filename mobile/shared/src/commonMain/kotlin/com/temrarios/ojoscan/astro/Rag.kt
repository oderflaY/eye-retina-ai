package com.temrarios.ojoscan.astro

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.math.ln

/** Una pregunta-respuesta de primeros auxilios (traducida al español y filtrada por seguridad). */
data class DocAuxilio(val id: String, val fuente: String, val tema: String, val pregunta: String, val respuesta: String) {
    /** Curado a mano o de fuente oficial (no sintético). */
    val curado get() = fuente == "b" || fuente == "p" || fuente == "m"
    /** Qué hacer paso a paso (primeros auxilios) frente a información general (MedlinePlus). */
    val esPrimerosAuxilios get() = fuente != "m"
    val etiquetaFuente get() = when (fuente) {
        "p" -> "AstroPulse (guías IMSS / Secretaría de Salud)"; "b" -> "First-Aid-Dataset (curado)"
        "m" -> "MedlinePlus, Biblioteca Nacional de Medicina de EE. UU."; else -> "FirstAidQA (sintético, filtrado)"
    }
}

data class Recuperado(val doc: DocAuxilio, val puntaje: Double)

/**
 * RAG sin conexión: recupera procedimientos de primeros auxilios con BM25 sobre texto en español
 * (raíces de palabra + sinónimos coloquiales). No genera texto: muestra las respuestas tal cual vienen
 * de las fuentes, con su origen, para no inventar indicaciones médicas.
 *
 * Base: assets/primeros_auxilios.json — src/rag/construir_primeros_auxilios.py
 * (First-Aid-Dataset curado + FirstAidQA sintético filtrado; léxico de síntomas de HUMADEX).
 */
class BaseAuxilios(val docs: List<DocAuxilio>, val temas: Map<String, String>, val lexico: Set<String>, val aviso: String) {
    private val k1 = 1.4
    private val b = 0.72
    private val terminos: List<Map<String, Int>>
    private val largo: IntArray
    private val largoMedio: Double
    private val df = HashMap<String, Int>()
    private val indice = HashMap<String, MutableList<Int>>()

    init {
        terminos = docs.map { d ->
            // la pregunta cuenta doble: describe la situación con las palabras con que la buscaría alguien
            val toks = Texto.raices(d.pregunta) + Texto.raices(d.pregunta) + Texto.raices(d.respuesta)
            toks.groupingBy { it }.eachCount()
        }
        largo = IntArray(docs.size) { i -> terminos[i].values.sum() }
        largoMedio = largo.average().coerceAtLeast(1.0)
        terminos.forEachIndexed { i, m ->
            for (t in m.keys) { df[t] = (df[t] ?: 0) + 1; indice.getOrPut(t) { mutableListOf() }.add(i) }
        }
    }

    fun buscar(consulta: String, k: Int = 3, minimo: Double = 4.0): List<Recuperado> {
        val q = Texto.raices(Sinonimos.expandir(consulta)).distinct()
        if (q.isEmpty()) return emptyList()
        val n = docs.size.toDouble()
        val puntos = HashMap<Int, Double>()
        for (t in q) {
            val lista = indice[t] ?: continue
            val idf = ln(1 + (n - lista.size + 0.5) / (lista.size + 0.5))
            for (i in lista) {
                val f = terminos[i][t]!!.toDouble()
                puntos[i] = (puntos[i] ?: 0.0) + idf * f * (k1 + 1) / (f + k1 * (1 - b + b * largo[i] / largoMedio))
            }
        }
        val elegidos = mutableListOf<Recuperado>()
        val vistos = mutableListOf<Set<String>>()
        for ((i, p) in puntos.entries.sortedByDescending { (i, p) -> p * if (docs[i].fuente == "b" || docs[i].fuente == "p") 1.15 else 1.0 }) {
            if (p < minimo) break
            // evita mostrar tres respuestas casi iguales
            val toks = terminos[i].keys
            if (vistos.any { v -> (v intersect toks).size > 0.7 * minOf(v.size, toks.size) }) continue
            elegidos += Recuperado(docs[i], p); vistos += toks
            if (elegidos.size == k) break
        }
        return elegidos
    }

    /** Para el diario: hasta 2 procedimientos de qué hacer + 1 resumen de MedlinePlus para saber más. */
    fun paraDiario(consulta: String, texto: String = consulta): List<Recuperado> {
        val r = buscar(consulta, 10)
        val mas = saberMas(texto) ?: r.firstOrNull { !it.doc.esPrimerosAuxilios }
        return r.filter { it.doc.esPrimerosAuxilios }.take(2) + listOfNotNull(mas)
    }

    // Títulos de MedlinePlus: «Presión arterial alta (Alta presión, Hipertensión)» → variantes con sus raíces
    private val titulos: List<Pair<Int, List<Set<String>>>> by lazy {
        docs.indices.filter { docs[it].fuente == "m" && docs[it].id.endsWith("_0") }.map { i ->
            val p = docs[i].pregunta
            val principal = p.substringBefore(" (")
            val alias = if ("(" in p) p.substringAfter("(").removeSuffix(")").split(", ") else emptyList()
            i to (listOf(principal) + alias).map { Texto.raices(it).toSet() }.filter { it.isNotEmpty() }
        }
    }

    /** El tema de MedlinePlus cuyo título (o nombre alterno) aparece completo en lo que escribió la persona. */
    fun saberMas(texto: String): Recuperado? = saberMas(texto, false) ?: saberMas(texto, true)

    private fun saberMas(texto: String, sinonimos: Boolean): Recuperado? {
        val q = Texto.raices(if (sinonimos) Sinonimos.expandir(texto) else texto).toSet()
        if (q.isEmpty()) return null
        val bm = buscar(texto, 40, minimo = 0.0).associate { it.doc.id to it.puntaje }
        return titulos.mapNotNull { (i, variantes) ->
            val cubre = variantes.filter { q.containsAll(it) }.maxOfOrNull { it.size } ?: return@mapNotNull null
            Recuperado(docs[i], cubre * 10.0 + (bm[docs[i].id] ?: 0.0))
        }.maxByOrNull { it.puntaje }
    }

    companion object {
        fun desdeJson(texto: String): BaseAuxilios {
            val o = Json.parseToJsonElement(texto).jsonObject
            val docs = o["docs"]!!.jsonArray.map { e ->
                val d = e.jsonObject
                DocAuxilio(d.s("id"), d.s("f"), d.s("t"), d.s("p"), d.s("r"))
            }
            val temas = o["temas"]?.jsonObject?.mapValues { it.value.jsonPrimitive.content }.orEmpty()
            val lexico = o["lexico"]?.jsonArray?.map { it.jsonPrimitive.content }?.toSet().orEmpty()
            return BaseAuxilios(docs, temas, lexico, o["aviso"]?.jsonPrimitive?.content.orEmpty())
        }

        private fun JsonObject.s(k: String) = this[k]?.jsonPrimitive?.content.orEmpty()
    }
}

/** Normalización y raíces de palabras en español (stemmer ligero, igual para la base y la consulta). */
object Texto {
    private val VACIAS = """a al algo algun alguna ante antes aqui asi aun bien cada como con contra cual cuando de del desde
        donde dos el ella ellas ellos en entre era eres es esa ese eso esta estan estar este esto estos estoy fue ha hace hacer
        han hasta hay la las le les lo los mas me mi mis mucho muy nada ni no nos o otra otro para pero poco por porque puede
        pueden puedo que se segun ser si sin sobre solo son su sus tambien tan te tengo tiene tienen todo tras tu tus un una
        uno unos unas usted y ya debe deben deberia hoy ayer siento sentia estaba estado haber he hemos""".split(Regex("\\s+")).toSet()

    private val SUFIJOS = listOf("amientos", "imientos", "amiento", "imiento", "aciones", "iciones", "uciones", "acion", "icion",
        "ucion", "mente", "ancias", "encias", "ancia", "encia", "adoras", "adores", "adora", "ador", "ables", "ibles", "able",
        "ible", "istas", "ista", "osos", "osas", "oso", "osa", "ando", "iendo", "ados", "idas", "idos", "adas", "ado", "ido",
        "ada", "ida", "aron", "ieron", "aba", "ian", "ar", "er", "ir", "es", "as", "os", "a", "o", "e", "s")

    fun palabras(t: String): List<String> = normalizar(t).split(Regex("[^a-z0-9]+")).filter { it.isNotEmpty() }

    fun raiz(p: String): String {
        if (p.length <= 4) return p
        for (s in SUFIJOS) if (p.endsWith(s) && p.length - s.length >= 4) return p.dropLast(s.length)
        return p
    }

    // Formas de «doler» → «dolor» (si no, «duele» y «duelo» comparten raíz)
    private val LEMAS = mapOf("duele" to "dolor", "duelen" to "dolor", "dolia" to "dolor", "dolio" to "dolor", "doliendo" to "dolor",
        "dolores" to "dolor", "adolorido" to "dolor", "adolorida" to "dolor", "dolorido" to "dolor")

    fun raices(t: String): List<String> = palabras(t).filter { it !in VACIAS && it.length > 1 }.map { LEMAS[it] ?: raiz(it) }
}

/** Cómo lo dice la gente → cómo lo dicen los manuales. Se AÑADE a la consulta, no la reemplaza. */
object Sinonimos {
    private val MAPA = listOf(
        "me corte" to "corte herida sangrado", "cortada" to "corte herida", "sangra" to "sangrado hemorragia",
        "me queme" to "quemadura", "quemada" to "quemadura", "ampolla" to "ampolla quemadura",
        "me pico" to "picadura", "piquete" to "picadura", "alacran" to "escorpion picadura", "vibora" to "serpiente mordedura",
        "me mordio" to "mordedura", "perro" to "mordedura animal",
        "atragant" to "atragantamiento asfixia vias respiratorias", "se trago" to "tragar objeto atragantamiento",
        "no puede respirar" to "respiracion via aerea", "falta de aire" to "dificultad respirar asma", "me ahogo" to "dificultad respirar",
        "dolor de pecho" to "dolor pecho ataque cardiaco infarto", "dolor en el pecho" to "dolor pecho ataque cardiaco infarto",
        "opresion" to "ataque cardiaco", "brazo izquierdo" to "ataque cardiaco",
        "rasp" to "raspadura abrasion herida", "temblando" to "azucar baja hipoglucemia", "sudando" to "sudoracion", "infarto" to "ataque cardiaco", "paro" to "paro cardiaco rcp",
        "desmay" to "desmayo inconsciente", "se cayo" to "caida lesion", "me cai" to "caida lesion", "chichon" to "golpe cabeza",
        "golpe en la cabeza" to "lesion cabeza conmocion", "torcedura" to "esguince", "me torci" to "esguince tobillo",
        "hueso" to "fractura", "roto" to "fractura", "quebr" to "fractura",
        "calentura" to "fiebre", "temperatura" to "fiebre", "vomit" to "vomito", "guacar" to "vomito",
        "chorro" to "diarrea", "deshidrat" to "deshidratacion", "insolacion" to "golpe calor", "mucho calor" to "golpe calor",
        "mucho frio" to "hipotermia", "congel" to "congelacion",
        "ronchas" to "urticaria alergia", "hinch" to "hinchazon alergia", "alergia" to "reaccion alergica anafilaxia",
        "convuls" to "convulsion", "ataque epilep" to "convulsion epilepsia",
        "intoxic" to "envenenamiento", "me envenen" to "envenenamiento veneno", "cloro" to "producto quimico envenenamiento",
        "pastillas de mas" to "sobredosis", "sobredosis" to "sobredosis envenenamiento",
        "azucar baja" to "hipoglucemia diabetes", "diabet" to "diabetes azucar sangre", "glucosa" to "azucar sangre diabetes",
        "nariz" to "nariz sangrado nasal", "ojo" to "ojo lesion", "diente" to "diente dental",
        "toque" to "descarga electrica", "electrocut" to "descarga electrica", "ahogado" to "ahogamiento agua",
        "cara chueca" to "derrame cerebral accidente cerebrovascular", "no puedo hablar" to "derrame cerebral",
        "ansiedad" to "ataque panico ansiedad", "no puedo dormir" to "insomnio problemas sueno", "no duermo" to "insomnio",
        "presion alta" to "presion arterial alta hipertension", "presion baja" to "presion arterial baja hipotension",
        "gripa" to "gripe influenza resfrio", "resfriado" to "resfrio", "mocos" to "resfrio", "panza" to "dolor abdominal",
        "estomago" to "dolor abdominal estomago", "agruras" to "acidez estomacal", "estrenid" to "estrenimiento", "panico" to "ataque panico", "bebe" to "bebe lactante nino",
    )

    fun expandir(t: String): String {
        val n = normalizar(t)
        val extra = MAPA.filter { (k, _) -> k in n }.joinToString(" ") { it.second }
        return if (extra.isEmpty()) t else "$t $extra"
    }
}
