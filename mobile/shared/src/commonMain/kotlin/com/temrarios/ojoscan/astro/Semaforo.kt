package com.temrarios.ojoscan.astro

import kotlinx.serialization.Serializable

/** Una entrada del diario de salud: lo que escribió la persona y cómo lo clasificó el semáforo. */
@Serializable
data class EntradaDiario(
    val horaMs: Long,
    val texto: String,
    /** 1 = muy mal … 5 = muy bien (opcional). */
    val animo: Int? = null,
    val triage: Triage,
    val motivos: List<String> = emptyList(),
    /** Síntomas reconocidos (clave corta), para ver si se repiten varios días. */
    val sintomas: List<String> = emptyList(),
    val pulso: Int? = null,
    val spo2: Double? = null,
)

data class Evaluacion(
    val triage: Triage,
    val titulo: String,
    val accion: String,
    val motivos: List<String>,
    val sintomas: List<String>,
    /** Lo que se busca en la base de primeros auxilios. */
    val consulta: String,
)

/**
 * Semáforo de salud del diario (reglas fijas, revisables a mano; el RAG no decide el color):
 *  ROJO     — señales de alarma reconocidas en el texto, signos vitales críticos o crisis emocional → 911 / SOS.
 *  AMARILLO — síntomas que hay que atender hoy, ánimo muy bajo, o el mismo síntoma 3 días o más en la semana.
 *  VERDE    — sin síntomas ni alarmas.
 * Las negaciones («no tengo fiebre», «ya no me duele el pecho») no cuentan.
 */
object Semaforo {
    private class Regla(val clave: String, val nombre: String, val patrones: List<Regex>)

    private fun r(clave: String, nombre: String, vararg p: String) = Regla(clave, nombre, p.map { Regex(it) })

    // Señales de alarma (AHA, Cruz Roja, NHS «call 999», IMSS). Texto ya sin acentos y en minúsculas.
    private val ROJAS = listOf(
        r("no_respira", "No respira o respira con mucha dificultad", "no respira", "no puedo respirar", "no puede respirar",
            "me estoy ahogando", "se esta ahogando", "labios? morados?", "se puso morad", "boqueando"),
        r("pecho", "Dolor u opresión en el pecho", "dolor (fuerte |intenso )?(de|en el) pecho", "opresion (en el )?pecho",
            "me aprieta el pecho", "dolor .{0,20}brazo izquierdo", "infarto"),
        r("inconsciente", "Inconsciente o desmayo", "inconsciente", "no reacciona", "no responde", "desmay", "perdi el conocimiento",
            "perdio el conocimiento", "me desvaneci"),
        r("convulsion", "Convulsión", "convuls", "ataque epilep"),
        r("acv", "Signos de derrame cerebral (cara, brazo, habla)", "cara (chueca|caida|torcida)", "se me cayo la cara",
            "no puedo (hablar|mover (el|un) (brazo|lado))", "habla rar", "arrastra las palabras", "se me durmio (medio|un lado)",
            "perdi la fuerza (de|en) (un|el) (brazo|lado)"),
        r("sangrado", "Sangrado abundante que no para", "sangrado (abundante|que no para|mucho)", "no (para|deja) de sangrar",
            "sangra (mucho|a chorros)(?!.{0,15}nariz)", "mucha sangre", "vomito con sangre", "vomite sangre", "toso sangre"),
        r("anafilaxia", "Reacción alérgica grave", "se me cierra la garganta", "garganta cerrada", "hinchazon de (la )?(cara|lengua|labios|garganta)",
            "se (me |le )?(hincho|hincharon|hinchan?|esta hinchando|estan hinchando) (la |los |el )?(cara|lengua|labios?|garganta|boca|parpados)", "anafilax"),
        r("cabeza", "Golpe en la cabeza con señales de alarma",
            "(golpe|pegu?e|pego|cai|cayo|azot).{0,30}cabeza.{0,40}(vomit|desmay|confus|sangr|dormid|convuls)",
            "(vomit|confus).{0,40}(golpe|pegu?e|pego).{0,30}cabeza", "el peor dolor de cabeza", "dolor de cabeza (subito|explosivo|como nunca)"),
        r("envenenamiento", "Posible envenenamiento o sobredosis", "envenen", "sobredosis", "se tomo (todas|muchas) (las )?pastillas",
            "me tome (todas|muchas) (las )?pastillas", "(trago|tomo|bebio|ingirio|se tomo|se trago|se bebio|me tome|me trague|comio) (un poco de |algo de )?(cloro|veneno|detergente|raticida|"
                + "gasolina|thinner|insecticida|plaguicida|herbicida|sosa|acido|blanqueador|limpiador|pila|bateria)"),
        r("ojo_quimico", "Químico o cuerpo clavado en el ojo", "(cloro|quimic|acido|\\bcal\\b|sosa|blanqueador|limpiador|gasolina|cemento).{0,25}\\bojo",
            "\\bojo.{0,25}(cloro|quimic|acido|sosa|blanqueador)", "(se me clavo|clavado|incrustado).{0,20}\\bojo"),
        r("electrico", "Descarga eléctrica", "toque electrico", "descarga electrica", "electrocut", "le dio (la )?corriente",
            "me dio (la )?corriente", "le cayo un rayo"),
        r("ahogamiento", "Casi ahogamiento", "se (estaba )?ahog(o|aba) en (la )?(alberca|piscina|rio|mar|agua)", "lo sacamos del agua"),
        r("quemadura", "Quemadura grande o en cara/genitales/vías aéreas", "quemadura (grande|grave|de tercer grado|en (la )?cara)",
            "se quemo (toda|todo|la cara)", "inhalo humo"),
        r("crisis", "Pensamientos de hacerte daño", "quiero morir", "me quiero morir", "suicid", "quitarme la vida", "hacerme dano",
            "matarme", "no quiero vivir"),
        r("embarazo", "Sangrado o dolor fuerte en el embarazo", "embarazad.{0,40}(sangr|dolor fuerte)", "(sangr|dolor fuerte).{0,40}embaraz"),
    )

    // Síntomas que se atienden hoy (amarillo).
    private val AMARILLAS = listOf(
        r("fiebre", "Fiebre", "fiebre", "calentura", "temperatura alta"), r("vomito", "Vómito", "vomit", "guacar"),
        r("diarrea", "Diarrea", "diarrea", "chorro"), r("nausea", "Náusea", "nause", "\\basco\\b"),
        r("dolor_cabeza", "Dolor de cabeza", "dolor de cabeza", "migran", "jaqueca"), r("mareo", "Mareo", "\\bmare[oa]", "\\bmaread", "vertigo"),
        r("tos", "Tos", "\\btos\\b", "tosiendo"), r("atragantamiento", "Atragantamiento (si no puede toser, hablar o respirar: ROJO)", "atragant", "se le atoro", "se me atoro"),
        r("hipoglucemia", "Posible azúcar baja (temblor y sudor)", "temblando y sudando", "sudor frio", "suda frio", "tembloroso"),
        r("falta_aire", "Falta de aire leve", "falta de aire", "me canso al respirar", "asma"),
        r("herida", "Herida o cortada", "\\bcort[eao]", "herida", "raspon", "\\brasp[eo]", "raspad"), r("quemadura_leve", "Quemadura", "\\bquem[eoa]", "ampolla"),
        r("golpe", "Golpe o caída", "\\bgolpe", "me cai", "se cayo", "chichon", "moreton"),
        r("esguince", "Torcedura o posible fractura", "\\btorc[ei]", "esguince", "fractur", "\\bhueso"),
        r("picadura", "Picadura o mordedura", "picadura", "piquete", "me pico", "\\bmordi", "alacran"),
        r("alergia", "Alergia", "alergi", "ronchas", "comezon", "urticaria"), r("dolor", "Dolor", "\\bdolor\\b", "\\bduele"),
        r("sangrado_leve", "Sangrado leve", "\\bsangr"), r("ojo", "Molestia en el ojo", "ojo (rojo|irritado)", "me entro algo en el ojo"),
        r("ansiedad", "Ansiedad o mucho estrés", "ansiedad", "panico", "estres", "angustia"), r("insomnio", "No dormí bien", "no (pude|puedo) dormir", "insomnio"),
        r("cansancio", "Cansancio fuera de lo normal", "muy cansad", "agotad", "sin fuerza"), r("deshidratacion", "Deshidratación", "deshidrat", "boca seca"),
        r("azucar", "Azúcar alta o baja", "azucar (alta|baja)", "glucosa (alta|baja)", "hipoglucem"),
        r("presion", "Presión alta o baja", "presion (alta|baja)"),
    )

    private val NEGACION = Regex("\\b(no|sin|nunca|ni|tampoco|ya no|nada de)\\b[^.,;]{0,18}$")

    /** Busca la regla en el texto ignorando las coincidencias negadas. Patrones que empiezan con «no» se aceptan tal cual. */
    private fun coincide(t: String, regla: Regla): Boolean = regla.patrones.any { p ->
        p.findAll(t).any { m ->
            p.pattern.startsWith("no ") || !NEGACION.containsMatchIn(t.substring(maxOf(0, m.range.first - 22), m.range.first))
        }
    }

    private fun temperatura(t: String): Double? = Regex("(\\d{2}(?:[.,]\\d)?)\\s*(?:°|grados|º|c\\b)").findAll(t)
        .mapNotNull { it.groupValues[1].replace(',', '.').toDoubleOrNull() }.filter { it in 34.0..44.0 }.maxOrNull()

    private fun dolorEscala(t: String): Int? = Regex("\\b(10|[0-9])\\s*(?:/|de)\\s*10\\b").find(t)?.groupValues?.get(1)?.toIntOrNull()

    fun evaluar(texto: String, animo: Int? = null, previas: List<EntradaDiario> = emptyList(),
                tm: MissionTelemetry? = null, ahora: Long = 0L, dia: (Long) -> Long = { it / 86_400_000L }): Evaluacion {
        val t = normalizar(texto).replace(Regex("\\s+"), " ")
        val rojas = ROJAS.filter { coincide(t, it) }
        val amarillas = AMARILLAS.filter { coincide(t, it) }
            // «dolor» genérico no se repite si ya hay un dolor concreto
            .let { a -> if (a.size > 1) a.filterNot { it.clave == "dolor" || (it.clave == "sangrado_leve" && rojas.any { r -> r.clave == "sangrado" }) } else a }
        val motivos = mutableListOf<String>()
        var triage = Triage.NOMINAL

        if (rojas.isNotEmpty()) { triage = Triage.CRITICO; motivos += rojas.map { it.nombre } }

        val temp = temperatura(t)
        if (temp != null) when {
            temp >= 41.0 -> { triage = Triage.CRITICO; motivos += "Temperatura de $temp °C" }
            temp >= 38.0 -> { triage = Triage.peor(triage, Triage.ADVERTENCIA); motivos += "Fiebre de $temp °C" }
            temp <= 35.0 -> { triage = Triage.CRITICO; motivos += "Temperatura baja de $temp °C (posible hipotermia)" }
        }
        val bebe = Regex("beb[eé]|recien nacid|(\\d) mes(es)? de (nacid|edad)").containsMatchIn(t)
        if (bebe && amarillas.any { it.clave == "fiebre" }) { triage = Triage.CRITICO; motivos += "Fiebre en un bebé" }
        dolorEscala(t)?.let { d -> if (d >= 8) { triage = Triage.peor(triage, Triage.ADVERTENCIA); motivos += "Dolor de $d/10" } }

        if (amarillas.isNotEmpty()) { triage = Triage.peor(triage, Triage.ADVERTENCIA); motivos += amarillas.map { it.nombre } }
        if (animo != null && animo <= 1) { triage = Triage.peor(triage, Triage.ADVERTENCIA); motivos += "Te sientes muy mal hoy" }

        // Persistencia: el mismo síntoma en 3 días distintos de la última semana
        val sintomas = (rojas + amarillas).map { it.clave }.distinct()
        if (ahora > 0) {
            val semana = previas.filter { ahora - it.horaMs in 0..7 * 86_400_000L }
            for (s in sintomas) {
                val dias = (semana.filter { s in it.sintomas }.map { dia(it.horaMs) } + dia(ahora)).distinct().size
                if (dias >= 3) { triage = Triage.peor(triage, Triage.ADVERTENCIA); motivos += "Llevas $dias días con: ${nombre(s)}" }
            }
            val diasAmarillos = semana.filter { it.triage != Triage.NOMINAL }.map { dia(it.horaMs) }.distinct().size
            if (diasAmarillos >= 4 && triage == Triage.NOMINAL) motivos += "Esta semana tuviste $diasAmarillos días con molestias"
        }

        // Signos vitales de la pulsera / cámara
        tm?.let {
            if (it.pulso != null && it.pulso !in 40..150) { triage = Triage.CRITICO; motivos += "Pulso de ${it.pulso} lpm" }
            if (it.spo2 != null && it.spo2 < 90) { triage = Triage.CRITICO; motivos += "Oxígeno en sangre de ${it.spo2} %" }
        }

        val crisis = rojas.any { it.clave == "crisis" }
        val (titulo, accion) = when (triage) {
            Triage.CRITICO -> "ROJO · Emergencia" to (if (crisis)
                "No estás solo. Llama ahora a la Línea de la Vida 800 911 2000 (24 h, gratis) o al 911. Quédate con alguien de confianza."
            else "Llama al 911 ahora (o activa la Baliza SOS). Mientras llega la ayuda, sigue los pasos de abajo y no dejes sola a la persona.")
            Triage.ADVERTENCIA -> "AMARILLO · Atiéndelo hoy" to
                "Sigue los cuidados de abajo y vigila. Si empeora, aparece dolor de pecho, falta de aire, desmayo o sangrado que no para, es ROJO: llama al 911. " +
                if (motivos.any { it.startsWith("Llevas") }) "Como se repite varios días, pide cita con un médico." else "Si no mejora en 24–48 h, consulta a un médico."
            else -> "VERDE · Todo en orden" to (if (t.isBlank()) "Escribe cómo te sientes hoy." else "No encontré señales de alarma. Sigue cuidándote y escribe mañana.")
        }
        val consulta = (rojas + amarillas).joinToString(" ") { it.nombre } + " " + texto
        return Evaluacion(triage, titulo, accion, motivos.distinct(), sintomas, consulta)
    }

    fun nombre(clave: String): String = (ROJAS + AMARILLAS).firstOrNull { it.clave == clave }?.nombre ?: clave
}
