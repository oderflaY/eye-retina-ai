package com.temrarios.ojoscan.guia

import com.temrarios.ojoscan.api.AnalisisAnemia
import com.temrarios.ojoscan.api.AnalisisOjo
import com.temrarios.ojoscan.api.AnalisisPerfil
import com.temrarios.ojoscan.api.AnalisisRostro
import kotlin.math.roundToInt

/** Qué tan pronto conviene actuar. El orden importa: el pronóstico toma el peor. */
enum class Nivel { OK, MEDIA, ALTA }

/** Una conclusión del chequeo, con las mediciones que la respaldan. */
data class Conclusion(
    val area: String,
    val nivel: Nivel,
    val titulo: String,
    val detalle: String,
    val fuentes: List<String>,
)

data class Pronostico(
    val nivel: Nivel,
    val resumen: String,
    val conclusiones: List<Conclusion>,
    /** Mediciones que dieron un resultado usable / mediciones intentadas. */
    val usadas: List<String>,
    val fallidas: List<String>,
)

/** Todo lo que regresó el servidor en un chequeo completo (null = no se hizo o falló). */
data class ResultadosChequeo(
    val ojoIzquierdo: AnalisisOjo? = null,
    val ojoDerecho: AnalisisOjo? = null,
    val conjuntiva: AnalisisAnemia? = null,
    val palma: AnalisisAnemia? = null,
    /** Experimental (sin validar con hemoglobina). */
    val encia: AnalisisAnemia? = null,
    val rostro: AnalisisRostro? = null,
    /** Lo calculado con los datos escritos (IMC, grasa, presión, corazón…). */
    val perfil: AnalisisPerfil? = null,
    val fallidas: List<String> = emptyList(),
)

/** Confianza mínima para tomar en serio un hallazgo de ojo. */
private const val CONF_OJO = 0.6

/**
 * Cruza todas las mediciones en un solo pronóstico. Reglas (simples y explicables a propósito):
 * - Ojos: la foto de cerca manda; la foto de la cara confirma o no (ve el ojo pequeño).
 * - Anemia: conjuntiva y palma votan; dos que coinciden pesan más que una sola.
 * - Ictericia y parálisis: solo las ve el rostro.
 * - Cruces: palidez + ojos amarillos (posible hemólisis), asimetría + ojo rojo.
 */
fun pronosticar(r: ResultadosChequeo): Pronostico {
    val c = mutableListOf<Conclusion>()
    val usadas = buildList {
        if (r.ojoIzquierdo != null) add("Ojo izquierdo de cerca")
        if (r.ojoDerecho != null) add("Ojo derecho de cerca")
        if (r.conjuntiva?.valida == true) add("Párpado (conjuntiva)")
        if (r.rostro?.cara == true) add("Rostro")
        if (r.palma?.valida == true) add("Palma")
        if (r.encia?.valida == true) add("Encía (experimental)")
        if (r.perfil?.indicadores?.isNotEmpty() == true) add("Tus datos")
    }
    val fallidas = r.fallidas + buildList {
        if (r.conjuntiva != null && !r.conjuntiva.valida) add("Párpado (no se vio la conjuntiva)")
        if (r.rostro != null && !r.rostro.cara) add("Rostro (no se encontró la cara)")
        if (r.palma != null && !r.palma.valida) add("Palma (no se encontró la palma)")
        if (r.encia != null && !r.encia.valida) add("Encía (${r.encia.titulo.lowercase()})")
    }

    ojos(r)?.let(c::add)
    val anemia = conEncia(anemia(r.conjuntiva, r.palma), r.encia)
    anemia?.let(c::add)

    val rostro = r.rostro?.takeIf { it.cara }
    val ictericia = rostro?.hallazgos?.firstOrNull { it.clave == "ictericia" }
    val paralisis = rostro?.hallazgos?.firstOrNull { it.clave == "paralisis_facial" }
    ictericia?.let { h ->
        c += when (h.resultado) {
            "posible" -> Conclusion("Ictericia", Nivel.ALTA, "El blanco de los ojos se ve amarillo",
                "Puede venir del hígado, la vesícula o la sangre. Pide revisión médica y un análisis de bilirrubina.",
                listOf("Rostro ${pct(h.probabilidad)}"))
            "no_concluyente" -> Conclusion("Ictericia", Nivel.MEDIA, "Color de ojos dudoso",
                "Repite la foto de la cara con luz de día, sin luz amarilla de foco.", listOf("Rostro ${pct(h.probabilidad)}"))
            else -> Conclusion("Ictericia", Nivel.OK, "Sin ojos amarillos", "", listOf("Rostro ${pct(h.probabilidad)}"))
        }
    }
    paralisis?.let { h ->
        c += when (h.resultado) {
            "posible" -> Conclusion("Rostro", Nivel.ALTA, "Un lado de la cara se mueve menos",
                "Si empezó de golpe, o hay debilidad en un brazo o dificultad para hablar, llama al 911 YA (puede ser un " +
                    "infarto cerebral). Si no, consulta pronto: puede ser parálisis de Bell.",
                listOf("Rostro ${pct(h.probabilidad)}"))
            "no_concluyente" -> Conclusion("Rostro", Nivel.MEDIA, "Asimetría dudosa",
                "Repite la foto con la cara de frente y relajada.", listOf("Rostro ${pct(h.probabilidad)}"))
            else -> Conclusion("Rostro", Nivel.OK, "Cara simétrica", "", listOf("Rostro ${pct(h.probabilidad)}"))
        }
    }

    // Perfil escrito: cada indicador fuera de rango es una conclusión; los sanos se resumen en una.
    r.perfil?.let { pf ->
        val malos = pf.indicadores.filter { it.nivel != "ok" }
        for (i in malos) c += Conclusion(i.nombre, if (i.nivel == "alta") Nivel.ALTA else Nivel.MEDIA,
            "${i.valor} · ${i.texto}", "", listOf(i.fuente))
        val sanos = pf.indicadores.filter { it.nivel == "ok" }
        if (sanos.isNotEmpty()) c += Conclusion("Tus datos", Nivel.OK,
            sanos.joinToString(" · ") { "${it.nombre} ${it.valor}" }, "En rango sano.", sanos.map { it.fuente }.distinct())
    }
    val indicador = { k: String -> r.perfil?.indicadores?.firstOrNull { it.clave == k } }

    // Cruces que ninguna medición ve sola
    if (ictericia?.resultado == "posible" && anemia?.nivel != null && anemia.nivel != Nivel.OK) {
        c += Conclusion("Cruce", Nivel.ALTA, "Palidez + ojos amarillos a la vez",
            "Juntas pueden indicar que los glóbulos rojos se están destruyendo (anemia hemolítica). Consulta pronto.",
            listOf("Anemia", "Ictericia"))
    }
    val ojoRojo = listOfNotNull(r.ojoIzquierdo, r.ojoDerecho).any { it.principal == "enrojecimiento" && it.confianza >= CONF_OJO }
    if (paralisis?.resultado == "posible" && ojoRojo) {
        c += Conclusion("Cruce", Nivel.MEDIA, "Asimetría + ojo rojo",
            "Un ojo que no cierra bien se reseca e irrita. Lubrícalo y pide revisión de oftalmología.",
            listOf("Rostro", "Ojo de cerca"))
    }

    val presionAlta = indicador("presion")?.nivel == "alta"
    val corazon = indicador("corazon")
    if (presionAlta && corazon != null && corazon.nivel != "ok") {
        c += Conclusion("Cruce", Nivel.ALTA, "Presión alta + dolor de pecho de posible origen cardiaco",
            "Juntos aumentan el riesgo. No lo dejes pasar: consulta con un médico esta semana.", listOf("Presión", "Corazón"))
    }
    val pulsoAlto = listOf("pulso", "pulso_historial").any { k -> indicador(k)?.let { it.nivel != "ok" && it.texto.contains("alto") } == true } ||
        indicador("pulso_tendencia") != null
    val duermeMucho = indicador("sueno")?.texto?.contains("más de 10") == true
    if ((pulsoAlto || duermeMucho) && anemia != null && anemia.nivel != Nivel.OK) {
        val que = listOfNotNull("pulso acelerado".takeIf { pulsoAlto }, "mucho sueño/cansancio".takeIf { duermeMucho }).joinToString(" y ")
        c += Conclusion("Cruce", Nivel.MEDIA, "Palidez + $que",
            "Con poca hemoglobina el corazón late más rápido para llevar oxígeno y el cuerpo se cansa. Juntas hacen más " +
                "probable la anemia: pide una biometría hemática.", listOf("Pulsera", "Anemia"))
    }
    val apnea = indicador("apnea")
    if (apnea != null && presionAlta) {
        c += Conclusion("Cruce", Nivel.ALTA, "Bajadas de oxígeno de noche + presión alta",
            "La apnea del sueño sin tratar sube la presión y el riesgo de infarto. Pide un estudio de sueño.",
            listOf("Pulsera", "Presión"))
    }
    val sedentario = indicador("pasos")?.nivel == "media"
    val abdomenAlto = indicador("cintura_estatura")?.nivel?.let { it != "ok" } == true
    val glucosaAlta = indicador("glucosa")?.nivel?.let { it != "ok" } == true
    if (abdomenAlto && (presionAlta || glucosaAlta || sedentario)) {
        c += Conclusion("Cruce", Nivel.MEDIA, "Grasa abdominal + " + listOfNotNull("presión alta".takeIf { presionAlta }, "glucosa alta".takeIf { glucosaAlta }, "poca actividad".takeIf { sedentario }).joinToString(" y "),
            "Es el patrón del síndrome metabólico. Bajar la cintura mejora ambos; pide revisión de glucosa, colesterol y presión.",
            listOf("Cintura", "Presión", "Glucosa"))
    }

    val nivel = c.maxOfOrNull { it.nivel } ?: Nivel.OK
    val atender = c.count { it.nivel != Nivel.OK }
    val resumen = when {
        usadas.isEmpty() -> "No se pudo obtener ninguna medición. Revisa la conexión y repite el chequeo."
        nivel == Nivel.ALTA -> "Hay señales que conviene revisar con un médico pronto."
        nivel == Nivel.MEDIA -> if (atender == 1) "Una señal leve o dudosa: vigílala o repite esa medición."
            else "$atender señales leves o dudosas: vigílalas o repite esas mediciones."
        else -> "Sin señales en las ${usadas.size} mediciones."
    }
    // Lo urgente primero
    return Pronostico(nivel, resumen, c.sortedByDescending { it.nivel }, usadas, fallidas)
}

private fun ojos(r: ResultadosChequeo): Conclusion? {
    val cerca = listOfNotNull(r.ojoIzquierdo?.let { "izquierdo" to it }, r.ojoDerecho?.let { "derecho" to it })
    val lejos = r.rostro?.takeIf { it.cara }?.ojos?.values.orEmpty()
    if (cerca.isEmpty() && lejos.isEmpty()) return null
    val fuentes = cerca.map { (lado, a) -> "Ojo $lado ${pct(a.confianza)}" } + (if (lejos.isNotEmpty()) listOf("Rostro") else emptyList())

    val anormales = cerca.filter { it.second.principal != "normal" }
    if (anormales.isEmpty()) {
        val lejosRaro = lejos.filter { it.principal != "normal" }
        return if (cerca.isEmpty() && lejosRaro.isNotEmpty())
            Conclusion("Ojos", Nivel.MEDIA, lejosRaro.first().titulo,
                "Solo se vio en la foto de la cara, donde el ojo sale pequeño. Repite el escaneo de cerca para confirmar.", fuentes)
        else Conclusion("Ojos", Nivel.OK, "Ojos sin hallazgos visibles",
            if (lejosRaro.isNotEmpty()) "La foto de la cara marcó algo, pero la de cerca (más fiable) no lo confirma." else "", fuentes)
    }
    val lineas = anormales.map { (lado, a) ->
        val confirmado = lejos.any { it.principal == a.principal }
        val seguro = a.confianza >= CONF_OJO
        "Ojo $lado: ${a.titulo} (${pct(a.confianza)}" +
            (if (!seguro) ", dudoso" else "") +
            (if (confirmado) ", también se ve en la foto de la cara" else "") + ")."
    }
    val seguros = anormales.any { it.second.confianza >= CONF_OJO }
    val titulo = anormales.map { it.second.titulo }.distinct().joinToString(" · ")
    return Conclusion("Ojos", Nivel.MEDIA, titulo,
        lineas.joinToString(" ") + if (seguros) " Conviene una revisión con oftalmología." else " Repite con buena luz y el ojo bien abierto.",
        fuentes)
}

/**
 * La encía no está validada con hemoglobina: puede confirmar o matizar lo que digan párpado y palma,
 * pero nunca sube el nivel por sí sola (sola, como mucho "vigilar").
 */
private fun conEncia(base: Conclusion?, encia: AnalisisAnemia?): Conclusion? {
    if (encia == null || !encia.valida) return base
    val fuente = "Encía (experimental)"
    val palida = encia.resultado == "palidez"
    val nota = when {
        encia.inflamacion -> "La encía se ve inflamada (roja), así que no ayuda para la anemia; si sangra, ve al dentista."
        palida -> "La encía también se ve pálida (método experimental)."
        encia.resultado == "sin_palidez" -> "La encía tiene color normal (método experimental)."
        else -> "El color de la encía está en el límite (método experimental)."
    }
    if (base == null) {
        return if (palida) Conclusion("Anemia", Nivel.MEDIA, "Posible anemia (solo por la encía)",
            "Solo la encía, que es experimental, ve palidez. Haz el paso del párpado o pide un análisis de sangre.", listOf(fuente))
        else Conclusion("Anemia", Nivel.OK, if (encia.inflamacion) "Encía inflamada" else "Sin palidez en la encía",
            "$nota No reemplaza a la medición del párpado.", listOf(fuente))
    }
    return base.copy(detalle = (base.detalle + " " + nota).trim(), fuentes = base.fuentes + fuente)
}

/** Conjuntiva y palma votan; cada una da palidez / sin_palidez / no_concluyente (o no es válida). */
private fun anemia(conj: AnalisisAnemia?, palma: AnalisisAnemia?): Conclusion? {
    val validas = listOfNotNull(conj?.takeIf { it.valida }?.let { "Párpado" to it }, palma?.takeIf { it.valida }?.let { "Palma" to it })
    if (validas.isEmpty()) return null
    val fuentes = validas.map { (n, a) -> "$n ${a.probabilidad?.let(::pct) ?: "—"}" }
    val palidas = validas.count { it.second.resultado == "palidez" }
    val sanas = validas.count { it.second.resultado == "sin_palidez" }
    // Promedio ponderado por lo que discrimina cada modelo (AUC − 0.5); solo informativo.
    val pesos = validas.mapNotNull { (_, a) -> a.probabilidad?.let { p -> p to ((a.validacion?.auc ?: 0.75) - 0.5).coerceAtLeast(0.05) } }
    val prom = if (pesos.isEmpty()) null else pesos.sumOf { it.first * it.second } / pesos.sumOf { it.second }
    val probTxt = prom?.let { " Probabilidad combinada: ${pct(it)}." } ?: ""
    return when {
        palidas >= 2 -> Conclusion("Anemia", Nivel.ALTA, "Posible anemia: párpado y palma coinciden",
            "Las dos mediciones ven palidez.$probTxt Pide una biometría hemática (análisis de sangre).", fuentes)
        palidas == 1 && sanas >= 1 -> Conclusion("Anemia", Nivel.MEDIA, "Anemia dudosa: las mediciones no coinciden",
            "Una ve palidez y la otra no.$probTxt Repite con luz de día o pide un análisis de sangre si te sientes cansado.", fuentes)
        palidas == 1 -> Conclusion("Anemia", Nivel.MEDIA, "Posible anemia (una sola medición)",
            "Solo una medición dio veredicto y ve palidez.$probTxt Repite la otra o pide un análisis de sangre.", fuentes)
        sanas >= 1 && sanas == validas.size -> Conclusion("Anemia", Nivel.OK,
            if (sanas >= 2) "Sin palidez en párpado ni palma" else "Sin palidez (una medición)", probTxt.trim(), fuentes)
        sanas >= 1 -> Conclusion("Anemia", Nivel.OK, "Sin palidez (la otra medición no fue concluyente)", probTxt.trim(), fuentes)
        else -> Conclusion("Anemia", Nivel.MEDIA, "Anemia no concluyente",
            "Ninguna medición fue clara.$probTxt Repite con luz blanca de día.", fuentes)
    }
}

private fun pct(p: Double) = "${(p * 100).roundToInt()}%"
