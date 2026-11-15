package com.temrarios.ojoscan.astro

import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Historial personal de salud: TODAS las mediciones de la persona en un solo lugar (en el teléfono, nada sale).
 * Encima: recordatorios de revisión, clasificación con semáforo, alertas por valor y por tendencia,
 * y un RAG personal que contesta preguntas en español con los datos reales de la persona.
 */
@Serializable
data class Registro(
    val horaMs: Long,
    /** pulso · spo2 · pulso_reposo · anemia · diario · pvt · impacto · ojo · rostro · perfil */
    val tipo: String,
    val nombre: String,
    val valor: Double? = null,
    val unidad: String = "",
    /** Texto libre buscable (resultado, lo que escribió en el diario…). */
    val texto: String = "",
    val triage: Triage = Triage.NOMINAL,
    val fuente: String = "",
)

/** Una revisión periódica: cada cuánto hacerla según cómo salió la última. */
data class TipoRevision(
    val clave: String,
    val nombre: String,
    /** Tipos de registro que cuentan como «hecha». */
    val tipos: Set<String>,
    val diasNormal: Int,
    val diasAtencion: Int,
    val diasAlerta: Int,
    val como: String,
)

enum class EstadoRevision { AL_DIA, PRONTO, VENCIDA, NUNCA }

data class Revision(val tipo: TipoRevision, val estado: EstadoRevision, val ultima: Registro?, val tocaMs: Long?, val diasParaTocar: Int?) {
    val texto: String get() = when (estado) {
        EstadoRevision.NUNCA -> "Nunca la has hecho"
        EstadoRevision.VENCIDA -> "Atrasada ${-(diasParaTocar ?: 0)} día(s)"
        EstadoRevision.PRONTO -> if ((diasParaTocar ?: 0) <= 0) "Toca hoy" else "Toca mañana"
        EstadoRevision.AL_DIA -> "En ${diasParaTocar} días"
    }
}

data class Alerta(val id: String, val triage: Triage, val titulo: String, val detalle: String, val horaMs: Long)

object Seguimiento {
    const val DIA = 86_400_000L

    /**
     * Anemia en MODO PRUEBAS: se mide y se guarda en el historial, pero NO da alertas, NO pide revisiones
     * y NO cuenta en el estado general (el modelo de la cámara aún no está validado lo suficiente).
     */
    const val ANEMIA_EN_PRUEBAS = true

    val REVISIONES = listOf(
        TipoRevision("diario", "Diario de salud", setOf("diario"), 1, 1, 1, "Escribe cómo te sientes en el Diario."),
        TipoRevision("pulso", "Pulso", setOf("pulso", "pulso_reposo"), 7, 2, 1, "Mide con la cámara (Biometría PPG) o la pulsera."),
        TipoRevision("pvt", "Test de alerta (PVT)", setOf("pvt"), 7, 3, 1, "Haz el Test Cognitivo PVT."),
        TipoRevision("anemia", "Tamizaje de anemia", setOf("anemia"), 30, 14, 7, "Párpado, palma o encía en el tablero."),
        TipoRevision("ojo", "Revisión del ojo", setOf("ojo"), 30, 14, 7, "Escaneo de ojo de cerca."),
        TipoRevision("perfil", "Peso y presión", setOf("perfil"), 30, 14, 7, "Actualiza «Tus datos» en el chequeo completo."),
    )

    // ── recordatorios ──
    fun revisiones(h: List<Registro>, ahora: Long): List<Revision> = REVISIONES.filterNot { ANEMIA_EN_PRUEBAS && it.clave == "anemia" }.map { r ->
        val ultima = h.filter { it.tipo in r.tipos }.maxByOrNull { it.horaMs }
        if (ultima == null) return@map Revision(r, EstadoRevision.NUNCA, null, null, null)
        // el intervalo se acorta si la última salió en amarillo o rojo (la peor del día de la última)
        val diaUlt = ultima.horaMs / DIA
        val peor = Triage.peor(*h.filter { it.tipo in r.tipos && it.horaMs / DIA == diaUlt }.map { it.triage }.toTypedArray())
        val dias = when (peor) { Triage.CRITICO -> r.diasAlerta; Triage.ADVERTENCIA -> r.diasAtencion; else -> r.diasNormal }
        val toca = ultima.horaMs + dias * DIA
        val faltan = ((toca - ahora).toDouble() / DIA).let { if (it < 0) -kotlin.math.ceil(-it) else kotlin.math.floor(it) }.toInt()
        val estado = when { toca <= ahora -> if (ahora - toca < DIA) EstadoRevision.PRONTO else EstadoRevision.VENCIDA
            toca - ahora <= DIA -> EstadoRevision.PRONTO; else -> EstadoRevision.AL_DIA }
        Revision(r, estado, ultima, toca, faltan)
    }.sortedBy { when (it.estado) { EstadoRevision.VENCIDA -> 0; EstadoRevision.PRONTO -> 1; EstadoRevision.NUNCA -> 2; else -> 3 } }

    // ── alertas (valor y tendencia) ──
    fun alertas(h: List<Registro>, ahora: Long): List<Alerta> {
        val a = mutableListOf<Alerta>()
        val dia = ahora / DIA
        // 1. cualquier medición en ROJO en las últimas 24 h
        h.filter { ahora - it.horaMs in 0..DIA && it.triage == Triage.CRITICO && !(ANEMIA_EN_PRUEBAS && it.tipo == "anemia") }.forEach {
            a += Alerta("rojo_${it.tipo}_${it.horaMs}", Triage.CRITICO, "${it.nombre} en rojo",
                "${fmt(it)} — ${hace(it.horaMs, ahora)}. Si te sientes mal, llama al 911.", it.horaMs)
        }
        // 2. pulso: subida contra tu propia línea base
        val pulsos = h.filter { it.tipo == "pulso" && it.valor != null }
        val sem = pulsos.filter { ahora - it.horaMs <= 7 * DIA }.mapNotNull { it.valor }
        val base = pulsos.filter { ahora - it.horaMs in (7 * DIA + 1)..(37 * DIA) }.mapNotNull { it.valor }
        if (sem.size >= 3 && base.size >= 3) {
            val d = mediana(sem) - mediana(base)
            if (d >= 10) a += Alerta("pulso_sube_$dia", Triage.ADVERTENCIA, "Tu pulso subió",
                "Mediana de esta semana ${mediana(sem).roundToInt()} lpm contra ${mediana(base).roundToInt()} lpm de tu mes anterior (+${d.roundToInt()}). " +
                    "Puede ser estrés, poco sueño, fiebre o deshidratación; si sigue así, consúltalo.", ahora)
        }
        val ult5 = pulsos.sortedBy { it.horaMs }.takeLast(5)
        if (ult5.size >= 3 && ult5.count { (it.valor ?: 0.0) > 100 } >= 3) a += Alerta("taquicardia_$dia", Triage.ADVERTENCIA,
            "Pulso alto repetido", "${ult5.count { (it.valor ?: 0.0) > 100 }} de tus últimas ${ult5.size} mediciones pasaron de 100 lpm en reposo.", ahora)
        // 3. oxígeno bajo repetido
        val ox = h.filter { it.tipo == "spo2" && it.valor != null && ahora - it.horaMs <= 7 * DIA }
        if (ox.count { it.valor!! < 94 } >= 2) a += Alerta("spo2_bajo_$dia", Triage.ADVERTENCIA, "Oxígeno bajo varias veces",
            "${ox.count { it.valor!! < 94 }} lecturas de SpO₂ menores a 94 % esta semana (mínimo ${ox.minOf { it.valor!! }} %).", ahora)
        // 4. anemia que empeora (mismo método, contra la medición anterior)
        if (!ANEMIA_EN_PRUEBAS) h.filter { it.tipo == "anemia" && it.valor != null }.groupBy { it.fuente }.forEach { (metodo, l) ->
            val s = l.sortedBy { it.horaMs }
            if (s.size >= 2 && s.last().valor!! - s[s.size - 2].valor!! >= 0.15 && ahora - s.last().horaMs <= 14 * DIA)
                a += Alerta("anemia_sube_${metodo}_${s.last().horaMs}", Triage.ADVERTENCIA, "Anemia: tu $metodo empeoró",
                    "De ${pct(s[s.size - 2].valor!!)} a ${pct(s.last().valor!!)} entre las dos últimas mediciones. Considera un análisis de sangre (biometría hemática).", s.last().horaMs)
        }
        // 5. diario: muchos días con molestias
        val diasMal = h.filter { it.tipo == "diario" && ahora - it.horaMs <= 7 * DIA && it.triage != Triage.NOMINAL }.map { it.horaMs / DIA }.distinct().size
        if (diasMal >= 3) a += Alerta("diario_semana_$dia", Triage.ADVERTENCIA, "Semana con molestias",
            "$diasMal días de los últimos 7 tu diario salió en amarillo o rojo. Si no mejora, pide cita con un médico.", ahora)
        // 6. revisiones muy atrasadas (más del doble de su intervalo)
        revisiones(h, ahora).filter { it.estado == EstadoRevision.VENCIDA && it.ultima != null && (it.tocaMs ?: ahora) < ahora - it.tipo.diasNormal * DIA }
            .forEach { a += Alerta("atrasada_${it.tipo.clave}_$dia", Triage.NOMINAL, "${it.tipo.nombre}: muy atrasada", it.tipo.como, ahora) }
        return a.sortedWith(compareByDescending<Alerta> { it.triage.orden }.thenByDescending { it.horaMs })
    }

    // ── clasificación de una medición nueva ──
    fun clasificarPulso(lpm: Double) = when { lpm < 40 || lpm > 150 -> Triage.CRITICO; lpm < 50 || lpm > 110 -> Triage.ADVERTENCIA; else -> Triage.NOMINAL }
    fun clasificarSpo2(v: Double) = when { v < 90 -> Triage.CRITICO; v < 94 -> Triage.ADVERTENCIA; else -> Triage.NOMINAL }
    fun clasificarProbabilidad(p: Double) = when { p >= 0.65 -> Triage.CRITICO; p >= 0.35 -> Triage.ADVERTENCIA; else -> Triage.NOMINAL }

    // ── utilidades ──
    fun mediana(v: List<Double>): Double { val s = v.sorted(); val n = s.size
        return if (n == 0) Double.NaN else if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2 }

    fun pct(p: Double) = "${(p * 100).roundToInt()} %"

    fun fmt(r: Registro): String = r.valor?.let { v ->
        val num = if (r.unidad == "%prob") pct(v) else if (abs(v - v.roundToInt()) < 0.05) "${v.roundToInt()} ${r.unidad}".trim() else "${(v * 10).roundToInt() / 10.0} ${r.unidad}".trim()
        if (r.texto.isNotBlank() && r.tipo != "diario") "$num (${r.texto})" else num
    } ?: r.texto.ifBlank { r.triage.etiqueta }

    fun hace(ms: Long, ahora: Long): String {
        val d = (ahora - ms) / 60_000
        return when { d < 60 -> "hace $d min"; d < 1440 -> "hace ${d / 60} h"; d < 2 * 1440 -> "ayer"; else -> "hace ${d / 1440} días" }
    }

    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = false }
    fun aJson(h: List<Registro>): String = json.encodeToString(ListSerializer(Registro.serializer()), h)
    fun deJson(s: String?): List<Registro> = s?.let { runCatching { json.decodeFromString(ListSerializer(Registro.serializer()), it) }.getOrNull() }.orEmpty()
}

/**
 * RAG personal: recupera los registros de la persona que responden a su pregunta (tipo + periodo + texto)
 * y arma la respuesta con SUS números (último valor, promedio, rango, tendencia, semáforo). Sin inventar nada.
 */
object ConsultaHistorial {
    private val TIPOS = listOf(
        "pulso" to listOf("pulso", "latido", "corazon", "frecuencia cardiaca", "lpm", "ppg", "pulsera"),
        "spo2" to listOf("oxigeno", "spo2", "saturacion", "oxigenacion"),
        "anemia" to listOf("anemia", "palidez", "palido", "hemoglobina", "parpado", "palma", "encia"),
        "diario" to listOf("diario", "senti", "sentido", "sintoma", "animo", "molestia", "dolor", "fiebre", "me dolio"),
        "pvt" to listOf("pvt", "reaccion", "alerta mental", "fatiga", "cansancio", "concentracion"),
        "impacto" to listOf("golpe", "impacto", "caida", "imu"),
        "ojo" to listOf("ojo", "vista", "pterigion", "catarata", "enrojecimiento"),
        "rostro" to listOf("rostro", "cara", "ictericia", "paralisis"),
        "perfil" to listOf("peso", "presion", "grasa", "imc", "perfil", "cintura"),
    )

    /** ¿La pregunta es sobre SU historial (y no un síntoma nuevo)? */
    fun esPersonal(t0: String): Boolean {
        val t = normalizar(t0)
        return listOf("historial", "mis mediciones", "mis resultados", "mi registro", "cuando fue", "cuando me toca", "me toca",
            "recordatorio", "revision", "ultima medicion", "ultimo resultado", "ultimas", "promedio", "como ha estado", "como ha ido",
            "como va mi", "como van mis", "como esta mi", "tendencia", "alertas", "algo raro", "algo malo", "mi semana", "mi mes",
            "esta semana", "este mes", "evolucion", "he mejorado", "he empeorado", "comparado").any { it in t } ||
            Regex("\\bmis? (pulso|oxigeno|anemia|diario|presion|peso|ojo|pvt|mediciones|datos)\\b").containsMatchIn(t)
    }

    private fun ventana(t: String, ahora: Long): Pair<Long, String> {
        val d = Seguimiento.DIA
        Regex("(\\d+) dias").find(t)?.let { val n = it.groupValues[1].toLong(); return ahora - n * d to "los últimos $n días" }
        return when {
            "hoy" in t -> ahora - ahora % d - 6 * 3_600_000L to "hoy"
            "ayer" in t -> ahora - 2 * d to "desde ayer"
            "semana" in t -> ahora - 7 * d to "esta semana"
            "mes" in t -> ahora - 30 * d to "este mes"
            "ano" in t -> ahora - 365 * d to "este año"
            else -> ahora - 90 * d to "los últimos 90 días"
        }
    }

    fun responder(pregunta: String, h: List<Registro>, ahora: Long): Mensaje {
        val t = normalizar(pregunta)
        if (h.isEmpty()) return Mensaje(false, "Aún no tengo mediciones tuyas. Usa el Diario, la Biometría PPG o un tamizaje y aquí irá quedando tu historial.")
        val (desde, periodo) = ventana(t, ahora)
        val revs = Seguimiento.revisiones(h, ahora)
        val alertas = Seguimiento.alertas(h, ahora)

        // ¿Cuándo me toca? → recordatorios
        if (listOf("me toca", "recordatorio", "revision", "cuando debo", "pendiente").any { it in t }) {
            val l = revs.joinToString("\n") { "• ${it.tipo.nombre}: ${it.texto}" + (it.ultima?.let { u -> " (última ${Seguimiento.hace(u.horaMs, ahora)})" } ?: "") }
            val peor = if (revs.any { it.estado == EstadoRevision.VENCIDA }) Triage.ADVERTENCIA else Triage.NOMINAL
            return Mensaje(false, "Tus revisiones:\n$l", peor)
        }
        // ¿Tengo alertas / algo raro?
        val tipos = TIPOS.filter { (_, k) -> k.any { it in t } }.map { it.first }.toSet()
        if (tipos.isEmpty() && listOf("alerta", "algo raro", "algo malo", "como voy", "resumen", "todo").any { it in t }) {
            val txt = if (alertas.isEmpty()) "Sin alertas en tu historial. " else alertas.take(5).joinToString("\n") { "• ${it.titulo}: ${it.detalle}" } + "\n"
            val conteo = h.filter { it.horaMs >= desde }.groupBy { it.nombre }.entries.joinToString(", ") { "${it.key} ${it.value.size}" }
            return Mensaje(false, txt + "Mediciones en $periodo: ${conteo.ifBlank { "ninguna" }}.",
                Triage.peor(*alertas.map { it.triage }.toTypedArray()).takeIf { it != Triage.SIN_DATOS } ?: Triage.NOMINAL)
        }

        // Recuperar: por tipo y periodo; si no se nombró tipo, por palabras del texto (p. ej. «fiebre» en el diario)
        val raices = Texto.raices(pregunta).toSet()
        val candidatos = h.filter { it.horaMs >= desde }
        val elegidos = if (tipos.isNotEmpty()) candidatos.filter { it.tipo in tipos }
        else candidatos.filter { r -> Texto.raices(r.texto + " " + r.nombre).any { it in raices } }
        // palabras concretas dentro del diario («fiebre», «cabeza»): solo las entradas que las mencionan
        val delDiario = candidatos.filter { it.tipo == "diario" && Texto.raices(it.texto).any { w -> w in raices && w !in setOf("diari", "senti", "dolor") } }
        if (elegidos.isEmpty() && delDiario.isEmpty())
            return Mensaje(false, "No encontré mediciones de eso en $periodo. Tienes ${h.size} registros en total; prueba «¿cómo va mi pulso este mes?» o «¿cuándo me toca revisión?».")

        val partes = mutableListOf<String>()
        for ((tipo, l0) in elegidos.groupBy { it.tipo }) {
            val l = l0.sortedBy { it.horaMs }
            val ult = l.last()
            val nombre = ult.nombre
            val vals = l.mapNotNull { it.valor }
            val sb = StringBuilder("$nombre — ${l.size} registro(s) en $periodo. Último: ${Seguimiento.fmt(ult)} (${Seguimiento.hace(ult.horaMs, ahora)}, ${ult.triage.etiqueta}).")
            if (vals.size >= 2) {
                val u = ult.unidad
                val f: (Double) -> String = { if (u == "%prob") Seguimiento.pct(it) else "${(it * 10).roundToInt() / 10.0} $u".trim() }
                sb.append("\n  Promedio ${f(vals.average())}, mínimo ${f(vals.min())}, máximo ${f(vals.max())}.")
                if (vals.size >= 4) {
                    val m = vals.size / 2
                    val d = vals.drop(m).average() - vals.take(m).average()
                    val umbral = if (u == "%prob") 0.08 else if (tipo == "pulso") 5.0 else if (tipo == "spo2") 1.0 else abs(vals.average()) * 0.1
                    sb.append(" Tendencia: " + when { d > umbral -> "subiendo (${f(vals.take(m).average())} → ${f(vals.drop(m).average())})"
                        d < -umbral -> "bajando (${f(vals.take(m).average())} → ${f(vals.drop(m).average())})"; else -> "estable" } + ".")
                }
            }
            val porColor = l.groupingBy { it.triage }.eachCount()
            if (l.size >= 2) sb.append("\n  Semáforo: " + listOf(Triage.NOMINAL to "verde", Triage.ADVERTENCIA to "amarillo", Triage.CRITICO to "rojo")
                .filter { (porColor[it.first] ?: 0) > 0 }.joinToString(", ") { "${porColor[it.first]} ${it.second}" } + ".")
            if (tipo == "diario") l.filter { it.triage != Triage.NOMINAL }.takeLast(3).forEach { sb.append("\n  · ${Seguimiento.hace(it.horaMs, ahora)}: «${it.texto.take(90)}»") }
            revs.firstOrNull { tipo in it.tipo.tipos }?.let { sb.append("\n  Próxima revisión: ${it.texto.lowercase()}.") }
            partes += sb.toString()
        }
        if (tipos.isEmpty() && delDiario.isNotEmpty() && elegidos.none { it.tipo == "diario" }) partes += "En tu diario lo mencionaste ${delDiario.size} vez/veces: " +
            delDiario.takeLast(3).joinToString("; ") { "${Seguimiento.hace(it.horaMs, ahora)} «${it.texto.take(70)}»" }
        val rel = alertas.filter { al -> tipos.isEmpty() || tipos.any { tp -> tp in al.id } }
        if (rel.isNotEmpty()) partes += "Alertas: " + rel.take(3).joinToString("; ") { it.titulo }
        val tri = Triage.peor(*(elegidos + delDiario).filter { ahora - it.horaMs <= 7 * Seguimiento.DIA }.map { it.triage }.toTypedArray(),
            *rel.map { it.triage }.toTypedArray()).takeIf { it != Triage.SIN_DATOS } ?: Triage.NOMINAL
        return Mensaje(false, partes.joinToString("\n\n"), tri)
    }
}
