package com.temrarios.ojoscan.astro

import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import kotlin.math.roundToInt

/** Una línea del resumen para el médico: el PDF y el texto se arman con esto. */
data class LineaReporte(val tipo: Tipo, val texto: String, val triage: Triage? = null) {
    enum class Tipo { TITULO, SUBTITULO, SECCION, TEXTO, VIÑETA, NOTA }
}

/** Respaldo completo (se cifra con contraseña en la plataforma). */
@Serializable
data class Respaldo(val version: Int = 1, val creadoMs: Long, val historial: List<Registro>, val diario: List<EntradaDiario>, val perfil: String? = null)

object ReporteSalud {
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = false }
    fun respaldoJson(r: Respaldo): String = json.encodeToString(Respaldo.serializer(), r)
    fun respaldoDe(s: String): Respaldo? = runCatching { json.decodeFromString(Respaldo.serializer(), s) }.getOrNull()

    /** Resumen de los últimos [dias] días, pensado para leerse en 1–2 minutos en consulta. */
    fun lineas(h: List<Registro>, ahora: Long, fecha: (Long) -> String, dias: Int = 30): List<LineaReporte> {
        val D = Seguimiento.DIA
        val desde = ahora - dias * D
        val per = h.filter { it.horaMs >= desde }
        val l = mutableListOf(
            LineaReporte(LineaReporte.Tipo.TITULO, "Resumen de salud · AstroPulse"),
            LineaReporte(LineaReporte.Tipo.SUBTITULO, "Del ${fecha(desde).take(5)} al ${fecha(ahora).take(5)} · ${per.size} registros · generado en el teléfono"),
        )
        val alertas = Seguimiento.alertas(h, ahora).filter { it.triage != Triage.NOMINAL }
        l += LineaReporte(LineaReporte.Tipo.SECCION, "Alertas activas")
        if (alertas.isEmpty()) l += LineaReporte(LineaReporte.Tipo.TEXTO, "Ninguna.")
        alertas.forEach { l += LineaReporte(LineaReporte.Tipo.VIÑETA, "${it.titulo}: ${it.detalle}", it.triage) }

        l += LineaReporte(LineaReporte.Tipo.SECCION, "Mediciones")
        val grupos = per.filter { it.tipo != "diario" }.groupBy { it.tipo }
        if (grupos.isEmpty()) l += LineaReporte(LineaReporte.Tipo.TEXTO, "Sin mediciones en el periodo.")
        for ((tipo, g0) in grupos) {
            val g = g0.sortedBy { it.horaMs }
            val ult = g.last()
            val v = g.mapNotNull { it.valor }
            val fmt: (Double) -> String = { x -> if (ult.unidad == "%prob") Seguimiento.pct(x) else "${(x * 10).roundToInt() / 10.0} ${ult.unidad}".trim() }
            val colores = g.groupingBy { it.triage }.eachCount()
            val sb = StringBuilder("${ult.nombre}: ${g.size} registro(s). Último ${Seguimiento.fmt(ult)} (${fecha(ult.horaMs)}).")
            if (v.size >= 2) sb.append(" Promedio ${fmt(v.average())}, rango ${fmt(v.min())}–${fmt(v.max())}.")
            if (v.size >= 4) {
                val m = v.size / 2; val a = v.take(m).average(); val b = v.drop(m).average()
                val umbral = if (tipo == "pulso") 5.0 else if (tipo == "spo2") 1.0 else kotlin.math.abs(a) * 0.1
                sb.append(" Tendencia: ${if (b - a > umbral) "sube" else if (a - b > umbral) "baja" else "estable"}.")
            }
            sb.append(" Semáforo: ${colores[Triage.NOMINAL] ?: 0} verde, ${colores[Triage.ADVERTENCIA] ?: 0} amarillo, ${colores[Triage.CRITICO] ?: 0} rojo.")
            if (tipo == "anemia" && Seguimiento.ANEMIA_EN_PRUEBAS) sb.append(" (Experimental, no validado: confirmar con biometría hemática.)")
            l += LineaReporte(LineaReporte.Tipo.VIÑETA, sb.toString(), colores.keys.maxByOrNull { it.orden })
        }

        val diario = per.filter { it.tipo == "diario" }
        l += LineaReporte(LineaReporte.Tipo.SECCION, "Diario (${diario.size} entradas; ${diario.count { it.triage != Triage.NOMINAL }} con molestias)")
        diario.filter { it.triage != Triage.NOMINAL }.takeLast(10).forEach {
            l += LineaReporte(LineaReporte.Tipo.VIÑETA, "${fecha(it.horaMs)} · ${it.triage.etiqueta}: «${it.texto.take(160)}»", it.triage)
        }

        l += LineaReporte(LineaReporte.Tipo.SECCION, "Revisiones")
        Seguimiento.revisiones(h, ahora).forEach { l += LineaReporte(LineaReporte.Tipo.VIÑETA, "${it.tipo.nombre}: ${it.texto}") }
        l += LineaReporte(LineaReporte.Tipo.NOTA, "Datos medidos por la persona con su teléfono (cámara, pulsera, cuestionarios). Son orientativos y no sustituyen " +
            "estudios clínicos. Pulso por cámara: error medio ≈ 4.5 lpm contra ECG. Anemia por foto: experimental.")
        return l
    }

    fun texto(l: List<LineaReporte>): String = l.joinToString("\n") {
        when (it.tipo) {
            LineaReporte.Tipo.TITULO -> it.texto.uppercase()
            LineaReporte.Tipo.SECCION -> "\n== ${it.texto} =="
            LineaReporte.Tipo.VIÑETA -> "• ${it.texto}"
            else -> it.texto
        }
    }
}
