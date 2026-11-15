package com.temrarios.ojoscan.astro

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt

/** Una fuente que aportó a la probabilidad combinada. */
data class FuenteAnemia(val nombre: String, val probabilidad: Double?, val peso: Double, val detalle: String)

/** Probabilidad única de anemia juntando todas las mediciones. */
data class AnemiaCombinada(val probabilidad: Double, val fuentes: List<FuenteAnemia>, val triage: Triage) {
    val porcentaje get() = (probabilidad * 100).roundToInt()
}

/**
 * Junta los tamizajes de anemia en UNA probabilidad.
 *
 * 1. Cada foto (párpado, palma, encía) da su probabilidad; se pasa a log-momios y se pondera por lo que su
 *    modelo separa de verdad: peso = 2·(AUC − 0.5). Párpado AUC 0.78 → 0.55, palma 0.75 → 0.50,
 *    encía sin validar → 0.10 (solo apoya).
 * 2. Promedio ponderado de los log-momios = consenso. Si varias fotos de partes distintas del cuerpo
 *    coinciden, cada una es evidencia independiente y la confianza sube (+20 % por fuente extra que coincide).
 * 3. El pulso en reposo de la pulsera suma evidencia: con poca hemoglobina el corazón late más rápido.
 *
 * Solo cuentan mediciones de los últimos 30 días.
 */
object FusionAnemia {
    private const val VIGENCIA_MS = 30L * 86_400_000L
    private val AUC_POR_DEFECTO = mapOf("párpado" to 0.777, "palma" to 0.7475, "encía" to 0.55)

    private fun logit(p: Double) = ln(p.coerceIn(0.01, 0.99) / (1 - p.coerceIn(0.01, 0.99)))
    private fun sigmoide(x: Double) = 1 / (1 + exp(-x))

    fun peso(metodo: String, auc: Double?): Double {
        val a = auc ?: AUC_POR_DEFECTO[metodo] ?: 0.6
        val w = (2 * (a - 0.5)).coerceIn(0.0, 1.0)
        return if (metodo == "encía") minOf(w, 0.1) else w
    }

    fun combinar(lecturas: Collection<LecturaAnemia>, pulsoReposo: Double?, ahoraMs: Long, pulsoAnormal: Boolean = false): AnemiaCombinada? {
        val vigentes = lecturas.filter { it.probabilidad != null && ahoraMs - it.horaMs <= VIGENCIA_MS }
        if (vigentes.isEmpty()) return null
        val fuentes = mutableListOf<FuenteAnemia>()
        var suma = 0.0
        var pesos = 0.0
        for (l in vigentes) {
            val w = peso(l.metodo, l.auc)
            suma += w * logit(l.probabilidad!!)
            pesos += w
            fuentes += FuenteAnemia(l.metodo.replaceFirstChar { it.uppercase() }, l.probabilidad, w,
                if (l.metodo == "encía") "experimental, apoyo" else "AUC ${((l.auc ?: AUC_POR_DEFECTO[l.metodo] ?: 0.6) * 100).roundToInt() / 100.0}")
        }
        var x = if (pesos > 0) suma / pesos else 0.0
        // coincidencia entre fuentes validadas (párpado y palma)
        val validadas = vigentes.filter { it.metodo != "encía" }
        if (validadas.size >= 2 && validadas.all { it.probabilidad!! >= 0.5 } || validadas.size >= 2 && validadas.all { it.probabilidad!! < 0.5 })
            x *= 1 + 0.2 * (validadas.size - 1)
        // pulso en reposo de la pulsera
        if (pulsoReposo != null) {
            val ajuste = when {
                pulsoReposo >= 100 -> 0.5
                pulsoReposo >= 90 -> 0.25
                pulsoReposo < 65 -> -0.15
                else -> 0.0
            }
            x += ajuste
            fuentes += FuenteAnemia("Pulso en reposo", null, kotlin.math.abs(ajuste), "${pulsoReposo.roundToInt()} lpm" +
                when { ajuste > 0 -> " · alto, sube la probabilidad"; ajuste < 0 -> " · tranquilo, la baja un poco"; else -> " · normal" })
        }
        val p = sigmoide(x)
        val t = when {
            p >= 0.5 && pulsoAnormal -> Triage.CRITICO
            p >= 0.5 -> Triage.ADVERTENCIA
            else -> Triage.NOMINAL
        }
        return AnemiaCombinada(p, fuentes.sortedByDescending { it.peso }, t)
    }
}
