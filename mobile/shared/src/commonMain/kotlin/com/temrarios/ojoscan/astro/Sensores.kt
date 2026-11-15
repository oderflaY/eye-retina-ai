package com.temrarios.ojoscan.astro

import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Caja negra inercial: separa la gravedad (filtro paso bajo) de la aceleración dinámica y detecta impactos.
 * Así funciona igual en la Tierra que en microgravedad: solo cuenta el golpe, no el peso.
 */
class DetectorImpactos(private val alfa: Double = 0.9) {
    private var gx = 0.0; private var gy = 0.0; private var gz = 0.0
    private var iniciado = false
    private var ultimoImpactoMs = 0L

    /**
     * Umbral de impacto severo en g según el entorno: en microgravedad un golpe de 2.5 g ya es un choque contra
     * la estructura; con más gravedad las caídas son más fuertes y se tolera más antes de alertar.
     */
    fun umbral(entorno: Entorno) = 2.5 + 1.5 * entorno.g

    /** Devuelve la aceleración dinámica (g) y el impacto si lo hubo. Entrada en m/s². */
    fun muestra(x: Double, y: Double, z: Double, horaMs: Long, entorno: Entorno): Pair<Double, Impacto?> {
        if (!iniciado) { gx = x; gy = y; gz = z; iniciado = true }
        gx = alfa * gx + (1 - alfa) * x
        gy = alfa * gy + (1 - alfa) * y
        gz = alfa * gz + (1 - alfa) * z
        val dx = x - gx; val dy = y - gy; val dz = z - gz
        val dinamica = sqrt(dx * dx + dy * dy + dz * dz) / 9.80665
        val u = umbral(entorno)
        val impacto = if (dinamica >= u * 0.6 && horaMs - ultimoImpactoMs > 1500) {
            ultimoImpactoMs = horaMs
            Impacto(horaMs, (dinamica * 100).roundToInt() / 100.0, dinamica >= u)
        } else null
        return dinamica to impacto
    }
}

/**
 * Prueba de vigilancia psicomotora (PVT, Dinges & Powell 1985), versión corta de 3 min de NASA:
 * estímulos con espera aleatoria de 2-10 s; lapso = reacción ≥ 500 ms; adelanto = tocar antes del estímulo.
 */
class MotorPvt(private val azar: () -> Double = { kotlin.random.Random.nextDouble() }) {
    enum class Fase { LISTO, ESPERANDO, ESTIMULO, TERMINADO }

    var fase = Fase.LISTO; private set
    var esperaMs = 0L; private set
    val reacciones = mutableListOf<Int>()
    var adelantos = 0; private set
    private var inicioEstimulo = 0L
    var ensayos = 10

    /** Arranca un ensayo y devuelve cuánto esperar antes de mostrar el estímulo. */
    fun siguiente(): Long {
        fase = Fase.ESPERANDO
        esperaMs = (2000 + azar() * 8000).toLong()
        return esperaMs
    }

    fun mostrar(ahoraMs: Long) { fase = Fase.ESTIMULO; inicioEstimulo = ahoraMs }

    /** Milisegundos desde que apareció el estímulo (el contador que ve el astronauta). */
    fun transcurrido(ahoraMs: Long): Long = ahoraMs - inicioEstimulo

    /** El astronauta toca la pantalla. Devuelve el tiempo de reacción o null si fue adelanto. */
    fun tocar(ahoraMs: Long): Int? = when (fase) {
        Fase.ESPERANDO -> { adelantos++; null }
        Fase.ESTIMULO -> {
            val ms = (ahoraMs - inicioEstimulo).toInt().coerceAtLeast(0)
            reacciones += ms
            fase = if (reacciones.size >= ensayos) Fase.TERMINADO else Fase.LISTO
            ms
        }
        else -> null
    }

    fun reiniciar() { reacciones.clear(); adelantos = 0; fase = Fase.LISTO }

    fun resultado(horaMs: Long): ResultadoPvt? {
        if (reacciones.isEmpty()) return null
        val ord = reacciones.sorted()
        val lapsos = reacciones.count { it >= 500 }
        val media = reacciones.average().roundToInt()
        // Basner 2011 (PVT-B): media > 355 ms o ≥ 2 lapsos en 3 min = rendimiento alterado
        val triage = when {
            media >= 450 || lapsos >= 3 || adelantos >= 4 -> Triage.CRITICO
            media >= 355 || lapsos >= 1 || adelantos >= 2 -> Triage.ADVERTENCIA
            else -> Triage.NOMINAL
        }
        return ResultadoPvt(reacciones.toList(), lapsos, adelantos, media, ord[ord.size / 2], triage, horaMs)
    }
}

/**
 * Pulso por fotopletismografía con la cámara: dedo sobre el lente con el flash encendido.
 * Recibe el promedio del canal rojo de cada cuadro; re-muestrea a 30 Hz y usa [ModeloPpg] (BUT PPG):
 * pulso por espectro + picos con prominencia y calidad por regresión logística (13 rasgos).
 */
class EstimadorPpg(private val ventanaMs: Long = 12_000) {
    private val t = ArrayDeque<Long>()
    private val v = ArrayDeque<Double>()
    val senal: List<Double> get() = v.toList()

    /** ¿Hay un dedo tapando el lente? (cuadro rojo saturado y poco verde) */
    fun dedoPresente(rojo: Double, verde: Double) = rojo > 120 && verde < rojo * 0.6

    fun agregar(horaMs: Long, rojo: Double) {
        t.addLast(horaMs); v.addLast(rojo)
        while (t.isNotEmpty() && horaMs - t.first() > ventanaMs) { t.removeFirst(); v.removeFirst() }
    }

    fun limpiar() { t.clear(); v.clear(); ultimoMs = Long.MIN_VALUE }

    /** Último análisis completo (rasgos y probabilidad), para depurar o mostrar. */
    var ultimo: ModeloPpg.Analisis? = null
        private set

    /**
     * Pulso + calidad con el modelo entrenado en BUT PPG (ver [ModeloPpg]). Necesita 10 s de señal.
     * calidad ≥ 0.5 = el modelo cree que el pulso está a ±5 lpm del real; si no, hay que seguir midiendo.
     */
    private var ultimoMs = Long.MIN_VALUE

    fun estimar(horaMs: Long): LecturaPpg? {
        // se llama en cada cuadro de la cámara: analizar como máximo 1 vez por segundo
        if (ultimoMs != Long.MIN_VALUE && horaMs - ultimoMs < 1_000) return null
        val x = ModeloPpg.remuestrear(t.toList(), v.toList()) ?: return null
        if (x.max() - x.min() <= 0) return null
        ultimoMs = horaMs
        val a = ModeloPpg.analizar(x)
        ultimo = a
        val dc = x.average()
        val perfusion = if (dc > 0) (x.max() - x.min()) / dc * 100 else 0.0
        return LecturaPpg(a.hr.roundToInt(), (perfusion * 100).roundToInt() / 100.0, (a.calidad * 100).roundToInt() / 100.0, horaMs)
    }
}
