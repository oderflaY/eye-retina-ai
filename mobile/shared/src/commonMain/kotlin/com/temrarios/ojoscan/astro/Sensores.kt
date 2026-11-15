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
 * Recibe el promedio del canal rojo de cada cuadro; filtra, busca picos y calcula lpm, índice de perfusión
 * (componente pulsátil / continua) y calidad (regularidad de los intervalos).
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

    fun limpiar() { t.clear(); v.clear() }

    fun estimar(horaMs: Long): LecturaPpg? {
        if (v.size < 60 || t.last() - t.first() < 6000) return null
        val x = v.toList()
        val tt = t.toList()
        // quitar tendencia con media móvil de ~1 s y suavizar con 3 puntos
        val fps = x.size * 1000.0 / (tt.last() - tt.first())
        val k = (fps).toInt().coerceAtLeast(3)
        val sinTend = x.indices.map { i ->
            val a = (i - k / 2).coerceAtLeast(0); val b = (i + k / 2).coerceAtMost(x.size - 1)
            x[i] - x.subList(a, b + 1).average()
        }
        val s = sinTend.indices.map { i -> sinTend.subList((i - 1).coerceAtLeast(0), (i + 2).coerceAtMost(x.size)).average() }
        // en PPG por reflexión la sangre ABSORBE: el latido es un valle del rojo → se invierte
        val y = s.map { -it }
        val amp = y.maxOrNull()!! - y.minOrNull()!!
        if (amp <= 0) return null
        val minDist = (fps * 0.33).toInt().coerceAtLeast(2) // ≤ 180 lpm
        val picos = mutableListOf<Int>()
        for (i in 1 until y.size - 1) {
            if (y[i] > y[i - 1] && y[i] >= y[i + 1] && y[i] > amp * 0.15) {
                if (picos.isEmpty() || i - picos.last() >= minDist) picos += i
                else if (y[i] > y[picos.last()]) picos[picos.size - 1] = i
            }
        }
        if (picos.size < 4) return null
        val intervalos = picos.zipWithNext { a, b -> (tt[b] - tt[a]).toDouble() }.filter { it in 330.0..1500.0 }
        if (intervalos.size < 3) return null
        val med = intervalos.sorted()[intervalos.size / 2]
        val dispersion = intervalos.map { abs(it - med) / med }.average()
        val lpm = (60_000 / med).roundToInt()
        val dc = x.average()
        val perfusion = if (dc > 0) (x.max() - x.min()) / dc * 100 else 0.0
        val calidad = (1 - dispersion * 4).coerceIn(0.0, 1.0)
        return LecturaPpg(lpm, (perfusion * 100).roundToInt() / 100.0, (calidad * 100).roundToInt() / 100.0, horaMs)
    }
}
