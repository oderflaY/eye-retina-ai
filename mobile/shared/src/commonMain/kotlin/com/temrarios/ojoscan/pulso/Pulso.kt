package com.temrarios.ojoscan.pulso

import androidx.compose.runtime.Composable
import kotlinx.coroutines.flow.StateFlow
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Pulso EN TIEMPO REAL de una pulsera o reloj por Bluetooth (servicio estándar Heart Rate 0x180D).
 * Probado con la Xiaomi Smart Band 10: hay que activar en la pulsera "Transmitir frecuencia cardíaca"
 * (en Mi Fitness: Dispositivo → Frecuencia cardíaca → Compartir/transmitir frecuencia cardíaca).
 * No depende de la nube de Xiaomi: la app lee la pulsera directamente, latido a latido.
 */
data class Muestra(val tiempoMs: Long, val lpm: Int, /** intervalos entre latidos (ms), si la pulsera los manda */ val rr: List<Int> = emptyList())

sealed interface EstadoPulso {
    data object Inactivo : EstadoPulso
    data object SinPermiso : EstadoPulso
    data object BluetoothApagado : EstadoPulso
    data class Buscando(val detalle: String) : EstadoPulso
    data class Conectando(val nombre: String) : EstadoPulso
    data class Midiendo(val nombre: String, val lpm: Int, val contacto: Boolean?) : EstadoPulso
    /** Conectó con la pulsera pero no expone el pulso: falta activar la transmisión. */
    data class SinTransmision(val nombre: String) : EstadoPulso
    data class Error(val mensaje: String) : EstadoPulso
}

interface MonitorPulso {
    val estado: StateFlow<EstadoPulso>
    /** Todas las lecturas desde que se conectó (la sesión del chequeo). */
    val muestras: StateFlow<List<Muestra>>
    fun iniciar()
    fun detener()
    fun reiniciarSesion()
}

@Composable
expect fun rememberMonitorPulso(): MonitorPulso

/**
 * Característica 0x2A37 (Heart Rate Measurement), formato del estándar Bluetooth:
 * byte 0 = banderas (bit0: pulso en 16 bits; bit1-2: contacto con la piel; bit3: energía; bit4: hay RR),
 * luego el pulso, la energía (2 bytes) si viene, y los RR en 1/1024 s.
 */
fun parsearHrm(b: ByteArray): Triple<Int, Boolean?, List<Int>>? {
    if (b.size < 2) return null
    val f = b[0].toInt() and 0xFF
    var i = 1
    val lpm: Int
    if (f and 0x01 != 0) {
        if (b.size < 3) return null
        lpm = (b[1].toInt() and 0xFF) or ((b[2].toInt() and 0xFF) shl 8); i = 3
    } else {
        lpm = b[1].toInt() and 0xFF; i = 2
    }
    val contacto = if (f and 0x04 != 0) (f and 0x02 != 0) else null
    if (f and 0x08 != 0) i += 2
    val rr = mutableListOf<Int>()
    if (f and 0x10 != 0) while (i + 1 < b.size) {
        val v = (b[i].toInt() and 0xFF) or ((b[i + 1].toInt() and 0xFF) shl 8)
        rr += (v * 1000.0 / 1024.0).roundToInt(); i += 2
    }
    return Triple(lpm, contacto, rr)
}

/** Resumen de la sesión que se manda al servidor junto con "Tus datos". */
data class ResumenPulso(
    val reposo: Int,
    val minimo: Int,
    val maximo: Int,
    val promedio: Int,
    val segundos: Int,
    val muestras: Int,
    /** RMSSD (ms) de los intervalos entre latidos; null si la pulsera no manda RR. */
    val rmssd: Double?,
    /** Fracción de latidos con salto > 20 % respecto al anterior (posible ritmo irregular). */
    val irregular: Double?,
)

/**
 * Pulso en reposo = mediana de los 30 s más tranquilos (el minuto con menos pulso), para no tomar
 * el momento en que la persona se movió para la foto.
 */
fun resumir(m: List<Muestra>): ResumenPulso? {
    val v = m.filter { it.lpm in 30..230 }
    if (v.size < 10) return null
    val dur = ((v.last().tiempoMs - v.first().tiempoMs) / 1000).toInt()
    val ventana = 30_000L
    var reposo = v.map { it.lpm }.sorted()[v.size / 2]
    if (dur > 40) {
        var mejor = Int.MAX_VALUE
        var ini = 0
        for (fin in v.indices) {
            while (v[fin].tiempoMs - v[ini].tiempoMs > ventana) ini++
            if (v[fin].tiempoMs - v[ini].tiempoMs >= ventana * 0.8) {
                val med = v.subList(ini, fin + 1).map { it.lpm }.sorted().let { it[it.size / 2] }
                if (med < mejor) mejor = med
            }
        }
        if (mejor != Int.MAX_VALUE) reposo = mejor
    }
    val rr = v.flatMap { it.rr }.filter { it in 300..2000 }
    val dif = rr.zipWithNext { a, b -> (b - a).toDouble() }
    val rmssd = if (dif.size >= 20) sqrt(dif.sumOf { it * it } / dif.size) else null
    val irregular = if (rr.size >= 30) rr.zipWithNext { a, b -> abs(b - a) > 0.2 * a }.count { it }.toDouble() / (rr.size - 1) else null
    return ResumenPulso(
        reposo = reposo, minimo = v.minOf { it.lpm }, maximo = v.maxOf { it.lpm },
        promedio = v.map { it.lpm }.average().roundToInt(), segundos = dur, muestras = v.size,
        rmssd = rmssd?.let { (it * 10).roundToInt() / 10.0 }, irregular = irregular?.let { (it * 1000).roundToInt() / 1000.0 },
    )
}
