package com.temrarios.ojoscan.historial

import androidx.compose.runtime.Composable
import kotlinx.coroutines.flow.StateFlow
import kotlin.math.roundToInt

/**
 * Historial de salud de los últimos días desde Health Connect (Android), donde Mi Fitness (Xiaomi Smart
 * Band) escribe pulso, oxígeno, sueño y pasos. Es lo que el correo mensual de Xiaomi trae, pero al día.
 * Todo se resume en el teléfono; al servidor solo viajan los promedios.
 */
sealed interface EstadoHistorial {
    data object Inactivo : EstadoHistorial
    /** Health Connect no está instalado / no es compatible. */
    data object NoDisponible : EstadoHistorial
    data object SinPermiso : EstadoHistorial
    data object Leyendo : EstadoHistorial
    data class Listo(val resumen: ResumenHistorial) : EstadoHistorial
    /** Hay permiso pero Health Connect está vacío: Mi Fitness no está sincronizando. */
    data object SinDatos : EstadoHistorial
    data class Error(val mensaje: String) : EstadoHistorial
}

interface FuenteHistorial {
    val estado: StateFlow<EstadoHistorial>
    /** Pide permisos si faltan y lee los últimos [dias] días. */
    fun conectar(dias: Int = 30)
    /** Lee en silencio si ya hay permiso (sin pedir nada). Se usa al abrir la app y al empezar cada chequeo. */
    fun leerSiHayPermiso(dias: Int = 30)
    /** Abre Health Connect para que la persona active Mi Fitness ahí. */
    fun abrirAjustes()
}

@Composable
expect fun rememberFuenteHistorial(): FuenteHistorial

/** Un valor por día (o por noche). */
data class Diario(val dia: String, val valor: Double)

data class ResumenHistorial(
    val dias: Int,
    /** Pulso en reposo por día (RestingHeartRate de la pulsera o el percentil 10 del día). */
    val pulsoReposo: List<Diario> = emptyList(),
    val pulsoMax: Int? = null,
    /** Lecturas de oxígeno en sangre (%). */
    val spo2: List<Double> = emptyList(),
    /** Horas dormidas por noche y fracción de sueño profundo. */
    val sueno: List<Diario> = emptyList(),
    val profundo: Double? = null,
    val pasos: List<Diario> = emptyList(),
    /** VFC nocturna (RMSSD, ms) si la pulsera la escribe. */
    val vfc: List<Double> = emptyList(),
) {
    val vacio get() = pulsoReposo.isEmpty() && spo2.isEmpty() && sueno.isEmpty() && pasos.isEmpty()
}

private fun List<Double>.media() = if (isEmpty()) null else average()
private fun r1(x: Double?) = x?.let { (it * 10).roundToInt() / 10.0 }

/** Lo que se manda al servidor (claves hist_* de /predict/perfil). */
data class HistorialParaServidor(
    val dias: Int,
    val pulsoReposo: Double?, val pulsoReposo7d: Double?, val pulsoReposoBase: Double?, val pulsoMax: Int?,
    val spo2Media: Double?, val spo2Min: Double?, val spo2Bajas: Double?, val spo2N: Int,
    val suenoHoras: Double?, val suenoNoches: Int, val suenoCortas: Double?, val profundo: Double?,
    val pasos: Double?, val pasosDias: Int, val vfc: Double?,
)

fun ResumenHistorial.paraServidor(): HistorialParaServidor {
    val pr = pulsoReposo.sortedBy { it.dia }
    val ult7 = pr.takeLast(7).map { it.valor }
    val base = pr.dropLast(7).map { it.valor }
    val horas = sueno.map { it.valor }.filter { it >= 2.0 } // siestas fuera
    val pasosD = pasos.map { it.valor }.filter { it > 300 } // días sin la pulsera puesta fuera
    return HistorialParaServidor(
        dias = dias,
        pulsoReposo = r1(pr.map { it.valor }.media()),
        pulsoReposo7d = r1(ult7.takeIf { it.size >= 3 }?.media()),
        pulsoReposoBase = r1(base.takeIf { it.size >= 7 }?.media()),
        pulsoMax = pulsoMax,
        spo2Media = r1(spo2.media()), spo2Min = spo2.minOrNull(),
        spo2Bajas = if (spo2.size >= 10) (spo2.count { it < 90 }.toDouble() / spo2.size).let { (it * 1000).roundToInt() / 1000.0 } else null,
        spo2N = spo2.size,
        suenoHoras = r1(horas.media()), suenoNoches = horas.size,
        suenoCortas = if (horas.size >= 5) (horas.count { it < 6 }.toDouble() / horas.size).let { (it * 100).roundToInt() / 100.0 } else null,
        profundo = profundo?.let { (it * 100).roundToInt() / 100.0 },
        pasos = pasosD.media()?.roundToInt()?.toDouble(), pasosDias = pasosD.size,
        vfc = r1(vfc.media()),
    )
}
