package com.temrarios.ojoscan.astro

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** Triage clínico: verde nominal, ámbar advertencia, rojo crítico. */
enum class Triage(val etiqueta: String, val orden: Int) {
    NOMINAL("NOMINAL", 0), ADVERTENCIA("ADVERTENCIA", 1), CRITICO("CRÍTICO", 2), SIN_DATOS("SIN DATOS", -1);

    companion object {
        fun peor(vararg t: Triage): Triage = t.filter { it != SIN_DATOS }.maxByOrNull { it.orden } ?: SIN_DATOS
    }
}

/**
 * Entorno de misión. La gravedad cambia el umbral de impacto de la IMU y la dosis de radiación cambia la
 * caducidad de los fármacos. Dosis diarias medidas en superficie/órbita: ISS ≈ 0.5 mSv/d (ISS-RAD),
 * Luna ≈ 1.37 mSv/d (Chang'e-4 LND, 2020), Marte ≈ 0.67 mSv/d (Curiosity RAD, 2013).
 */
enum class Entorno(val nombre: String, val corto: String, val gravedad: Double, val dosisMsvDia: Double) {
    ISS("Estación Espacial Internacional", "ISS", 0.0, 0.5),
    LUNA("Base Lunar", "LUNA", 1.62, 1.37),
    MARTE("Superficie de Marte", "MARTE", 3.71, 0.67);

    /** Gravedad en g terrestres. */
    val g: Double get() = gravedad / 9.80665
}

/**
 * Retraso de la señal (solo ida) con la Tierra, en segundos.
 * ISS: relevo por satélites TDRS (~0.6 s). Luna: 384 400 km / c. Marte: distancia real del día con órbitas
 * circulares coplanares (error de unos minutos, suficiente para planear comunicaciones).
 */
object Latencia {
    private const val UA_SEG_LUZ = 499.005
    private const val J2000_MS = 946_728_000_000L // 2000-01-01T12:00Z

    fun segundos(entorno: Entorno, ahoraMs: Long): Double = when (entorno) {
        Entorno.ISS -> 0.6
        Entorno.LUNA -> 384_400.0 / 299_792.458
        Entorno.MARTE -> distanciaMarteUa(ahoraMs) * UA_SEG_LUZ
    }

    fun distanciaMarteUa(ahoraMs: Long): Double {
        val dias = (ahoraMs - J2000_MS) / 86_400_000.0
        val lt = rad(100.464 + 0.985_609_1 * dias)   // longitud media de la Tierra
        val lm = rad(355.453 + 0.524_033_1 * dias)   // longitud media de Marte
        val rt = 1.0
        val rm = 1.523_68
        return sqrt(rt * rt + rm * rm - 2 * rt * rm * cos(lm - lt))
    }

    /** "+14m 22s", "+1.3 s"… */
    fun texto(seg: Double): String =
        if (seg < 60) "+${(seg * 10).roundToInt() / 10.0} s"
        else "+${(seg / 60).toInt()}m ${(seg % 60).toInt().toString().padStart(2, '0')}s"

    /** Con más de 2 s de ida la tripulación decide sola (no hay control de misión en tiempo real). */
    fun autonomo(seg: Double) = seg > 2.0

    private fun rad(g: Double) = (g % 360.0) * PI / 180.0
}

data class ResultadoPvt(
    val reacciones: List<Int>,
    val lapsos: Int,
    val adelantos: Int,
    val mediaMs: Int,
    val medianaMs: Int,
    val triage: Triage,
    val horaMs: Long,
)

data class Impacto(val horaMs: Long, val g: Double, val severo: Boolean)

/** Último tamizaje de anemia con la cámara (párpado, palma o encía). */
data class LecturaAnemia(val metodo: String, val resultado: String, val titulo: String, val probabilidad: Double?, val auc: Double?, val horaMs: Long)

data class LecturaPpg(val lpm: Int, val perfusion: Double, val calidad: Double, val horaMs: Long)

/** Estado global que pinta la pantalla. */
data class MissionTelemetry(
    val entorno: Entorno = Entorno.ISS,
    val latenciaSeg: Double = 0.6,
    /** SpO2 óptico (%): de la pulsera vía Health Connect. */
    val spo2: Double? = null,
    val spo2Fuente: String = "",
    /** Pulso más reciente: pulsera en vivo o PPG con la cámara. */
    val pulso: Int? = null,
    val pulsoFuente: String = "",
    /** Aceleración dinámica actual (g) sin la gravedad. */
    val imuG: Double = 0.0,
    val imuPicoG: Double = 0.0,
    val impactos: List<Impacto> = emptyList(),
    val pvt: ResultadoPvt? = null,
    val ppg: LecturaPpg? = null,
    /** Último tamizaje por método (párpado, palma, encía). */
    val anemias: Map<String, LecturaAnemia> = emptyMap(),
    /** Probabilidad única de anemia juntando todo. */
    val anemia: AnemiaCombinada? = null,
    /** Pulso en reposo de 30 días (pulsera vía Health Connect). */
    val pulsoReposo: Double? = null,
    val estres: Triage = Triage.SIN_DATOS,
    /** Semáforo del diario de salud (última entrada de las últimas 12 h). */
    val diario: Triage = Triage.SIN_DATOS,
    val triageSpo2: Triage = Triage.SIN_DATOS,
    val triageImu: Triage = Triage.NOMINAL,
    val triage: Triage = Triage.SIN_DATOS,
    val diaMision: Int = 1,
    val dosisAcumuladaMsv: Double = 0.0,
)
