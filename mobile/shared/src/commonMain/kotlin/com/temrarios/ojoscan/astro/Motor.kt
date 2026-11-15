package com.temrarios.ojoscan.astro

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.temrarios.ojoscan.platform.Almacen
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.time.Clock
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
fun ahoraMs(): Long = Clock.System.now().toEpochMilliseconds()

private const val DIA_MS = 86_400_000L

/**
 * Motor de AstroPulse: estado global reactivo de la misión. Todo se calcula en el teléfono (offline-first);
 * la pulsera, la cámara, la IMU y el PVT le mandan lecturas y él recalcula el triage.
 */
class AstroPulseEngine(private val almacen: Almacen, private val reloj: () -> Long = ::ahoraMs) : ViewModel() {
    private val _estado = MutableStateFlow(MissionTelemetry())
    val estado: StateFlow<MissionTelemetry> = _estado.asStateFlow()

    private val _chat = MutableStateFlow(listOf(Mensaje(false,
        "Asistente de a bordo en línea, sin conexión a la Tierra. Dime qué sientes o pregúntame por un fármaco, la radiación o tu estado.")))
    val chat: StateFlow<List<Mensaje>> = _chat.asStateFlow()

    val pvt = MotorPvt()
    private val imu = DetectorImpactos()
    private var ultimaImuMs = 0L
    private var picoVentana = 0.0

    private var inicioMs = almacen.leer("astro_inicio")?.toLongOrNull() ?: reloj().also { almacen.guardar("astro_inicio", it.toString()) }
    private var dosis = almacen.leer("astro_dosis")?.toDoubleOrNull() ?: 0.0
    private var dosisMs = almacen.leer("astro_dosis_ms")?.toLongOrNull() ?: reloj()

    init {
        val e = almacen.leer("astro_entorno")?.let { n -> Entorno.entries.firstOrNull { it.name == n } } ?: Entorno.ISS
        _estado.update { it.copy(entorno = e, anemias = anemiasGuardadas()) }
        tic()
        // la latencia con Marte y la dosis cambian con el tiempo: recalcular cada 30 s
        viewModelScope.launch { while (true) { delay(30_000); tic() } }
    }

    /** Recalcula latencia, día de misión y dosis acumulada. */
    fun tic() {
        val ahora = reloj()
        val e = _estado.value.entorno
        dosis += (ahora - dosisMs).coerceAtLeast(0) / DIA_MS.toDouble() * e.dosisMsvDia
        dosisMs = ahora
        almacen.guardar("astro_dosis", dosis.toString()); almacen.guardar("astro_dosis_ms", dosisMs.toString())
        recalcular {
            it.copy(latenciaSeg = Latencia.segundos(e, ahora), diaMision = ((ahora - inicioMs) / DIA_MS).toInt() + 1,
                dosisAcumuladaMsv = dosis)
        }
    }

    // ── Entorno ──────────────────────────────────────────────────────────────
    fun cambiarEntorno(e: Entorno) {
        tic() // cierra la dosis del entorno anterior
        almacen.guardar("astro_entorno", e.name)
        recalcular { it.copy(entorno = e, latenciaSeg = Latencia.segundos(e, reloj())) }
    }

    /** Fija el día de misión (para planear: la caducidad de los fármacos se recalcula). */
    fun fijarDiaMision(dia: Int) {
        val d = dia.coerceIn(1, 3000)
        inicioMs = reloj() - (d - 1) * DIA_MS
        dosis = (d - 1) * _estado.value.entorno.dosisMsvDia
        dosisMs = reloj()
        almacen.guardar("astro_inicio", inicioMs.toString())
        tic()
    }

    // ── Lecturas ─────────────────────────────────────────────────────────────
    fun pulsoPulsera(lpm: Int?) {
        if (lpm == null || lpm <= 0) return
        recalcular { it.copy(pulso = lpm, pulsoFuente = "pulsera en vivo") }
    }

    fun spo2(valor: Double?, fuente: String) {
        if (valor == null) return
        recalcular { it.copy(spo2 = (valor * 10).roundToInt() / 10.0, spo2Fuente = fuente) }
    }

    fun ppg(l: LecturaPpg) {
        recalcular { it.copy(ppg = l, pulso = if (l.calidad >= 0.5) l.lpm else it.pulso, pulsoFuente = if (l.calidad >= 0.5) "PPG cámara" else it.pulsoFuente) }
    }

    /** Acelerómetro en m/s². Los impactos se guardan siempre; la pantalla se refresca 10 veces por segundo. */
    fun muestraImu(x: Double, y: Double, z: Double) {
        val ahora = reloj()
        val (g, impacto) = imu.muestra(x, y, z, ahora, _estado.value.entorno)
        picoVentana = max(picoVentana, g)
        if (impacto == null && ahora - ultimaImuMs < 100) return
        ultimaImuMs = ahora
        val pico = picoVentana; picoVentana = 0.0
        recalcular {
            it.copy(imuG = (pico * 100).roundToInt() / 100.0, imuPicoG = max(it.imuPicoG, (pico * 100).roundToInt() / 100.0),
                impactos = if (impacto != null) (it.impactos + impacto).takeLast(50) else it.impactos)
        }
    }

    /** Resultado de un tamizaje de anemia; se junta con los demás en una sola probabilidad. */
    fun anemia(metodo: String, resultado: String, titulo: String, probabilidad: Double?, auc: Double?) {
        val l = LecturaAnemia(metodo, resultado, titulo, probabilidad, auc, reloj())
        recalcular { it.copy(anemias = it.anemias + (metodo to l)) }
        almacen.guardar("astro_anemias", _estado.value.anemias.values.joinToString("\n") {
            listOf(it.metodo, it.resultado, it.probabilidad ?: "", it.auc ?: "", it.horaMs, it.titulo.replace("|", " ")).joinToString("|")
        })
    }

    fun pulsoReposo(lpm: Double?) { if (lpm != null) recalcular { it.copy(pulsoReposo = lpm) } }

    private fun anemiasGuardadas(): Map<String, LecturaAnemia> = almacen.leer("astro_anemias").orEmpty().lines().mapNotNull { linea ->
        val c = linea.split("|")
        if (c.size < 6) null else LecturaAnemia(c[0], c[1], c[5], c[2].toDoubleOrNull(), c[3].toDoubleOrNull(), c[4].toLongOrNull() ?: 0L)
    }.associateBy { it.metodo }

    fun umbralImpacto() = imu.umbral(_estado.value.entorno)

    fun reiniciarImu() = recalcular { it.copy(imuPicoG = 0.0, impactos = emptyList()) }

    fun registrarPvt() {
        val r = pvt.resultado(reloj()) ?: return
        recalcular { it.copy(pvt = r) }
    }

    // ── Asistente ────────────────────────────────────────────────────────────
    fun preguntar(texto: String) {
        if (texto.isBlank()) return
        val r = Cdss.responder(texto, _estado.value)
        _chat.update { it + Mensaje(true, texto.trim()) + r }
    }

    // ── Baliza SOS ───────────────────────────────────────────────────────────
    /** Reporte de emergencia (JSON) con todo lo que se sabe ahora; la interfaz lo cifra y lo guarda. */
    fun reporteSos(): String {
        val t = _estado.value
        return buildJsonObject {
            put("tipo", "BALIZA_SOS")
            put("hora_ms", reloj())
            put("entorno", t.entorno.name)
            put("dia_mision", t.diaMision)
            put("latencia_s", (t.latenciaSeg * 10).roundToInt() / 10.0)
            put("triage", t.triage.name)
            putJsonObject("signos") {
                t.pulso?.let { put("pulso_lpm", it) }; put("pulso_fuente", t.pulsoFuente)
                t.spo2?.let { put("spo2", it) }; put("spo2_fuente", t.spo2Fuente)
                t.ppg?.let { put("ppg_perfusion", it.perfusion); put("ppg_calidad", it.calidad) }
            }
            putJsonObject("imu") {
                put("pico_g", t.imuPicoG)
                putJsonArray("impactos") { t.impactos.takeLast(10).forEach { i -> add(buildJsonObject { put("hora_ms", i.horaMs); put("g", i.g); put("severo", i.severo) }) } }
            }
            t.pvt?.let { p -> putJsonObject("pvt") { put("media_ms", p.mediaMs); put("lapsos", p.lapsos); put("triage", p.triage.name) } }
            t.anemia?.let { a -> putJsonObject("anemia") {
                put("probabilidad_combinada", (a.probabilidad * 1000).roundToInt() / 1000.0)
                putJsonArray("fuentes") { a.fuentes.forEach { f -> add("${f.nombre}: ${f.probabilidad?.let { p -> "${(p * 100).roundToInt()} %" } ?: f.detalle}") } }
            } }
            put("dosis_mision_msv", (t.dosisAcumuladaMsv * 100).roundToInt() / 100.0)
            putJsonArray("farmacos_caducados") { Farmacia.matriz(t.diaMision, t.entorno).filter { it.triage == Triage.CRITICO }.forEach { add(it.farmaco.nombre) } }
            putJsonArray("chat_reciente") { _chat.value.takeLast(6).forEach { add((if (it.deAstronauta) "astronauta: " else "cdss: ") + it.texto) } }
        }.toString()
    }

    // ── Triage ───────────────────────────────────────────────────────────────
    private fun recalcular(cambio: (MissionTelemetry) -> MissionTelemetry) = _estado.update { triage(cambio(it)) }

    private fun triage(t: MissionTelemetry): MissionTelemetry {
        val ahora = reloj()
        val spo2 = when (val s = t.spo2) { null -> Triage.SIN_DATOS; in 0.0..89.9 -> Triage.CRITICO; in 90.0..93.9 -> Triage.ADVERTENCIA; else -> Triage.NOMINAL }
        val pulso = when (val p = t.pulso) { null -> Triage.SIN_DATOS; !in 40..150 -> Triage.CRITICO; !in 50..110 -> Triage.ADVERTENCIA; else -> Triage.NOMINAL }
        val recientes = t.impactos.filter { ahora - it.horaMs < 10 * 60_000 }
        val imu = when { recientes.any { it.severo } -> Triage.CRITICO; recientes.isNotEmpty() -> Triage.ADVERTENCIA; else -> Triage.NOMINAL }
        // Estrés: desempeño cognitivo (PVT) + pulso elevado en reposo
        val estres = Triage.peor(t.pvt?.triage ?: Triage.SIN_DATOS,
            if (t.pulso != null && t.pulso > 100) Triage.ADVERTENCIA else if (t.pulso != null) Triage.NOMINAL else Triage.SIN_DATOS)
        // Todas las mediciones de anemia en una probabilidad; con pulso anormal el corazón está compensando
        val combinada = FusionAnemia.combinar(t.anemias.values, t.pulsoReposo, ahora,
            pulsoAnormal = pulso == Triage.ADVERTENCIA || pulso == Triage.CRITICO)
        val anemia = combinada?.triage ?: Triage.SIN_DATOS
        val general = Triage.peor(spo2, pulso, imu, estres, anemia).let { if (it == Triage.SIN_DATOS) Triage.NOMINAL else it }
        return t.copy(anemia = combinada, triageSpo2 = spo2, triageImu = imu, estres = estres, triage = general)
    }
}
