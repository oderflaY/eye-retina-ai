package com.temrarios.ojoscan.astro

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.temrarios.ojoscan.platform.Almacen
import com.temrarios.ojoscan.platform.desfaseHorarioMs
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
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

    // ── Diario de salud con semáforo + RAG de primeros auxilios ──
    private val _diario = MutableStateFlow(diarioGuardado())
    val diario: StateFlow<List<EntradaDiario>> = _diario.asStateFlow()
    private val _evaluacion = MutableStateFlow<Pair<Evaluacion, List<Recuperado>>?>(null)
    /** Última evaluación del semáforo con los procedimientos recuperados. */
    val evaluacion: StateFlow<Pair<Evaluacion, List<Recuperado>>?> = _evaluacion.asStateFlow()
    private val _base = MutableStateFlow<BaseAuxilios?>(null)
    val base: StateFlow<BaseAuxilios?> = _base.asStateFlow()

    // ── Historial personal: todas las mediciones + recordatorios + alertas ──
    private val _registros = MutableStateFlow(Seguimiento.deJson(almacen.leer("historial_v1")))
    private val _historial = MutableStateFlow(emptyList<Registro>())
    /** Todas las mediciones de la persona (incluye las entradas del diario), de la más vieja a la más nueva. */
    val historial: StateFlow<List<Registro>> = _historial.asStateFlow()
    private val _alertas = MutableStateFlow(emptyList<Alerta>())
    val alertas: StateFlow<List<Alerta>> = _alertas.asStateFlow()
    /** Alertas nuevas que la interfaz debe convertir en notificación (se vacía al consumirlas). */
    private val _porAvisar = MutableStateFlow(emptyList<Alerta>())
    val porAvisar: StateFlow<List<Alerta>> = _porAvisar.asStateFlow()
    private val ultimoRegistro = mutableMapOf<String, Long>()

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
        actualizarHistorial(avisar = false)
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
        registrar(Registro(reloj(), "pulso", "Pulso", lpm.toDouble(), "lpm", "pulsera", Seguimiento.clasificarPulso(lpm.toDouble()), "pulsera"), cadaMin = 30)
    }

    fun spo2(valor: Double?, fuente: String) {
        if (valor == null) return
        recalcular { it.copy(spo2 = (valor * 10).roundToInt() / 10.0, spo2Fuente = fuente) }
        registrar(Registro(reloj(), "spo2", "Oxígeno (SpO₂)", (valor * 10).roundToInt() / 10.0, "%", fuente, Seguimiento.clasificarSpo2(valor), fuente), cadaMin = 30)
    }

    fun ppg(l: LecturaPpg) {
        recalcular { it.copy(ppg = l, pulso = if (l.calidad >= 0.5) l.lpm else it.pulso, pulsoFuente = if (l.calidad >= 0.5) "PPG cámara" else it.pulsoFuente) }
        // una lectura buena por medición (la cámara analiza cada segundo): como máximo cada 10 min
        if (l.calidad >= 0.5) registrar(Registro(reloj(), "pulso", "Pulso", l.lpm.toDouble(), "lpm", "cámara",
            Seguimiento.clasificarPulso(l.lpm.toDouble()), "cámara"), cadaMin = 10)
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
        if (impacto != null) registrar(Registro(impacto.horaMs, "impacto", "Impacto / caída", impacto.g, "g",
            if (impacto.severo) "severo" else "leve", if (impacto.severo) Triage.CRITICO else Triage.ADVERTENCIA, "IMU"), cadaMin = 1)
    }

    /** Resultado de un tamizaje de anemia; se junta con los demás en una sola probabilidad. */
    fun anemia(metodo: String, resultado: String, titulo: String, probabilidad: Double?, auc: Double?) {
        val l = LecturaAnemia(metodo, resultado, titulo, probabilidad, auc, reloj())
        recalcular { it.copy(anemias = it.anemias + (metodo to l)) }
        registrar(Registro(l.horaMs, "anemia", if (Seguimiento.ANEMIA_EN_PRUEBAS) "Anemia ($metodo) · pruebas" else "Anemia ($metodo)", probabilidad, if (probabilidad != null) "%prob" else "", titulo,
            probabilidad?.let { Seguimiento.clasificarProbabilidad(it) } ?: if (resultado == "palidez") Triage.ADVERTENCIA else Triage.NOMINAL, metodo))
        almacen.guardar("astro_anemias", _estado.value.anemias.values.joinToString("\n") {
            listOf(it.metodo, it.resultado, it.probabilidad ?: "", it.auc ?: "", it.horaMs, it.titulo.replace("|", " ")).joinToString("|")
        })
    }

    /** Carga la base de primeros auxilios (JSON del asset). Llamar fuera del hilo principal: indexa ~5 800 documentos. */
    fun cargarBase(json: String) {
        if (_base.value == null) _base.value = runCatching { BaseAuxilios.desdeJson(json) }.getOrNull()
    }

    fun buscarAuxilios(consulta: String, k: Int = 3): List<Recuperado> = _base.value?.buscar(consulta, k).orEmpty()

    /** Guarda lo que escribió la persona, lo pasa por el semáforo y recupera qué hacer. */
    fun escribirDiario(texto: String, animo: Int?): Evaluacion? {
        if (texto.isBlank() && animo == null) return null
        val ahora = reloj()
        val tm = _estado.value
        val ev = Semaforo.evaluar(texto, animo, _diario.value, tm, ahora, ::diaLocal)
        val docs = if (ev.sintomas.isEmpty()) emptyList() else _base.value?.paraDiario(ev.consulta, texto).orEmpty()
        val e = EntradaDiario(ahora, texto.trim(), animo, ev.triage, ev.motivos, ev.sintomas, tm.pulso, tm.spo2)
        _diario.update { (it + e).takeLast(400) }
        almacen.guardar("diario_v1", jsonDiario.encodeToString(ListSerializer(EntradaDiario.serializer()), _diario.value))
        _evaluacion.value = ev to docs
        recalcular { it }
        actualizarHistorial()
        return ev
    }

    fun borrarEntradaDiario(horaMs: Long) {
        _diario.update { l -> l.filterNot { it.horaMs == horaMs } }
        almacen.guardar("diario_v1", jsonDiario.encodeToString(ListSerializer(EntradaDiario.serializer()), _diario.value))
        recalcular { it }
        actualizarHistorial(avisar = false)
    }

    /** Número de día en la zona horaria del teléfono. */
    fun diaLocal(ms: Long): Long = (ms + desfaseHorarioMs(ms)).floorDiv(DIA_MS)

    private fun diarioGuardado(): List<EntradaDiario> = almacen.leer("diario_v1")?.let {
        runCatching { jsonDiario.decodeFromString(ListSerializer(EntradaDiario.serializer()), it) }.getOrNull()
    }.orEmpty()

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
        registrar(Registro(reloj(), "pvt", "Test de alerta (PVT)", r.medianaMs.toDouble(), "ms", "${r.lapsos} lapsos", r.triage, "PVT"))
    }

    /** Resultado de un escaneo del ojo, rostro o perfil (pantalla de resultados del chequeo). */
    fun registrarEstudio(tipo: String, nombre: String, texto: String, probabilidad: Double?, triage: Triage) =
        registrar(Registro(reloj(), tipo, nombre, probabilidad, if (probabilidad != null) "%prob" else "", texto, triage, tipo))

    /** Guarda una medición (con un mínimo de minutos entre dos del mismo tipo y fuente, salvo que cambie de color). */
    fun registrar(r: Registro, cadaMin: Int = 0) {
        val k = r.tipo + "|" + r.fuente
        val previo = _registros.value.lastOrNull { it.tipo == r.tipo && it.fuente == r.fuente }
        if (cadaMin > 0 && r.horaMs - (ultimoRegistro[k] ?: 0L) < cadaMin * 60_000L && previo?.triage == r.triage) return
        ultimoRegistro[k] = r.horaMs
        _registros.update { (it + r).takeLast(3000) }
        almacen.guardar("historial_v1", Seguimiento.aJson(_registros.value))
        actualizarHistorial()
    }

    fun borrarRegistro(horaMs: Long, tipo: String) {
        if (tipo == "diario") return borrarEntradaDiario(horaMs)
        _registros.update { l -> l.filterNot { it.horaMs == horaMs && it.tipo == tipo } }
        almacen.guardar("historial_v1", Seguimiento.aJson(_registros.value))
        actualizarHistorial()
    }

    /** Une registros + diario, recalcula alertas y separa las nuevas para notificar. */
    fun actualizarHistorial(avisar: Boolean = true) {
        val diario = _diario.value.map { Registro(it.horaMs, "diario", "Diario", it.animo?.toDouble(), "", it.texto, it.triage, "diario") }
        _historial.value = (_registros.value + diario).sortedBy { it.horaMs }
        val ahora = reloj()
        val a = Seguimiento.alertas(_historial.value, ahora)
        _alertas.value = a
        val vistas = almacen.leer("alertas_vistas").orEmpty().split(",").filter { it.isNotBlank() }.toSet()
        val nuevas = a.filter { it.id !in vistas && it.triage != Triage.NOMINAL }
        if (nuevas.isNotEmpty()) {
            almacen.guardar("alertas_vistas", (vistas + nuevas.map { it.id }).toList().takeLast(300).joinToString(","))
            if (avisar) _porAvisar.update { it + nuevas }
        }
    }

    /** Medición que la persona anota a mano (baumanómetro, báscula, glucómetro, termómetro, laboratorio). */
    fun anotar(tipo: String, valor: Double, valor2: Double? = null, mujer: Boolean = false, ayunas: Boolean = false) {
        val ahora = reloj()
        val r = when (tipo) {
            "presion" -> Registro(ahora, "presion", "Presión arterial", valor, "mmHg", "${valor.toInt()}/${(valor2 ?: 0.0).toInt()}",
                Seguimiento.clasificarPresion(valor, valor2 ?: 0.0), "baumanómetro")
            "peso" -> Registro(ahora, "peso", "Peso", valor, "kg", "", Triage.NOMINAL, "báscula")
            "glucosa" -> Registro(ahora, "glucosa", "Glucosa", valor, "mg/dL", if (ayunas) "en ayunas" else "casual",
                Seguimiento.clasificarGlucosa(valor, ayunas), "glucómetro")
            "temperatura" -> Registro(ahora, "temperatura", "Temperatura", valor, "°C", "", Seguimiento.clasificarTemperatura(valor), "termómetro")
            "hb_lab" -> Registro(ahora, "hb_lab", "Hemoglobina (laboratorio)", valor, "g/dL", "biometría hemática",
                Seguimiento.clasificarHb(valor, mujer), "laboratorio")
            else -> return
        }
        registrar(r)
    }

    fun respaldo(): String = ReporteSalud.respaldoJson(Respaldo(creadoMs = reloj(), historial = _registros.value, diario = _diario.value,
        perfil = almacen.leer("perfil")))

    /** Une el respaldo con lo que ya hay (no borra nada; los repetidos se ignoran). Devuelve cuántos registros nuevos entraron. */
    fun restaurar(json: String): Int? {
        val r = ReporteSalud.respaldoDe(json) ?: return null
        val antes = _registros.value.size + _diario.value.size
        if (almacen.leer("perfil") == null) r.perfil?.let { almacen.guardar("perfil", it) }
        _registros.update { (it + r.historial).distinctBy { x -> x.horaMs to x.tipo }.sortedBy { x -> x.horaMs }.takeLast(3000) }
        _diario.update { (it + r.diario).distinctBy { x -> x.horaMs }.sortedBy { x -> x.horaMs }.takeLast(400) }
        almacen.guardar("historial_v1", Seguimiento.aJson(_registros.value))
        almacen.guardar("diario_v1", jsonDiario.encodeToString(ListSerializer(EntradaDiario.serializer()), _diario.value))
        actualizarHistorial(avisar = false)
        recalcular { it }
        return _registros.value.size + _diario.value.size - antes
    }

    fun avisosEntregados() { _porAvisar.value = emptyList() }

    fun revisiones(): List<Revision> = Seguimiento.revisiones(_historial.value, reloj())

    fun consultarHistorial(pregunta: String): Mensaje = ConsultaHistorial.responder(pregunta, _historial.value, reloj())

    // ── Asistente ────────────────────────────────────────────────────────────
    fun preguntar(texto: String) {
        if (texto.isBlank()) return
        // preguntas sobre SUS datos («¿cómo va mi pulso?», «¿cuándo me toca?») → RAG del historial personal
        val r = if (ConsultaHistorial.esPersonal(texto)) consultarHistorial(texto) else Cdss.responder(texto, _estado.value, _base.value)
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
        // en modo pruebas se muestra en su tarjeta pero no mueve el estado general
        val anemia = if (Seguimiento.ANEMIA_EN_PRUEBAS) Triage.SIN_DATOS else combinada?.triage ?: Triage.SIN_DATOS
        // Diario: la última entrada de las últimas 12 h
        val diario = _diario.value.lastOrNull()?.takeIf { ahora - it.horaMs < 12 * 3_600_000L }?.triage ?: Triage.SIN_DATOS
        val general = Triage.peor(spo2, pulso, imu, estres, anemia, diario).let { if (it == Triage.SIN_DATOS) Triage.NOMINAL else it }
        return t.copy(diario = diario, anemia = combinada, triageSpo2 = spo2, triageImu = imu, estres = estres, triage = general)
    }
}

private val jsonDiario = Json { ignoreUnknownKeys = true; encodeDefaults = false }
