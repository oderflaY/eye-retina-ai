package com.temrarios.ojoscan.historial

import android.content.Context
import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.platform.LocalContext
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.PermissionController
import androidx.health.connect.client.permission.HealthPermission
import androidx.health.connect.client.records.HeartRateRecord
import androidx.health.connect.client.records.HeartRateVariabilityRmssdRecord
import androidx.health.connect.client.records.OxygenSaturationRecord
import androidx.health.connect.client.records.Record
import androidx.health.connect.client.records.RestingHeartRateRecord
import androidx.health.connect.client.records.SleepSessionRecord
import androidx.health.connect.client.records.StepsRecord
import androidx.health.connect.client.request.AggregateGroupByPeriodRequest
import androidx.health.connect.client.request.ReadRecordsRequest
import androidx.health.connect.client.time.TimeRangeFilter
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.time.Duration
import java.time.Instant
import java.time.LocalDateTime
import java.time.Period
import java.time.ZoneId
import kotlin.reflect.KClass

private val TIPOS: List<KClass<out Record>> = listOf(
    HeartRateRecord::class, RestingHeartRateRecord::class, OxygenSaturationRecord::class,
    SleepSessionRecord::class, StepsRecord::class, HeartRateVariabilityRmssdRecord::class,
)
private val PERMISOS: Set<String> = TIPOS.map { HealthPermission.getReadPermission(it) }.toSet()

private class FuenteHC(private val ctx: Context, private val scope: CoroutineScope, private val pedir: (Set<String>) -> Unit) : FuenteHistorial {
    override val estado = MutableStateFlow<EstadoHistorial>(EstadoHistorial.Inactivo)
    private var dias = 30

    private fun cliente(): HealthConnectClient? =
        if (HealthConnectClient.getSdkStatus(ctx) == HealthConnectClient.SDK_AVAILABLE) HealthConnectClient.getOrCreate(ctx) else null

    override fun conectar(dias: Int) {
        this.dias = dias
        val c = cliente() ?: run { estado.value = EstadoHistorial.NoDisponible; return }
        scope.launch {
            // Basta con que haya permiso para algo: se lee lo que se pueda.
            val dados = runCatching { c.permissionController.getGrantedPermissions() }.getOrDefault(emptySet())
            if (dados.intersect(PERMISOS).isEmpty()) { estado.value = EstadoHistorial.SinPermiso; pedir(PERMISOS) }
            else leer(c, dados)
        }
    }

    override fun leerSiHayPermiso(dias: Int) {
        if (estado.value == EstadoHistorial.Leyendo) return
        this.dias = dias
        val c = cliente() ?: run { estado.value = EstadoHistorial.NoDisponible; return }
        scope.launch {
            val dados = runCatching { c.permissionController.getGrantedPermissions() }.getOrDefault(emptySet())
            if (dados.intersect(PERMISOS).isEmpty()) estado.value = EstadoHistorial.SinPermiso
            else leer(c, dados)
        }
    }

    fun trasPermiso(dados: Set<String>) {
        val c = cliente() ?: return
        if (dados.intersect(PERMISOS).isEmpty()) { estado.value = EstadoHistorial.SinPermiso; return }
        scope.launch { leer(c, dados) }
    }

    override fun abrirAjustes() {
        val i = Intent(HealthConnectClient.ACTION_HEALTH_CONNECT_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        runCatching { ctx.startActivity(i) }
    }

    private suspend fun <T : Record> todos(c: HealthConnectClient, tipo: KClass<T>, filtro: TimeRangeFilter): List<T> {
        val out = mutableListOf<T>()
        var token: String? = null
        do {
            val r = c.readRecords(ReadRecordsRequest(tipo, filtro, pageSize = 1000, pageToken = token))
            out += r.records
            token = r.pageToken
        } while (token != null && out.size < 50_000)
        return out
    }

    private suspend fun leer(c: HealthConnectClient, dados: Set<String>) {
        estado.value = EstadoHistorial.Leyendo
        try {
            val res = withContext(Dispatchers.IO) { resumir(c, dados) }
            estado.value = if (res.vacio) EstadoHistorial.SinDatos else EstadoHistorial.Listo(res)
        } catch (e: Exception) {
            estado.value = EstadoHistorial.Error(e.message ?: "No pude leer Health Connect")
        }
    }

    private suspend fun resumir(c: HealthConnectClient, dados: Set<String>): ResumenHistorial {
        val zona = ZoneId.systemDefault()
        val fin = Instant.now()
        val ini = fin.minus(Duration.ofDays(dias.toLong()))
        val f = TimeRangeFilter.between(ini, fin)
        fun puede(t: KClass<out Record>) = HealthPermission.getReadPermission(t) in dados
        fun dia(i: Instant) = i.atZone(zona).toLocalDate().toString()

        // Pulso en reposo: el de la pulsera si lo escribe; si no, el percentil 10 de cada día (≥ 60 lecturas).
        var reposo = emptyList<Diario>()
        var maximo: Int? = null
        if (puede(RestingHeartRateRecord::class)) {
            reposo = todos(c, RestingHeartRateRecord::class, f).groupBy { dia(it.time) }
                .map { (d, l) -> Diario(d, l.map { it.beatsPerMinute.toDouble() }.average()) }
        }
        if (puede(HeartRateRecord::class)) {
            val muestras = todos(c, HeartRateRecord::class, f).flatMap { it.samples }.filter { it.beatsPerMinute in 30..230 }
            maximo = muestras.maxOfOrNull { it.beatsPerMinute.toInt() }
            if (reposo.isEmpty()) reposo = muestras.groupBy { dia(it.time) }.filterValues { it.size >= 60 }
                .map { (d, l) -> Diario(d, l.map { it.beatsPerMinute.toDouble() }.sorted()[l.size / 10]) }
        }
        val spo2 = if (puede(OxygenSaturationRecord::class))
            todos(c, OxygenSaturationRecord::class, f).map { it.percentage.value }.filter { it in 70.0..100.0 } else emptyList()

        var sueno = emptyList<Diario>()
        var profundo: Double? = null
        if (puede(SleepSessionRecord::class)) {
            val sesiones = todos(c, SleepSessionRecord::class, f)
            // La noche se asigna al día en que te despiertas.
            sueno = sesiones.groupBy { dia(it.endTime) }.map { (d, l) ->
                Diario(d, l.sumOf { Duration.between(it.startTime, it.endTime).toMinutes() } / 60.0)
            }
            val etapas = sesiones.flatMap { it.stages }
            val total = etapas.sumOf { Duration.between(it.startTime, it.endTime).toMinutes() }
            if (total > 0) profundo = etapas.filter { it.stage == SleepSessionRecord.STAGE_TYPE_DEEP }
                .sumOf { Duration.between(it.startTime, it.endTime).toMinutes() }.toDouble() / total
        }
        val pasos = if (puede(StepsRecord::class)) {
            val hoy = LocalDateTime.now(zona)
            c.aggregateGroupByPeriod(AggregateGroupByPeriodRequest(
                metrics = setOf(StepsRecord.COUNT_TOTAL),
                timeRangeFilter = TimeRangeFilter.between(hoy.minusDays(dias.toLong()).toLocalDate().atStartOfDay(), hoy.toLocalDate().atStartOfDay()),
                timeRangeSlicer = Period.ofDays(1),
            )).mapNotNull { g -> g.result[StepsRecord.COUNT_TOTAL]?.let { Diario(g.startTime.toLocalDate().toString(), it.toDouble()) } }
        } else emptyList()
        val vfc = if (puede(HeartRateVariabilityRmssdRecord::class))
            todos(c, HeartRateVariabilityRmssdRecord::class, f).map { it.heartRateVariabilityMillis } else emptyList()

        return ResumenHistorial(dias, reposo, maximo, spo2, sueno, profundo, pasos, vfc)
    }
}

@Composable
actual fun rememberFuenteHistorial(): FuenteHistorial {
    val ctx = LocalContext.current.applicationContext
    val scope = rememberCoroutineScope()
    var fuente: FuenteHC? = null
    val launcher = rememberLauncherForActivityResult(PermissionController.createRequestPermissionResultContract()) { dados ->
        fuente?.trasPermiso(dados)
    }
    val f = remember { FuenteHC(ctx, scope) { launcher.launch(it) } }
    fuente = f
    return f
}
