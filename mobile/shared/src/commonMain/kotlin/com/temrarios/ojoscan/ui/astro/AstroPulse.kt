package com.temrarios.ojoscan.ui.astro

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.temrarios.ojoscan.astro.AstroPulseEngine
import com.temrarios.ojoscan.astro.Entorno
import com.temrarios.ojoscan.astro.Farmacia
import com.temrarios.ojoscan.astro.Latencia
import com.temrarios.ojoscan.astro.MissionTelemetry
import com.temrarios.ojoscan.astro.Triage
import com.temrarios.ojoscan.astro.ahoraMs
import com.temrarios.ojoscan.guia.Modo
import com.temrarios.ojoscan.historial.EstadoHistorial
import com.temrarios.ojoscan.historial.FuenteHistorial
import com.temrarios.ojoscan.historial.paraServidor
import com.temrarios.ojoscan.platform.EscuchaAcelerometro
import com.temrarios.ojoscan.platform.rememberBoveda
import com.temrarios.ojoscan.platform.rememberVoz
import com.temrarios.ojoscan.pulso.EstadoPulso
import com.temrarios.ojoscan.pulso.MonitorPulso
import kotlin.math.roundToInt

/** Paleta de cabina de vuelo. */
object Cabina {
    val fondo = Color(0xFF07090E)
    val vidrio = Color(0x0FFFFFFF)
    val vidrioAlto = Color(0x17FFFFFF)
    val borde = Color(0x1FFFFFFF)
    val texto = Color(0xFFE6EDF7)
    val suave = Color(0xFF8B98AD)
    val acento = Color(0xFF7DD3FC)
    val verde = Color(0xFF34D399)
    val ambar = Color(0xFFFBBF24)
    val rojo = Color(0xFFF87171)
    val mono = TextStyle(fontFamily = FontFamily.Monospace, letterSpacing = 1.5.sp)
}

fun Triage.color(): Color = when (this) {
    Triage.NOMINAL -> Cabina.verde; Triage.ADVERTENCIA -> Cabina.ambar; Triage.CRITICO -> Cabina.rojo; Triage.SIN_DATOS -> Cabina.suave
}

/** Módulos que se abren encima del tablero (sin pestañas). */
enum class Modulo { PPG, PVT, IMU, MANUALES, VOZ, SISTEMA, FARMACIA, SOS, IA }

@Composable
fun AstroPulseMainScreen(
    motor: AstroPulseEngine,
    pulso: MonitorPulso,
    historial: FuenteHistorial,
    alIaCardiaca: () -> Unit,
    alEscaneos: () -> Unit,
    alAnemia: (Modo) -> Unit,
    alChequeoCompleto: () -> Unit,
) {
    val tm by motor.estado.collectAsState()
    val estadoPulso by pulso.estado.collectAsState()
    val estadoHist by historial.estado.collectAsState()
    var abierto by remember { mutableStateOf<Modulo?>(null) }
    var manualPedido by remember { mutableStateOf<String?>(null) }
    val boveda = rememberBoveda()
    val voz = rememberVoz()

    // Caja negra inercial: escucha siempre, no solo con el módulo abierto.
    EscuchaAcelerometro(true) { x, y, z -> motor.muestraImu(x, y, z) }
    // Pulsera Xiaomi en vivo y SpO2 de 30 días (Health Connect).
    LaunchedEffect(estadoPulso) { (estadoPulso as? EstadoPulso.Midiendo)?.let { motor.pulsoPulsera(it.lpm) } }
    LaunchedEffect(estadoHist) {
        (estadoHist as? EstadoHistorial.Listo)?.resumen?.paraServidor()?.let {
            motor.spo2(it.spo2Media, "pulsera · 30 días")
            motor.pulsoReposo(it.pulsoReposo)
        }
    }
    // Aviso hablado si aparece un impacto severo.
    val severos = tm.impactos.count { it.severo }
    LaunchedEffect(severos) { if (severos > 0) voz.decir("Impacto severo detectado. Revisa tu estado o activa la baliza.") }

    fun sos() {
        if (boveda.disponible) runCatching { boveda.guardar("sos_${ahoraMs()}.json.enc", motor.reporteSos().encodeToByteArray()) }
        voz.decir("Baliza de emergencia activada. Reporte cifrado guardado.")
        abierto = Modulo.SOS
    }

    Box(Modifier.fillMaxSize().background(Cabina.fondo)) {
        FondoCabina()
        Column(
            Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(horizontal = 18.dp)
                .padding(top = 10.dp, bottom = 96.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Cabecera(tm.entorno, motor::cambiarEntorno)
            TarjetaAnemia(tm, alAnemia, alChequeoCompleto)
            Resumen(tm, estadoPulso)
            val modulos = listOf(
                Triple(Modulo.PPG, Ico.ppg, "Biometría PPG") to (tm.ppg?.let { "${it.lpm} lpm · cámara" } ?: "Cámara y fotograma"),
                Triple(Modulo.PVT, Ico.pvt, "Test Cognitivo PVT") to (tm.pvt?.let { "${it.mediaMs} ms · ${it.lapsos} lapsos" } ?: "Reacción en milisegundos"),
                Triple(Modulo.IMU, Ico.imu, "Caja Negra IMU") to "${tm.impactos.size} eventos · pico ${tm.imuPicoG} g",
                Triple(Modulo.MANUALES, Ico.manual, "Manuales Médicos") to "${com.temrarios.ojoscan.astro.Manuales.todos.size} procedimientos offline",
                Triple(Modulo.VOZ, Ico.voz, "Bitácora de Voz") to "Audio cifrado local",
                Triple(Modulo.SISTEMA, Ico.sistema, "Estado del Sistema") to "Autonomía y sensores",
            )
            val triages = mapOf(Modulo.PPG to (tm.ppg?.let { if (it.calidad >= 0.5) Triage.NOMINAL else Triage.ADVERTENCIA } ?: Triage.SIN_DATOS),
                Modulo.PVT to (tm.pvt?.triage ?: Triage.SIN_DATOS), Modulo.IMU to tm.triageImu)
            for (fila in modulos.chunked(2)) Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                for ((m, sub) in fila) TarjetaModulo(m.second, m.third, sub, triages[m.first], Modifier.weight(1f)) { abierto = m.first }
            }
            TarjetaFarmacia(tm) { abierto = Modulo.FARMACIA }
            Text("AstroPulse funciona sin conexión. No es un dispositivo médico certificado.",
                color = Cabina.suave.copy(alpha = 0.6f), fontSize = 10.sp, modifier = Modifier.fillMaxWidth())
        }

        // Barra flotante de acción rápida
        Row(
            Modifier.align(Alignment.BottomCenter).safeDrawingPadding().padding(horizontal = 18.dp, vertical = 12.dp)
                .fillMaxWidth().clip(RoundedCornerShape(22.dp)).background(Color(0xE6101420))
                .border(1.dp, Cabina.borde, RoundedCornerShape(22.dp)).padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            BotonAccion("Baliza SOS", Ico.sos, Cabina.rojo, Modifier.weight(1f), ::sos)
            BotonAccion("IA Asistente", Ico.ia, Cabina.acento, Modifier.weight(1f)) { abierto = Modulo.IA }
        }

        // Módulo abierto (capa superior)
        AnimatedVisibility(abierto != null, enter = fadeIn() + slideInVertically { it / 6 }, exit = fadeOut() + slideOutVertically { it / 6 }) {
            val m = abierto ?: return@AnimatedVisibility
            Capa(titulo = when (m) {
                Modulo.PPG -> "Biometría PPG"; Modulo.PVT -> "Test Cognitivo PVT"; Modulo.IMU -> "Caja Negra Inercial"
                Modulo.MANUALES -> "Manuales Médicos Offline"; Modulo.VOZ -> "Bitácora de Voz"; Modulo.SISTEMA -> "Estado del Sistema"
                Modulo.FARMACIA -> "Matriz Farmacéutica & Radiación"; Modulo.SOS -> "Baliza SOS"; Modulo.IA -> "IA Asistente (CDSS offline)"
            }, alCerrar = { abierto = null; manualPedido = null }) {
                when (m) {
                    Modulo.PPG -> PanelPpg(motor, alIaCardiaca, alEscaneos)
                    Modulo.PVT -> PanelPvt(motor)
                    Modulo.IMU -> PanelImu(motor)
                    Modulo.MANUALES -> PanelManuales(tm, manualPedido)
                    Modulo.VOZ -> PanelVoz()
                    Modulo.SISTEMA -> PanelSistema(motor, estadoPulso, estadoHist)
                    Modulo.FARMACIA -> PanelFarmacia(tm)
                    Modulo.SOS -> PanelSos(motor)
                    Modulo.IA -> PanelIa(motor) { id -> manualPedido = id; abierto = Modulo.MANUALES }
                }
            }
        }
    }
}

// ── Piezas del tablero ───────────────────────────────────────────────────────

@Composable
private fun FondoCabina() {
    Canvas(Modifier.fillMaxSize()) {
        drawRect(Brush.radialGradient(listOf(Color(0x332563EB), Color.Transparent), Offset(size.width * 0.9f, 0f), size.width))
        drawRect(Brush.radialGradient(listOf(Color(0x1A10B981), Color.Transparent), Offset(0f, size.height * 0.7f), size.width * 0.9f))
        for (i in 0 until 70) {
            val x = size.width * ((i * 0.618034f) % 1f)
            val y = size.height * ((i * 0.381966f + i * 0.0071f) % 1f)
            drawCircle(Color.White.copy(alpha = 0.08f + 0.25f * ((i * 37) % 10) / 10f), (0.6f + (i % 3) * 0.5f) * density, Offset(x, y))
        }
    }
}

@Composable
fun Vidrio(modifier: Modifier = Modifier, borde: Color = Cabina.borde, contenido: @Composable ColumnScope.() -> Unit) {
    val forma = RoundedCornerShape(20.dp)
    Column(
        modifier.clip(forma).background(Brush.verticalGradient(listOf(Cabina.vidrioAlto, Cabina.vidrio)))
            .border(1.dp, borde, forma).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp), content = contenido,
    )
}

@Composable
fun Etiqueta(t: String, color: Color = Cabina.suave, tam: Int = 10) =
    Text(t.uppercase(), style = Cabina.mono.copy(fontSize = tam.sp, color = color, fontWeight = FontWeight.SemiBold))

@Composable
fun LedTriage(t: Triage, tam: Dp = 10.dp) {
    val inf = rememberInfiniteTransition()
    val a by inf.animateFloat(0.35f, 1f, infiniteRepeatable(tween(if (t == Triage.CRITICO) 450 else 1200), RepeatMode.Reverse))
    Canvas(Modifier.size(tam * 2)) {
        drawCircle(t.color().copy(alpha = 0.22f * a), size.minDimension / 2)
        drawCircle(t.color().copy(alpha = 0.6f + 0.4f * a), size.minDimension / 4)
    }
}

@Composable
private fun Cabecera(entorno: Entorno, alCambiar: (Entorno) -> Unit) {
    var menu by remember { mutableStateOf(false) }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text("ASTROPULSE", style = Cabina.mono.copy(fontSize = 24.sp, color = Cabina.texto, fontWeight = FontWeight.Black, letterSpacing = 4.sp))
            Etiqueta("Telemetría y salud autónoma", tam = 9)
        }
        Box {
            Row(
                Modifier.clip(RoundedCornerShape(12.dp)).background(Cabina.vidrioAlto).border(1.dp, Cabina.borde, RoundedCornerShape(12.dp))
                    .clickable { menu = true }.padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(Ico.satelite, null, tint = Cabina.acento, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(6.dp))
                Text(entorno.corto, style = Cabina.mono.copy(fontSize = 12.sp, color = Cabina.texto, fontWeight = FontWeight.Bold))
                Icon(Ico.abajo, null, tint = Cabina.suave, modifier = Modifier.size(16.dp))
            }
            DropdownMenu(menu, { menu = false }, Modifier.background(Color(0xFF111622))) {
                for (e in Entorno.entries) DropdownMenuItem(
                    text = {
                        Column {
                            Text(e.nombre, color = if (e == entorno) Cabina.acento else Cabina.texto, fontSize = 14.sp)
                            Text("${e.gravedad} m/s² · ${e.dosisMsvDia} mSv/día", color = Cabina.suave, fontSize = 11.sp)
                        }
                    },
                    onClick = { alCambiar(e); menu = false },
                )
            }
        }
    }
}

@Composable
private fun Resumen(tm: MissionTelemetry, estadoPulso: EstadoPulso) {
    val inf = rememberInfiniteTransition()
    val brillo by inf.animateFloat(0.55f, 1f, infiniteRepeatable(tween(1600), RepeatMode.Reverse))
    Vidrio(Modifier.fillMaxWidth(), borde = tm.triage.color().copy(alpha = 0.45f)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            LedTriage(tm.triage, 12.dp)
            Spacer(Modifier.width(8.dp))
            Column(Modifier.weight(1f)) {
                Etiqueta("Estado de misión · día ${tm.diaMision}", tam = 9)
                Text(tm.triage.etiqueta, style = Cabina.mono.copy(fontSize = 26.sp, fontWeight = FontWeight.Black,
                    color = tm.triage.color().copy(alpha = brillo), letterSpacing = 3.sp))
            }
            Column(horizontalAlignment = Alignment.End) {
                Etiqueta("Pulso", tam = 9)
                Text(tm.pulso?.let { "$it" } ?: "--", color = Cabina.texto, fontSize = 22.sp, fontWeight = FontWeight.Bold)
                Text(if (estadoPulso is EstadoPulso.Midiendo) "pulsera" else tm.pulsoFuente.ifEmpty { "sin señal" },
                    color = Cabina.suave, fontSize = 10.sp)
            }
        }
        val lat = tm.latenciaSeg
        Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(Color(0x0D7DD3FC)).padding(10.dp),
            verticalAlignment = Alignment.CenterVertically) {
            Icon(Ico.satelite, null, tint = if (Latencia.autonomo(lat)) Cabina.ambar else Cabina.acento, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(8.dp))
            Text(
                (if (tm.entorno == Entorno.ISS) "Latencia LEO: " else "Latencia Deep Space: ") + Latencia.texto(lat) + " · " +
                    if (Latencia.autonomo(lat)) "Modo Autónomo" else "Enlace en tiempo real",
                style = Cabina.mono.copy(fontSize = 11.sp, color = if (Latencia.autonomo(lat)) Cabina.ambar else Cabina.acento),
            )
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Metrica("SpO2 Óptico", tm.spo2?.let { "$it" } ?: "--", "%", tm.triageSpo2, Modifier.weight(1f))
            Metrica("Aceleración IMU", "${tm.imuG}", "g", tm.triageImu, Modifier.weight(1f))
            Metrica("Estrés / PVT", tm.pvt?.let { "${it.mediaMs}" } ?: "--", "ms", tm.estres, Modifier.weight(1f))
        }
    }
}

@Composable
private fun Metrica(nombre: String, valor: String, unidad: String, t: Triage, modifier: Modifier) {
    Column(
        modifier.clip(RoundedCornerShape(14.dp)).background(t.color().copy(alpha = 0.08f))
            .border(1.dp, t.color().copy(alpha = 0.35f), RoundedCornerShape(14.dp)).padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Etiqueta(nombre, tam = 8)
        Row(verticalAlignment = Alignment.Bottom) {
            Text(valor, color = Cabina.texto, fontSize = 20.sp, fontWeight = FontWeight.Bold)
            Text(" $unidad", color = Cabina.suave, fontSize = 11.sp, modifier = Modifier.padding(bottom = 3.dp))
        }
        Text(t.etiqueta, style = Cabina.mono.copy(fontSize = 8.sp, color = t.color()))
    }
}

@Composable
private fun TarjetaModulo(icono: ImageVector, titulo: String, sub: String, t: Triage?, modifier: Modifier, alPulsar: () -> Unit) {
    val forma = RoundedCornerShape(18.dp)
    Column(
        modifier.height(128.dp).clip(forma).background(Brush.verticalGradient(listOf(Cabina.vidrioAlto, Cabina.vidrio)))
            .border(1.dp, Cabina.borde, forma).clickable(onClick = alPulsar).padding(14.dp),
        verticalArrangement = Arrangement.SpaceBetween,
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(icono, null, tint = Cabina.acento, modifier = Modifier.size(26.dp))
            Spacer(Modifier.weight(1f))
            if (t != null && t != Triage.SIN_DATOS) Box(Modifier.size(8.dp).clip(CircleShape).background(t.color()))
        }
        Column {
            Text(titulo, color = Cabina.texto, fontWeight = FontWeight.SemiBold, fontSize = 14.sp, lineHeight = 17.sp)
            Text(sub, color = Cabina.suave, fontSize = 11.sp, maxLines = 2, lineHeight = 14.sp)
        }
    }
}

@Composable
private fun TarjetaAnemia(tm: MissionTelemetry, alAnemia: (Modo) -> Unit, alCompleto: () -> Unit) {
    val a = tm.anemia
    val t = a?.triage ?: Triage.SIN_DATOS
    val veredicto = when (t) {
        Triage.CRITICO -> "Probabilidad ALTA"; Triage.ADVERTENCIA -> "Probabilidad MEDIA"
        Triage.NOMINAL -> "Probabilidad BAJA"; Triage.SIN_DATOS -> "Sin medir todavía"
    }
    Vidrio(Modifier.fillMaxWidth(), borde = (if (a != null) t.color() else Cabina.acento).copy(alpha = 0.8f)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Ico.gota, null, tint = Cabina.rojo, modifier = Modifier.size(30.dp))
            Spacer(Modifier.width(10.dp))
            Text("ANEMIA", color = Cabina.texto, fontWeight = FontWeight.Black, fontSize = 22.sp)
        }
        Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(if (a != null) "${a.porcentaje} %" else "— %",
                style = Cabina.mono.copy(fontSize = 72.sp, color = if (a != null) t.color() else Cabina.suave, fontWeight = FontWeight.Black))
            Text(veredicto, color = if (a != null) t.color() else Cabina.suave, fontWeight = FontWeight.Bold, fontSize = 18.sp)
            Text(if (a != null) "de tener anemia" else "Mide tu párpado, palma o encía", color = Cabina.suave, fontSize = 13.sp)
        }
        Box(Modifier.fillMaxWidth().height(14.dp).clip(RoundedCornerShape(7.dp)).background(Color(0x14FFFFFF))) {
            if (a != null) Box(Modifier.fillMaxWidth(a.probabilidad.toFloat().coerceIn(0.02f, 1f)).height(14.dp)
                .clip(RoundedCornerShape(7.dp)).background(t.color()))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Chip("Párpado", Modifier.weight(1f)) { alAnemia(Modo.ANEMIA) }
            Chip("Palma", Modifier.weight(1f)) { alAnemia(Modo.PALMA) }
            Chip("Encía", Modifier.weight(1f)) { alAnemia(Modo.ENCIA) }
        }
    }
}

@Composable
private fun Chip(t: String, modifier: Modifier, alPulsar: () -> Unit) {
    Box(modifier.height(40.dp).clip(RoundedCornerShape(12.dp)).background(Cabina.acento.copy(alpha = 0.1f))
        .border(1.dp, Cabina.acento.copy(alpha = 0.35f), RoundedCornerShape(12.dp)).clickable(onClick = alPulsar),
        contentAlignment = Alignment.Center) { Text(t, color = Cabina.acento, fontSize = 12.sp, fontWeight = FontWeight.SemiBold) }
}

@Composable
private fun TarjetaFarmacia(tm: MissionTelemetry, alPulsar: () -> Unit) {
    val matriz = Farmacia.matriz(tm.diaMision, tm.entorno)
    val criticos = matriz.count { it.triage == Triage.CRITICO }
    val avisos = matriz.count { it.triage == Triage.ADVERTENCIA }
    val t = Triage.peor(*matriz.map { it.triage }.toTypedArray())
    Vidrio(Modifier.fillMaxWidth().clickable(onClick = alPulsar), borde = t.color().copy(alpha = 0.4f)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Ico.farmacia, null, tint = Cabina.acento, modifier = Modifier.size(24.dp))
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text("Matriz Farmacéutica & Radiación", color = Cabina.texto, fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
                Text("${tm.entorno.dosisMsvDia} mSv/día · dosis de misión ${(tm.dosisAcumuladaMsv * 10).roundToInt() / 10.0} mSv",
                    color = Cabina.suave, fontSize = 11.sp)
            }
            LedTriage(t)
        }
        for (e in matriz.take(3)) BarraCaducidad(e.farmaco.nombre, e.diasRestantes, e.potencia, e.farmaco.vidaDias, e.triage)
        Text("$criticos caducados · $avisos por caducar · ${matriz.size} en el botiquín", style = Cabina.mono.copy(fontSize = 10.sp, color = Cabina.suave))
    }
}

@Composable
fun BarraCaducidad(nombre: String, dias: Int, potencia: Double, vida: Int, t: Triage) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row {
            Text(nombre, color = Cabina.texto, fontSize = 12.sp, modifier = Modifier.weight(1f), maxLines = 1)
            Text(if (dias < 0) "CADUCADO" else "$dias d · $potencia %", style = Cabina.mono.copy(fontSize = 10.sp, color = t.color()))
        }
        Canvas(Modifier.fillMaxWidth().height(5.dp)) {
            val f = (dias.toFloat() / vida).coerceIn(0f, 1f)
            drawRoundRect(Color(0x22FFFFFF), cornerRadius = androidx.compose.ui.geometry.CornerRadius(size.height))
            drawRoundRect(t.color(), size = size.copy(width = size.width * f), cornerRadius = androidx.compose.ui.geometry.CornerRadius(size.height))
        }
    }
}

@Composable
private fun BotonAccion(texto: String, icono: ImageVector, color: Color, modifier: Modifier, alPulsar: () -> Unit) {
    Row(
        modifier.height(52.dp).clip(RoundedCornerShape(16.dp)).background(color.copy(alpha = 0.14f))
            .border(1.dp, color.copy(alpha = 0.5f), RoundedCornerShape(16.dp)).clickable(onClick = alPulsar),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.Center,
    ) {
        Icon(icono, null, tint = color, modifier = Modifier.size(20.dp))
        Spacer(Modifier.width(8.dp))
        Text(texto, color = color, fontWeight = FontWeight.Bold, fontSize = 14.sp)
    }
}

/** Capa de pantalla completa para un módulo, con su botón de cerrar. */
@Composable
private fun Capa(titulo: String, alCerrar: () -> Unit, contenido: @Composable ColumnScope.() -> Unit) {
    Box(Modifier.fillMaxSize().background(Cabina.fondo.copy(alpha = 0.97f)).clickable(enabled = false) {}) {
        Column(Modifier.safeDrawingPadding().fillMaxSize()) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(titulo, color = Cabina.texto, fontSize = 18.sp, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                Box(Modifier.size(40.dp).clip(CircleShape).background(Cabina.vidrioAlto).clickable(onClick = alCerrar), contentAlignment = Alignment.Center) {
                    Icon(Ico.cerrar, "Cerrar", tint = Cabina.texto, modifier = Modifier.size(20.dp))
                }
            }
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 18.dp).padding(bottom = 24.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp), content = contenido)
        }
    }
}
