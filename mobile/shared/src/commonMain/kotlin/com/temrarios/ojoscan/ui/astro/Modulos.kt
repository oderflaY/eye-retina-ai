package com.temrarios.ojoscan.ui.astro

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameMillis
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.temrarios.ojoscan.astro.AstroPulseEngine
import com.temrarios.ojoscan.astro.Cdss
import com.temrarios.ojoscan.astro.EstimadorPpg
import com.temrarios.ojoscan.astro.Farmacia
import com.temrarios.ojoscan.astro.Latencia
import com.temrarios.ojoscan.astro.Manual
import com.temrarios.ojoscan.astro.Manuales
import com.temrarios.ojoscan.astro.MissionTelemetry
import com.temrarios.ojoscan.astro.MotorPvt
import com.temrarios.ojoscan.astro.Triage
import com.temrarios.ojoscan.astro.ahoraMs
import com.temrarios.ojoscan.astro.normalizar
import com.temrarios.ojoscan.historial.EstadoHistorial
import com.temrarios.ojoscan.platform.CamaraPpg
import com.temrarios.ojoscan.platform.rememberBoveda
import com.temrarios.ojoscan.platform.rememberDispositivo
import com.temrarios.ojoscan.platform.rememberGrabadora
import com.temrarios.ojoscan.pulso.EstadoPulso
import kotlinx.coroutines.delay
import kotlin.math.roundToInt

private fun hace(ms: Long): String {
    val s = ((ahoraMs() - ms) / 1000).coerceAtLeast(0)
    return when { s < 60 -> "hace ${s} s"; s < 3600 -> "hace ${s / 60} min"; s < 86_400 -> "hace ${s / 3600} h"; else -> "hace ${s / 86_400} d" }
}

@Composable
private fun Boton(texto: String, color: Color = Cabina.acento, modifier: Modifier = Modifier.fillMaxWidth(), activo: Boolean = true, alPulsar: () -> Unit) {
    Box(
        modifier.height(50.dp).clip(RoundedCornerShape(14.dp)).background(color.copy(alpha = if (activo) 0.16f else 0.05f))
            .border(1.dp, color.copy(alpha = if (activo) 0.55f else 0.2f), RoundedCornerShape(14.dp)).clickable(enabled = activo, onClick = alPulsar),
        contentAlignment = Alignment.Center,
    ) { Text(texto, color = if (activo) color else Cabina.suave, fontWeight = FontWeight.Bold, fontSize = 14.sp) }
}

@Composable
private fun Dato(nombre: String, valor: String, modifier: Modifier = Modifier, color: Color = Cabina.texto) {
    Column(modifier) {
        Etiqueta(nombre, tam = 8)
        Text(valor, color = color, fontSize = 20.sp, fontWeight = FontWeight.Bold)
    }
}

/** Línea de señal (osciloscopio). */
@Composable
private fun Osciloscopio(v: List<Double>, color: Color, modifier: Modifier, umbral: Double? = null, maxFijo: Double? = null) {
    Canvas(modifier.clip(RoundedCornerShape(12.dp)).background(Color(0x0AFFFFFF))) {
        for (i in 1..3) drawLine(Color(0x14FFFFFF), Offset(0f, size.height * i / 4), Offset(size.width, size.height * i / 4), 1f)
        if (v.size < 2) return@Canvas
        val lo = if (maxFijo != null) 0.0 else v.min()
        val hi = maxFijo ?: v.max()
        val r = (hi - lo).takeIf { it > 1e-9 } ?: 1.0
        fun y(x: Double) = (size.height * (1 - (x - lo) / r)).toFloat()
        val p = Path()
        v.forEachIndexed { i, x -> val px = size.width * i / (v.size - 1); if (i == 0) p.moveTo(px, y(x)) else p.lineTo(px, y(x)) }
        drawPath(p, color, style = Stroke(2.dp.toPx()))
        umbral?.let { drawLine(Cabina.rojo, Offset(0f, y(it)), Offset(size.width, y(it)), 1.5.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(12f, 8f))) }
    }
}

// ── 1. Biometría PPG ─────────────────────────────────────────────────────────

@Composable
fun PanelPpg(motor: AstroPulseEngine, alIaCardiaca: () -> Unit, alEscaneos: () -> Unit) {
    val est = remember { EstimadorPpg() }
    var midiendo by remember { mutableStateOf(false) }
    var dedo by remember { mutableStateOf(false) }
    var senal by remember { mutableStateOf(emptyList<Double>()) }
    val tm by motor.estado.collectAsState()
    Text("Pon la yema del dedo índice cubriendo la cámara trasera y el flash. Quédate quieto 15 segundos.",
        color = Cabina.suave, fontSize = 13.sp, lineHeight = 18.sp)
    if (midiendo) {
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
            CamaraPpg(Modifier.size(96.dp).clip(CircleShape).border(2.dp, if (dedo) Cabina.verde else Cabina.ambar, CircleShape)) { t, r, g ->
                dedo = est.dedoPresente(r, g)
                if (dedo) est.agregar(t, r) else est.limpiar()
                senal = est.senal.takeLast(240)
                est.estimar(t)?.let { motor.ppg(it) }
            }
            Column(Modifier.weight(1f)) {
                Text(if (dedo) "Señal óptica capturada" else "Cubre el lente con el dedo", color = if (dedo) Cabina.verde else Cabina.ambar,
                    fontWeight = FontWeight.SemiBold)
                Text("Fotograma: rojo del centro de cada cuadro, ~30 fps", color = Cabina.suave, fontSize = 11.sp)
            }
        }
        Osciloscopio(senal.map { -it }, Cabina.rojo, Modifier.fillMaxWidth().height(120.dp))
        Boton("Detener medición", Cabina.suave) { midiendo = false; est.limpiar() }
    } else Boton("Iniciar medición PPG") { midiendo = true }
    tm.ppg?.let { l ->
        Vidrio(Modifier.fillMaxWidth()) {
            Row {
                Dato("Pulso", "${l.lpm} lpm", Modifier.weight(1f))
                Dato("Perfusión", "${l.perfusion} %", Modifier.weight(1f))
                Dato("Calidad", "${(l.calidad * 100).roundToInt()} %", Modifier.weight(1f),
                    if (l.calidad >= 0.5) Cabina.verde else Cabina.ambar)
            }
            Text(if (l.calidad >= 0.5) "Lectura válida: actualiza el pulso de la misión." else
                "Calidad baja: no muevas el dedo ni lo aprietes demasiado.", color = Cabina.suave, fontSize = 11.sp)
        }
    }
    Etiqueta("Más biometría")
    Boton("IA cardiaca con la pulsera Xiaomi", Cabina.verde, alPulsar = alIaCardiaca)
    Boton("Escaneos de ojo, anemia y rostro", Cabina.acento, alPulsar = alEscaneos)
}

// ── 2. Test PVT ──────────────────────────────────────────────────────────────

@Composable
fun PanelPvt(motor: AstroPulseEngine) {
    val pvt = motor.pvt
    var version by remember { mutableIntStateOf(0) }
    var ahora by remember { mutableLongStateOf(0L) }
    var mensaje by remember { mutableStateOf("Toca la pantalla en cuanto aparezca el contador rojo. ${pvt.ensayos} estímulos, ~1 min.") }
    val fase = remember(version) { pvt.fase }
    val tm by motor.estado.collectAsState()

    LaunchedEffect(version) {
        if (pvt.fase == MotorPvt.Fase.ESPERANDO) {
            delay(pvt.esperaMs)
            if (pvt.fase == MotorPvt.Fase.ESPERANDO) { pvt.mostrar(ahoraMs()); version++ }
        } else if (pvt.fase == MotorPvt.Fase.ESTIMULO) {
            while (true) { withFrameMillis { }; ahora = ahoraMs() }
        } else if (pvt.fase == MotorPvt.Fase.LISTO && pvt.reacciones.isNotEmpty()) {
            delay(900); pvt.siguiente(); version++
        }
    }

    fun tocar() {
        when (pvt.fase) {
            MotorPvt.Fase.LISTO -> if (pvt.reacciones.isEmpty()) { pvt.siguiente(); mensaje = "Espera el estímulo…" }
            MotorPvt.Fase.ESPERANDO -> { pvt.tocar(ahoraMs()); mensaje = "¡Muy pronto! Adelanto registrado. Espera el contador." }
            MotorPvt.Fase.ESTIMULO -> {
                val ms = pvt.tocar(ahoraMs())
                mensaje = "${ms} ms" + if ((ms ?: 0) >= 500) " — LAPSO" else ""
                if (pvt.fase == MotorPvt.Fase.TERMINADO) { motor.registrarPvt(); mensaje = "Prueba terminada." }
            }
            MotorPvt.Fase.TERMINADO -> {}
        }
        version++
    }

    val estimulo = fase == MotorPvt.Fase.ESTIMULO
    Box(
        Modifier.fillMaxWidth().height(260.dp).clip(RoundedCornerShape(24.dp))
            .background(if (estimulo) Color(0x33F87171) else Cabina.vidrio)
            .border(1.dp, if (estimulo) Cabina.rojo else Cabina.borde, RoundedCornerShape(24.dp))
            .clickable(enabled = fase != MotorPvt.Fase.TERMINADO) { tocar() },
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                when (fase) {
                    MotorPvt.Fase.ESTIMULO -> "${pvt.transcurrido(ahora).coerceAtLeast(0)}"
                    MotorPvt.Fase.LISTO -> if (pvt.reacciones.isEmpty()) "TOCA PARA EMPEZAR" else "···"
                    MotorPvt.Fase.ESPERANDO -> "···"
                    MotorPvt.Fase.TERMINADO -> "FIN"
                },
                style = Cabina.mono.copy(fontSize = if (estimulo) 64.sp else 22.sp, color = if (estimulo) Cabina.rojo else Cabina.texto,
                    fontWeight = FontWeight.Black),
            )
            Text("${pvt.reacciones.size}/${pvt.ensayos}", style = Cabina.mono.copy(fontSize = 12.sp, color = Cabina.suave))
        }
    }
    Text(mensaje, color = Cabina.texto, fontSize = 14.sp, textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth())
    if (fase == MotorPvt.Fase.TERMINADO) Boton("Repetir prueba") { pvt.reiniciar(); mensaje = "Toca para empezar."; version++ }
    tm.pvt?.let { r ->
        Vidrio(Modifier.fillMaxWidth(), borde = r.triage.color().copy(alpha = 0.5f)) {
            Row {
                Dato("Media", "${r.mediaMs} ms", Modifier.weight(1f))
                Dato("Mediana", "${r.medianaMs} ms", Modifier.weight(1f))
                Dato("Lapsos", "${r.lapsos}", Modifier.weight(1f), if (r.lapsos > 0) Cabina.ambar else Cabina.texto)
            }
            Osciloscopio(r.reacciones.map { it.toDouble() }, r.triage.color(), Modifier.fillMaxWidth().height(70.dp), umbral = 500.0,
                maxFijo = maxOf(600.0, r.reacciones.max().toDouble()))
            Text("${r.triage.etiqueta}: " + when (r.triage) {
                Triage.CRITICO -> "fatiga importante. No hagas tareas críticas sin relevo."
                Triage.ADVERTENCIA -> "alerta reducida. Descansa antes de maniobras delicadas."
                else -> "alerta normal."
            } + " Referencia: media < 355 ms y sin lapsos (≥ 500 ms).", color = Cabina.suave, fontSize = 11.sp, lineHeight = 15.sp)
        }
    }
}

// ── 3. Caja negra IMU ────────────────────────────────────────────────────────

@Composable
fun PanelImu(motor: AstroPulseEngine) {
    val tm by motor.estado.collectAsState()
    val serie = remember { mutableStateListOf<Double>() }
    LaunchedEffect(tm.imuG, tm.impactos.size) { serie.add(tm.imuG); if (serie.size > 200) serie.removeAt(0) }
    val umbral = motor.umbralImpacto()
    Row {
        Dato("Ahora", "${tm.imuG} g", Modifier.weight(1f))
        Dato("Pico", "${tm.imuPicoG} g", Modifier.weight(1f), tm.triageImu.color())
        Dato("Umbral", "${(umbral * 10).roundToInt() / 10.0} g", Modifier.weight(1f), Cabina.rojo)
    }
    Osciloscopio(serie.toList(), Cabina.acento, Modifier.fillMaxWidth().height(140.dp), umbral = umbral, maxFijo = maxOf(umbral * 1.3, serie.maxOrNull() ?: 0.0))
    Text("Acelerómetro siempre activo. Se separa la gravedad (${tm.entorno.gravedad} m/s² en ${tm.entorno.corto}) de la aceleración " +
        "del cuerpo; un golpe ≥ umbral se registra como impacto severo con hora y fuerza.", color = Cabina.suave, fontSize = 11.sp, lineHeight = 15.sp)
    Etiqueta("Registro de eventos")
    if (tm.impactos.isEmpty()) Text("Sin eventos. Prueba agitando el teléfono con fuerza.", color = Cabina.suave, fontSize = 13.sp)
    for (i in tm.impactos.reversed().take(15)) Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(8.dp).clip(CircleShape).background(if (i.severo) Cabina.rojo else Cabina.ambar))
        Spacer(Modifier.width(10.dp))
        Text("${i.g} g", style = Cabina.mono.copy(fontSize = 13.sp, color = Cabina.texto), modifier = Modifier.width(80.dp))
        Text(if (i.severo) "IMPACTO SEVERO" else "sacudida", color = if (i.severo) Cabina.rojo else Cabina.ambar, fontSize = 12.sp, modifier = Modifier.weight(1f))
        Text(hace(i.horaMs), color = Cabina.suave, fontSize = 11.sp)
    }
    Boton("Reiniciar caja negra", Cabina.suave) { motor.reiniciarImu() }
}

// ── 4. Manuales ──────────────────────────────────────────────────────────────

@Composable
fun PanelManuales(tm: MissionTelemetry, pedido: String?) {
    var busca by remember { mutableStateOf("") }
    var abierto by remember(pedido) { mutableStateOf(Manuales.todos.firstOrNull { it.id == pedido }) }
    val m = abierto
    if (m != null) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            LedTriage(m.triage)
            Text(m.titulo, color = Cabina.texto, fontWeight = FontWeight.Bold, fontSize = 17.sp, modifier = Modifier.weight(1f))
        }
        Etiqueta(m.categoria)
        m.pasos.forEachIndexed { i, p ->
            Row {
                Text("${i + 1}", style = Cabina.mono.copy(fontSize = 14.sp, color = Cabina.acento, fontWeight = FontWeight.Bold), modifier = Modifier.width(24.dp))
                Text(p, color = Cabina.texto, fontSize = 14.sp, lineHeight = 20.sp)
            }
        }
        if (m.farmacos.isNotEmpty()) {
            Etiqueta("Fármacos (estado por radiación)")
            for (n in m.farmacos) Farmacia.botiquin.firstOrNull { it.nombre.startsWith(n.take(10)) }?.let {
                val e = Farmacia.estado(it, tm.diaMision, tm.entorno)
                BarraCaducidad(it.nombre, e.diasRestantes, e.potencia, it.vidaDias, e.triage)
            }
        }
        Boton("Volver a la lista", Cabina.suave) { abierto = null }
        return
    }
    OutlinedTextField(busca, { busca = it }, Modifier.fillMaxWidth(), placeholder = { Text("Buscar: mareo, sangrado, fuga…") },
        singleLine = true, colors = campo())
    val lista = Manuales.todos.filter { busca.isBlank() || normalizar(busca).let { b -> b in normalizar(it.titulo) || it.claves.any { c -> b in c } } }
        .sortedByDescending { it.triage.orden }
    for (x in lista) Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(Cabina.vidrio).border(1.dp, Cabina.borde, RoundedCornerShape(14.dp))
            .clickable { abierto = x }.padding(14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(8.dp).clip(CircleShape).background(x.triage.color()))
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(x.titulo, color = Cabina.texto, fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
            Text("${x.categoria} · ${x.pasos.size} pasos", color = Cabina.suave, fontSize = 11.sp)
        }
    }
}

@Composable
private fun campo() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor = Cabina.acento, unfocusedBorderColor = Cabina.borde, focusedTextColor = Cabina.texto,
    unfocusedTextColor = Cabina.texto, cursorColor = Cabina.acento, focusedPlaceholderColor = Cabina.suave, unfocusedPlaceholderColor = Cabina.suave,
)

// ── 5. Bitácora de voz ───────────────────────────────────────────────────────

@Composable
fun PanelVoz() {
    val boveda = rememberBoveda()
    val grab = rememberGrabadora()
    var grabando by remember { mutableStateOf(false) }
    var inicio by remember { mutableLongStateOf(0L) }
    var nivel by remember { mutableStateOf(0f) }
    var reproduciendo by remember { mutableStateOf<String?>(null) }
    var aviso by remember { mutableStateOf<String?>(null) }
    var lista by remember { mutableStateOf(boveda.listar("voz_")) }
    LaunchedEffect(grabando) { while (grabando) { nivel = grab.nivel(); delay(80) } }
    if (!grab.disponible || !boveda.disponible) { Text("Grabación cifrada no disponible en este dispositivo.", color = Cabina.ambar); return }

    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
        Box(
            Modifier.size(140.dp).clip(CircleShape).background((if (grabando) Cabina.rojo else Cabina.acento).copy(alpha = 0.12f + 0.3f * nivel))
                .border(2.dp, if (grabando) Cabina.rojo else Cabina.acento, CircleShape)
                .clickable {
                    if (!grabando) {
                        if (grab.iniciar()) { grabando = true; inicio = ahoraMs(); aviso = null }
                        else aviso = "Falta el permiso del micrófono."
                    } else {
                        grabando = false
                        grab.detener()?.let { audio ->
                            val a = boveda.guardar("voz_${ahoraMs()}.m4a.enc", audio)
                            aviso = "Guardado y cifrado · ${a.bytes / 1024} KB · huella ${a.huella}"
                        }
                        lista = boveda.listar("voz_")
                    }
                },
            contentAlignment = Alignment.Center,
        ) { Icon(if (grabando) Ico.stop else Ico.voz, null, tint = if (grabando) Cabina.rojo else Cabina.acento, modifier = Modifier.size(48.dp)) }
    }
    Text(if (grabando) "Grabando… ${(ahoraMs() - inicio) / 1000} s" else "Toca para grabar una entrada de bitácora",
        color = Cabina.texto, textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth())
    aviso?.let { Text(it, style = Cabina.mono.copy(fontSize = 11.sp, color = Cabina.verde), textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth()) }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(Ico.candado, null, tint = Cabina.verde, modifier = Modifier.size(14.dp))
        Spacer(Modifier.width(6.dp))
        Text("AES-256-GCM · llave en el chip seguro del teléfono · el audio sin cifrar se borra al terminar",
            color = Cabina.suave, fontSize = 11.sp)
    }
    Etiqueta("Entradas (${lista.size})")
    for (a in lista) Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(Cabina.vidrio).padding(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text("Bitácora · ${hace(a.horaMs)}", color = Cabina.texto, fontSize = 13.sp)
            Text("${a.bytes / 1024} KB · ${a.huella}", style = Cabina.mono.copy(fontSize = 10.sp, color = Cabina.suave))
        }
        Icon(if (reproduciendo == a.nombre) Ico.stop else Ico.play, "Reproducir", tint = Cabina.acento,
            modifier = Modifier.size(36.dp).clip(CircleShape).clickable {
                if (reproduciendo == a.nombre) { grab.pararReproduccion(); reproduciendo = null }
                else boveda.leer(a.nombre)?.let { audio -> reproduciendo = a.nombre; grab.reproducir(audio) { reproduciendo = null } }
            }.padding(8.dp))
        Icon(Ico.basura, "Borrar", tint = Cabina.suave, modifier = Modifier.size(36.dp).clip(CircleShape).clickable {
            boveda.borrar(a.nombre); lista = boveda.listar("voz_")
        }.padding(8.dp))
    }
}

// ── 6. Estado del sistema ────────────────────────────────────────────────────

@Composable
fun PanelSistema(motor: AstroPulseEngine, pulso: EstadoPulso, hist: EstadoHistorial) {
    val tm by motor.estado.collectAsState()
    val d = rememberDispositivo()
    var tic by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) { while (true) { delay(5000); tic++ } }
    val bat = remember(tic) { d.bateria() }
    Vidrio(Modifier.fillMaxWidth()) {
        Etiqueta("Autonomía")
        Row {
            Dato("Batería", bat?.let { "$it %" } ?: "--", Modifier.weight(1f),
                if ((bat ?: 100) < 20) Cabina.rojo else if ((bat ?: 100) < 40) Cabina.ambar else Cabina.verde)
            Dato("Carga", if (remember(tic) { d.cargando() }) "Sí" else "No", Modifier.weight(1f))
            Dato("Libre", remember(tic) { d.libreMb() }?.let { "${it / 1024} GB" } ?: "--", Modifier.weight(1f))
        }
        Text(d.modelo(), color = Cabina.suave, fontSize = 11.sp)
    }
    Vidrio(Modifier.fillMaxWidth()) {
        Etiqueta("Sensores y enlaces")
        for ((n, ok) in d.sensores()) Fila(n, if (ok) "OK" else "NO", if (ok) Cabina.verde else Cabina.rojo)
        Fila("Pulsera Xiaomi", when (pulso) { is EstadoPulso.Midiendo -> "EN VIVO"; is EstadoPulso.Buscando, is EstadoPulso.Conectando -> "BUSCANDO"; else -> "DESCONECTADA" },
            if (pulso is EstadoPulso.Midiendo) Cabina.verde else Cabina.ambar)
        Fila("Historial Health Connect", when (hist) { is EstadoHistorial.Listo -> "LISTO"; EstadoHistorial.SinDatos -> "VACÍO"; EstadoHistorial.SinPermiso -> "SIN PERMISO"; else -> "—" },
            if (hist is EstadoHistorial.Listo) Cabina.verde else Cabina.ambar)
        Fila("Asistente CDSS", "LOCAL", Cabina.verde)
        Fila("Manuales", "${Manuales.todos.size} OFFLINE", Cabina.verde)
    }
    Vidrio(Modifier.fillMaxWidth()) {
        Etiqueta("Misión")
        Fila("Entorno", tm.entorno.nombre, Cabina.texto)
        Fila("Gravedad", "${tm.entorno.gravedad} m/s² (${(tm.entorno.g * 100).roundToInt() / 100.0} g)", Cabina.texto)
        Fila("Retraso con la Tierra", Latencia.texto(tm.latenciaSeg) + " ida", Cabina.texto)
        if (tm.entorno == com.temrarios.ojoscan.astro.Entorno.MARTE) {
            val ua = Latencia.distanciaMarteUa(ahoraMs())
            Fila("Distancia Tierra–Marte", "${(ua * 100).roundToInt() / 100.0} UA · ${(ua * 149.6).roundToInt()} M km", Cabina.texto)
        }
        Fila("Dosis de misión", "${(tm.dosisAcumuladaMsv * 10).roundToInt() / 10.0} mSv de ${Farmacia.LIMITE_CARRERA_MSV.toInt()}", Cabina.texto)
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Día ${tm.diaMision}", color = Cabina.texto, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
            Boton("−30", Cabina.suave, Modifier.width(64.dp)) { motor.fijarDiaMision(tm.diaMision - 30) }
            Boton("+30", Cabina.acento, Modifier.width(64.dp)) { motor.fijarDiaMision(tm.diaMision + 30) }
            Boton("+180", Cabina.acento, Modifier.width(72.dp)) { motor.fijarDiaMision(tm.diaMision + 180) }
        }
        Text("Ajusta el día para planear: la caducidad por radiación y la dosis se recalculan.", color = Cabina.suave, fontSize = 11.sp)
    }
}

@Composable
private fun Fila(n: String, v: String, c: Color) = Row(Modifier.fillMaxWidth()) {
    Text(n, color = Cabina.suave, fontSize = 13.sp, modifier = Modifier.weight(1f))
    Text(v, style = Cabina.mono.copy(fontSize = 12.sp, color = c, fontWeight = FontWeight.SemiBold))
}

// ── Matriz farmacéutica ──────────────────────────────────────────────────────

@Composable
fun PanelFarmacia(tm: MissionTelemetry) {
    Text("Cada día en ${tm.entorno.corto} (${tm.entorno.dosisMsvDia} mSv/día) envejece a los fármacos más rápido según su " +
        "sensibilidad a la radiación (Du et al., 2011: 9 de 35 fármacos de la ISS fallaron potencia a los 880 días). " +
        "Día de misión ${tm.diaMision}.", color = Cabina.suave, fontSize = 12.sp, lineHeight = 17.sp)
    for (e in Farmacia.matriz(tm.diaMision, tm.entorno)) Vidrio(Modifier.fillMaxWidth(), borde = e.triage.color().copy(alpha = 0.3f)) {
        BarraCaducidad(e.farmaco.nombre, e.diasRestantes, e.potencia, e.farmaco.vidaDias, e.triage)
        Text("${e.farmaco.uso} · sensibilidad ${e.farmaco.sensibilidad.texto}", color = Cabina.suave, fontSize = 11.sp)
    }
}

// ── Baliza SOS ───────────────────────────────────────────────────────────────

@Composable
fun PanelSos(motor: AstroPulseEngine) {
    val boveda = rememberBoveda()
    val tm by motor.estado.collectAsState()
    val uri = LocalUriHandler.current
    val balizas = remember { boveda.listar("sos_") }
    val ultima = balizas.firstOrNull()
    Vidrio(Modifier.fillMaxWidth(), borde = Cabina.rojo.copy(alpha = 0.6f)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            LedTriage(Triage.CRITICO, 12.dp)
            Text("BALIZA ACTIVADA", style = Cabina.mono.copy(fontSize = 18.sp, color = Cabina.rojo, fontWeight = FontWeight.Black))
        }
        if (ultima != null) Text("Reporte de emergencia cifrado guardado en el teléfono.\n${ultima.bytes} bytes · huella ${ultima.huella}",
            style = Cabina.mono.copy(fontSize = 11.sp, color = Cabina.texto))
        else Text("No se pudo cifrar en este dispositivo: el reporte queda solo en pantalla.", color = Cabina.ambar, fontSize = 12.sp)
        Text("Incluye: signos vitales, triage, impactos de la IMU, PVT, dosis de radiación, fármacos caducados y las últimas consultas al asistente. " +
            "Con ${Latencia.texto(tm.latenciaSeg)} de retraso, la Tierra lo recibirá cuando haya enlace.", color = Cabina.suave, fontSize = 12.sp, lineHeight = 17.sp)
    }
    Etiqueta("Primeros pasos")
    Text(Cdss.resumen(tm).texto, color = Cabina.texto, fontSize = 13.sp, lineHeight = 19.sp)
    Boton("Llamar a emergencias (911)", Cabina.rojo) { uri.openUri("tel:911") }
    Etiqueta("Balizas guardadas (${balizas.size})")
    for (b in balizas.take(10)) Row {
        Text(hace(b.horaMs), color = Cabina.texto, fontSize = 12.sp, modifier = Modifier.weight(1f))
        Text(b.huella, style = Cabina.mono.copy(fontSize = 10.sp, color = Cabina.suave))
    }
}

// ── IA asistente ─────────────────────────────────────────────────────────────

@Composable
fun PanelIa(motor: AstroPulseEngine, alVerManual: (String) -> Unit) {
    val chat by motor.chat.collectAsState()
    var texto by remember { mutableStateOf("") }
    for (m in chat.takeLast(14)) {
        val c = if (m.deAstronauta) Cabina.acento else m.triage.color()
        Column(
            Modifier.fillMaxWidth().padding(start = if (m.deAstronauta) 48.dp else 0.dp, end = if (m.deAstronauta) 0.dp else 24.dp)
                .clip(RoundedCornerShape(16.dp)).background(c.copy(alpha = 0.08f)).border(1.dp, c.copy(alpha = 0.3f), RoundedCornerShape(16.dp))
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Etiqueta(if (m.deAstronauta) "Tú" else "CDSS · ${m.triage.etiqueta}", c, 8)
            Text(m.texto, color = Cabina.texto, fontSize = 13.sp, lineHeight = 19.sp)
            m.manualId?.let { id -> Text("Ver procedimiento completo →", color = Cabina.acento, fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                modifier = Modifier.clickable { alVerManual(id) }) }
        }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for (s in Cdss.sugerencias.take(3)) Sugerencia(s, Modifier.weight(1f)) { motor.preguntar(s) }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for (s in Cdss.sugerencias.drop(3)) Sugerencia(s, Modifier.weight(1f)) { motor.preguntar(s) }
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(texto, { texto = it }, Modifier.weight(1f), placeholder = { Text("Describe lo que sientes…") }, colors = campo(), maxLines = 3)
        Spacer(Modifier.width(8.dp))
        Box(Modifier.size(52.dp).clip(CircleShape).background(Cabina.acento.copy(alpha = 0.2f))
            .clickable { motor.preguntar(texto); texto = "" }, contentAlignment = Alignment.Center) {
            Icon(Ico.enviar, "Enviar", tint = Cabina.acento, modifier = Modifier.size(22.dp))
        }
    }
    Text("Responde con reglas clínicas y tu telemetría, en el teléfono y sin conexión.", color = Cabina.suave, fontSize = 10.sp)
}

@Composable
private fun Sugerencia(t: String, modifier: Modifier, alPulsar: () -> Unit) {
    Box(modifier.clip(RoundedCornerShape(12.dp)).background(Cabina.vidrioAlto).clickable(onClick = alPulsar).padding(8.dp),
        contentAlignment = Alignment.Center) { Text(t, color = Cabina.texto, fontSize = 11.sp, textAlign = TextAlign.Center, maxLines = 2) }
}
