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
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.temrarios.ojoscan.astro.AstroPulseEngine
import com.temrarios.ojoscan.astro.EstadoRevision
import com.temrarios.ojoscan.astro.Mensaje
import com.temrarios.ojoscan.astro.Registro
import com.temrarios.ojoscan.astro.Seguimiento
import com.temrarios.ojoscan.astro.Triage
import com.temrarios.ojoscan.astro.ahoraMs
import com.temrarios.ojoscan.platform.desfaseHorarioMs

private val PREGUNTAS = listOf("¿Cuándo me toca revisión?", "¿Cómo va mi pulso este mes?", "¿Tengo alertas?",
    "¿Cómo ha estado mi diario esta semana?", "¿Cómo va mi anemia?", "¿Cuándo tuve fiebre?")

private val FILTROS = listOf(null to "Todo", "pulso" to "Pulso", "spo2" to "SpO₂", "diario" to "Diario", "anemia" to "Anemia",
    "pvt" to "PVT", "ojo" to "Ojo", "rostro" to "Rostro", "perfil" to "Perfil", "impacto" to "Impactos")

private fun fecha(ms: Long): String {
    val l = ms + desfaseHorarioMs(ms)
    val dias = (l / 86_400_000L)
    // fecha civil a partir de días desde 1970 (algoritmo de Howard Hinnant)
    val z = dias + 719468; val era = (if (z >= 0) z else z - 146096) / 146097; val doe = z - era * 146097
    val yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; val doy = doe - (365 * yoe + yoe / 4 - yoe / 100)
    val mp = (5 * doy + 2) / 153; val d = doy - (153 * mp + 2) / 5 + 1; val m = if (mp < 10) mp + 3 else mp - 9
    val h = (l / 3_600_000).mod(24L); val mi = (l / 60_000).mod(60L)
    return "${d.toString().padStart(2, '0')}/${m.toString().padStart(2, '0')} ${h.toString().padStart(2, '0')}:${mi.toString().padStart(2, '0')}"
}

/** Tarjeta del tablero: cuántas revisiones tocan y la peor alerta. */
@Composable
fun TarjetaHistorial(motor: AstroPulseEngine, alAbrir: () -> Unit) {
    val h by motor.historial.collectAsState()
    val alertas by motor.alertas.collectAsState()
    val revs = remember(h) { motor.revisiones() }
    val pendientes = revs.count { it.estado == EstadoRevision.VENCIDA || it.estado == EstadoRevision.PRONTO }
    val peor = Triage.peor(*alertas.map { it.triage }.toTypedArray()).takeIf { it != Triage.SIN_DATOS && alertas.any { a -> a.triage != Triage.NOMINAL } }
    Vidrio(Modifier.fillMaxWidth().clickable(onClick = alAbrir), borde = (peor?.color() ?: Cabina.acento).copy(alpha = 0.6f)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("MI HISTORIAL", color = Cabina.texto, fontWeight = FontWeight.Black, fontSize = 18.sp)
                Text("${h.size} mediciones · " + if (pendientes > 0) "$pendientes revisión(es) pendiente(s)" else "revisiones al día",
                    color = if (pendientes > 0) Cabina.ambar else Cabina.suave, fontSize = 12.sp)
                alertas.firstOrNull { it.triage != Triage.NOMINAL }?.let {
                    Text("⚠ ${it.titulo}", color = it.triage.color(), fontSize = 12.sp, fontWeight = FontWeight.SemiBold, maxLines = 1)
                }
            }
            Semaforo3(peor ?: if (h.isEmpty()) null else Triage.NOMINAL, 16)
        }
    }
}

@Composable
fun PanelHistorial(motor: AstroPulseEngine) {
    val h by motor.historial.collectAsState()
    val alertas by motor.alertas.collectAsState()
    val revs = remember(h) { motor.revisiones() }
    var filtro by remember { mutableStateOf<String?>(null) }
    var pregunta by remember { mutableStateOf("") }
    val respuestas = remember { mutableStateListOf<Pair<String, Mensaje>>() }
    val ahora = ahoraMs()

    // ── Alertas ──
    val activas = alertas.filter { it.triage != Triage.NOMINAL }
    Etiqueta("Alertas (${activas.size})")
    if (activas.isEmpty()) Text("Sin alertas. Se revisan valores fuera de rango y tendencias contra tu propio historial.", color = Cabina.suave, fontSize = 12.sp)
    for (a in alertas.take(6)) {
        val c = a.triage.color().let { if (a.triage == Triage.NOMINAL) Cabina.suave else it }
        Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(c.copy(alpha = 0.1f))
            .border(1.dp, c.copy(alpha = 0.5f), RoundedCornerShape(14.dp)).padding(12.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(a.titulo, color = c, fontWeight = FontWeight.Bold, fontSize = 14.sp)
            Text(a.detalle, color = Cabina.texto, fontSize = 12.sp, lineHeight = 17.sp)
        }
    }

    // ── Recordatorios ──
    Etiqueta("Revisiones periódicas")
    for (r in revs) {
        val c = when (r.estado) { EstadoRevision.VENCIDA -> Cabina.rojo; EstadoRevision.PRONTO -> Cabina.ambar; EstadoRevision.NUNCA -> Cabina.suave; else -> Cabina.verde }
        Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(Cabina.vidrio).padding(10.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(10.dp).clip(CircleShape).background(c))
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text(r.tipo.nombre, color = Cabina.texto, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                Text(r.tipo.como + " Cada ${r.tipo.diasNormal} día(s); más seguido si sale en amarillo o rojo.", color = Cabina.suave, fontSize = 10.sp, lineHeight = 13.sp)
            }
            Text(r.texto, color = c, fontSize = 12.sp, fontWeight = FontWeight.Bold)
        }
    }

    // ── RAG personal ──
    Etiqueta("Pregúntale a tu historial")
    for ((q, m) in respuestas.takeLast(4)) {
        Text("Tú: $q", color = Cabina.acento, fontSize = 12.sp)
        Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(m.triage.color().copy(alpha = 0.08f))
            .border(1.dp, m.triage.color().copy(alpha = 0.35f), RoundedCornerShape(14.dp)).padding(12.dp)) {
            Text(m.texto, color = Cabina.texto, fontSize = 13.sp, lineHeight = 19.sp)
        }
    }
    fun preguntar(q: String) { if (q.isNotBlank()) { respuestas += q to motor.consultarHistorial(q); pregunta = "" } }
    for (fila in PREGUNTAS.chunked(2)) Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for (q in fila) Rapido(q, Modifier.weight(1f)) { preguntar(q) }
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(pregunta, { pregunta = it }, Modifier.weight(1f), singleLine = true, colors = campoDiario(),
            placeholder = { Text("¿cómo va mi oxígeno esta semana?") })
        Spacer(Modifier.width(8.dp))
        Box(Modifier.size(52.dp).clip(CircleShape).background(Cabina.acento.copy(alpha = 0.2f)).clickable { preguntar(pregunta) },
            contentAlignment = Alignment.Center) { Icon(Ico.enviar, "Preguntar", tint = Cabina.acento, modifier = Modifier.size(22.dp)) }
    }

    // ── Gráfica del pulso (30 días) ──
    val pulsos = h.filter { it.tipo == "pulso" && it.valor != null && ahora - it.horaMs <= 30 * Seguimiento.DIA }
    if (pulsos.size >= 2) {
        Etiqueta("Pulso · últimos 30 días (${pulsos.size})")
        Grafica(pulsos)
    }

    // ── Lista clasificada ──
    Etiqueta("Todas las mediciones")
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for ((k, n) in FILTROS.filter { (k, _) -> k == null || h.any { it.tipo == k } }.take(6)) {
            val sel = filtro == k
            Box(Modifier.clip(RoundedCornerShape(10.dp)).background(if (sel) Cabina.acento.copy(alpha = 0.25f) else Cabina.vidrio)
                .clickable { filtro = k }.padding(horizontal = 10.dp, vertical = 6.dp)) {
                Text(n, color = if (sel) Cabina.texto else Cabina.suave, fontSize = 11.sp)
            }
        }
    }
    val lista = h.filter { filtro == null || it.tipo == filtro }.asReversed().take(60)
    if (lista.isEmpty()) Text("Aún no hay mediciones aquí.", color = Cabina.suave, fontSize = 12.sp)
    for (r in lista) FilaRegistro(r) { motor.borrarRegistro(r.horaMs, r.tipo) }
    Text("Todo se guarda solo en este teléfono. Orientación, no diagnóstico: ante una alerta roja llama al 911.", color = Cabina.suave, fontSize = 10.sp)
}

@Composable
private fun FilaRegistro(r: Registro, alBorrar: () -> Unit) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(Cabina.vidrio).padding(10.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(10.dp).clip(CircleShape).background(r.triage.color()))
        Spacer(Modifier.width(10.dp))
        Column(Modifier.weight(1f)) {
            Text("${r.nombre} · ${fecha(r.horaMs)}", color = Cabina.suave, fontSize = 10.sp)
            Text(if (r.tipo == "diario") r.texto.ifBlank { "Ánimo ${r.valor?.toInt() ?: "-"}/5" } else Seguimiento.fmt(r),
                color = Cabina.texto, fontSize = 13.sp, maxLines = 2)
        }
        Icon(Ico.basura, "Borrar", tint = Cabina.suave, modifier = Modifier.size(18.dp).clickable(onClick = alBorrar))
    }
}

@Composable
private fun Grafica(p: List<Registro>) {
    val vals = p.map { it.valor!! }
    val lo = minOf(vals.min(), 50.0) - 5; val hi = maxOf(vals.max(), 110.0) + 5
    val t0 = p.first().horaMs; val t1 = maxOf(p.last().horaMs, t0 + 1)
    Canvas(Modifier.fillMaxWidth().height(120.dp).clip(RoundedCornerShape(12.dp)).background(Color(0xFF0B0E15)).padding(8.dp)) {
        fun y(v: Double) = (size.height * (1 - (v - lo) / (hi - lo))).toFloat()
        fun x(t: Long) = (size.width * (t - t0).toDouble() / (t1 - t0)).toFloat()
        // banda normal 50–110 lpm
        drawRect(Cabina.verde.copy(alpha = 0.08f), Offset(0f, y(110.0)), androidx.compose.ui.geometry.Size(size.width, y(50.0) - y(110.0)))
        for (i in 1 until p.size) drawLine(Cabina.acento.copy(alpha = 0.6f), Offset(x(p[i - 1].horaMs), y(vals[i - 1])), Offset(x(p[i].horaMs), y(vals[i])), 2f)
        for ((i, r) in p.withIndex()) drawCircle(r.triage.color(), 4f, Offset(x(r.horaMs), y(vals[i])))
    }
}
