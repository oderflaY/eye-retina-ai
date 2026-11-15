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
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.temrarios.ojoscan.astro.AstroPulseEngine
import com.temrarios.ojoscan.astro.EntradaDiario
import com.temrarios.ojoscan.astro.Recuperado
import com.temrarios.ojoscan.astro.Triage
import com.temrarios.ojoscan.astro.ahoraMs
import com.temrarios.ojoscan.platform.desfaseHorarioMs

private val ANIMOS = listOf(1 to "Muy mal", 2 to "Mal", 3 to "Regular", 4 to "Bien", 5 to "Muy bien")
private val RAPIDOS = listOf("Me siento bien", "Me duele la cabeza", "Tengo fiebre", "Me corté", "Me quemé", "Tengo mareo")

private const val DIA = 86_400_000L
private fun diaLocal(ms: Long) = (ms + desfaseHorarioMs(ms)).floorDiv(DIA)

private fun horaTexto(ms: Long): String {
    val local = ms + desfaseHorarioMs(ms)
    val min = (local / 60_000).mod(60L); val h = (local / 3_600_000).mod(24L)
    return "${h.toString().padStart(2, '0')}:${min.toString().padStart(2, '0')}"
}

private fun diaTexto(ms: Long): String {
    val d = diaLocal(ahoraMs()) - diaLocal(ms)
    return when (d) { 0L -> "Hoy"; 1L -> "Ayer"; else -> "Hace $d días" }
}

/** Color del día: el peor de sus entradas. */
private fun colorDelDia(entradas: List<EntradaDiario>, dia: Long): Triage? =
    entradas.filter { diaLocal(it.horaMs) == dia }.map { it.triage }.takeIf { it.isNotEmpty() }?.let { Triage.peor(*it.toTypedArray()) }

/** Semáforo de tres luces; la encendida brilla. */
@Composable
fun Semaforo3(t: Triage?, tam: Int = 30) {
    Row(Modifier.clip(RoundedCornerShape(50)).background(Color(0xFF0B0E15)).border(1.dp, Cabina.borde, RoundedCornerShape(50))
        .padding(horizontal = 10.dp, vertical = 6.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        for (luz in listOf(Triage.CRITICO, Triage.ADVERTENCIA, Triage.NOMINAL)) {
            val on = luz == t
            Canvas(Modifier.size(tam.dp)) {
                if (on) drawCircle(luz.color().copy(alpha = 0.3f), size.minDimension / 2)
                drawCircle(if (on) luz.color() else luz.color().copy(alpha = 0.13f), size.minDimension / 2.6f)
            }
        }
    }
}

/** Tira de los últimos 14 días (un punto por día). */
@Composable
private fun TiraDias(entradas: List<EntradaDiario>, dias: Int = 14) {
    val hoy = diaLocal(ahoraMs())
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        for (i in dias - 1 downTo 0) {
            val t = colorDelDia(entradas, hoy - i)
            Box(Modifier.size(14.dp).clip(CircleShape).background(t?.color() ?: Color(0x14FFFFFF))
                .then(if (i == 0) Modifier.border(1.5.dp, Cabina.texto, CircleShape) else Modifier))
        }
    }
}

@Composable
fun TarjetaDiario(motor: AstroPulseEngine, alAbrir: () -> Unit) {
    val diario by motor.diario.collectAsState()
    val hoy = colorDelDia(diario, diaLocal(ahoraMs()))
    Vidrio(Modifier.fillMaxWidth().clickable(onClick = alAbrir), borde = (hoy?.color() ?: Cabina.acento).copy(alpha = 0.7f)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("DIARIO DE SALUD", color = Cabina.texto, fontWeight = FontWeight.Black, fontSize = 18.sp)
                Text(when (hoy) {
                    null -> "¿Cómo te sientes hoy? Escríbelo y te digo qué hacer."
                    Triage.CRITICO -> "Hoy: ROJO · revisa los pasos de emergencia"
                    Triage.ADVERTENCIA -> "Hoy: AMARILLO · atiéndelo hoy"
                    else -> "Hoy: VERDE · todo en orden"
                }, color = hoy?.color() ?: Cabina.suave, fontSize = 12.sp, lineHeight = 15.sp)
            }
            Semaforo3(hoy, 18)
        }
        TiraDias(diario)
    }
}

@Composable
fun PanelDiario(motor: AstroPulseEngine) {
    val diario by motor.diario.collectAsState()
    val ev by motor.evaluacion.collectAsState()
    val base by motor.base.collectAsState()
    val uri = LocalUriHandler.current
    var texto by remember { mutableStateOf("") }
    var animo by remember { mutableStateOf<Int?>(null) }
    var busca by remember { mutableStateOf("") }
    var resultados by remember { mutableStateOf<List<Recuperado>>(emptyList()) }

    val actual = ev?.first
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Semaforo3(actual?.triage ?: colorDelDia(diario, diaLocal(ahoraMs())), 44)
    }

    Etiqueta("¿Cómo te sientes hoy?")
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for ((n, nombre) in ANIMOS) {
            val sel = animo == n
            val c = when (n) { 1 -> Cabina.rojo; 2 -> Cabina.ambar; 3 -> Cabina.suave; else -> Cabina.verde }
            Box(Modifier.weight(1f).height(40.dp).clip(RoundedCornerShape(12.dp)).background(c.copy(alpha = if (sel) 0.3f else 0.08f))
                .border(1.dp, c.copy(alpha = if (sel) 0.9f else 0.3f), RoundedCornerShape(12.dp)).clickable { animo = if (sel) null else n },
                contentAlignment = Alignment.Center) {
                Text(nombre, color = if (sel) Cabina.texto else Cabina.suave, fontSize = 10.sp, textAlign = TextAlign.Center, maxLines = 2, lineHeight = 11.sp)
            }
        }
    }
    OutlinedTextField(texto, { texto = it }, Modifier.fillMaxWidth(), minLines = 3, maxLines = 6, colors = campoDiario(),
        placeholder = { Text("Ej.: desde ayer tengo fiebre de 38.5 y me duele la cabeza") })
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for (r in RAPIDOS.take(3)) Rapido(r, Modifier.weight(1f)) { texto = if (texto.isBlank()) r else "$texto. $r" }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for (r in RAPIDOS.drop(3)) Rapido(r, Modifier.weight(1f)) { texto = if (texto.isBlank()) r else "$texto. $r" }
    }
    val listo = texto.isNotBlank() || animo != null
    Box(Modifier.fillMaxWidth().height(52.dp).clip(RoundedCornerShape(14.dp)).background(Cabina.acento.copy(alpha = if (listo) 0.2f else 0.05f))
        .border(1.dp, Cabina.acento.copy(alpha = if (listo) 0.6f else 0.2f), RoundedCornerShape(14.dp))
        .clickable(enabled = listo) { motor.escribirDiario(texto, animo); texto = ""; animo = null },
        contentAlignment = Alignment.Center) {
        Text("Guardar y evaluar", color = if (listo) Cabina.acento else Cabina.suave, fontWeight = FontWeight.Bold, fontSize = 15.sp)
    }

    // ── Resultado del semáforo ──
    if (actual != null) {
        val c = actual.triage.color()
        Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(c.copy(alpha = 0.12f))
            .border(1.dp, c.copy(alpha = 0.6f), RoundedCornerShape(16.dp)).padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(actual.titulo, color = c, fontWeight = FontWeight.Black, fontSize = 20.sp)
            Text(actual.accion, color = Cabina.texto, fontSize = 14.sp, lineHeight = 20.sp, fontWeight = FontWeight.SemiBold)
            for (m in actual.motivos) Text("• $m", color = Cabina.texto, fontSize = 13.sp)
            if (actual.triage == Triage.CRITICO) {
                val crisis = actual.sintomas.contains("crisis")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Llamar("Llamar al 911", Cabina.rojo, Modifier.weight(1f)) { runCatching { uri.openUri("tel:911") } }
                    if (crisis) Llamar("Línea de la Vida", Cabina.ambar, Modifier.weight(1f)) { runCatching { uri.openUri("tel:8009112000") } }
                }
            }
        }
        val docs = ev?.second.orEmpty()
        if (docs.isNotEmpty()) {
            val pasos = docs.filter { it.doc.esPrimerosAuxilios }
            val mas = docs.filterNot { it.doc.esPrimerosAuxilios }
            if (pasos.isNotEmpty()) Etiqueta("Qué hacer · primeros auxilios")
            for (r in pasos) TarjetaDoc(r, base?.temas?.get(r.doc.tema))
            if (mas.isNotEmpty()) Etiqueta("Para saber más · MedlinePlus")
            for (r in mas) TarjetaDoc(r, base?.temas?.get(r.doc.tema))
        } else if (actual.sintomas.isNotEmpty() && base == null) {
            Text("La base de primeros auxilios se está cargando…", color = Cabina.suave, fontSize = 12.sp)
        }
    }

    // ── Búsqueda libre en la base ──
    Etiqueta("Buscar en primeros auxilios (${base?.docs?.size ?: 0} procedimientos, sin conexión)")
    Row(verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(busca, { busca = it }, Modifier.weight(1f), singleLine = true, colors = campoDiario(),
            placeholder = { Text("picadura de alacrán, nariz sangra…") })
        Spacer(Modifier.width(8.dp))
        Box(Modifier.size(52.dp).clip(CircleShape).background(Cabina.acento.copy(alpha = 0.2f))
            .clickable { resultados = motor.buscarAuxilios(busca, 4) }, contentAlignment = Alignment.Center) {
            Icon(Ico.enviar, "Buscar", tint = Cabina.acento, modifier = Modifier.size(22.dp))
        }
    }
    for (r in resultados) TarjetaDoc(r, base?.temas?.get(r.doc.tema))
    if (resultados.isEmpty() && busca.isNotBlank()) Text("Pulsa buscar. Si no aparece nada, prueba con otras palabras.", color = Cabina.suave, fontSize = 11.sp)

    // ── Historial ──
    Etiqueta("Últimos 14 días")
    TiraDias(diario)
    for (e in diario.takeLast(30).reversed()) Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(Cabina.vidrio).padding(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(12.dp).clip(CircleShape).background(e.triage.color()))
        Spacer(Modifier.width(10.dp))
        Column(Modifier.weight(1f)) {
            Text("${diaTexto(e.horaMs)} · ${horaTexto(e.horaMs)}" + (e.animo?.let { a -> " · ${ANIMOS.first { it.first == a }.second}" } ?: ""),
                color = Cabina.suave, fontSize = 11.sp)
            if (e.texto.isNotBlank()) Text(e.texto, color = Cabina.texto, fontSize = 13.sp, maxLines = 3, lineHeight = 17.sp)
            if (e.motivos.isNotEmpty()) Text(e.motivos.joinToString(" · "), color = e.triage.color(), fontSize = 11.sp, maxLines = 2)
        }
        Icon(Ico.basura, "Borrar", tint = Cabina.suave, modifier = Modifier.size(20.dp).clickable { motor.borrarEntradaDiario(e.horaMs) })
    }
    Text("El color lo deciden reglas fijas de señales de alarma (no la IA). Las respuestas vienen de First-Aid-Dataset y FirstAidQA " +
        "(traducidas y filtradas) y de MedlinePlus en español (Biblioteca Nacional de Medicina de EE. UU.). " +
        "Orientación general: no sustituye a un médico. Emergencia: 911.",
        color = Cabina.suave, fontSize = 10.sp, lineHeight = 13.sp)
}

@Composable
private fun TarjetaDoc(r: Recuperado, tema: String?) {
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(Cabina.vidrioAlto)
        .border(1.dp, Cabina.borde, RoundedCornerShape(14.dp)).padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        tema?.let { Etiqueta(it, Cabina.acento, 9) }
        Text(r.doc.pregunta, color = Cabina.texto, fontWeight = FontWeight.Bold, fontSize = 14.sp, lineHeight = 18.sp)
        Text(r.doc.respuesta, color = Cabina.texto, fontSize = 13.sp, lineHeight = 19.sp)
        Text("Fuente: " + r.doc.etiquetaFuente, color = Cabina.suave, fontSize = 10.sp)
    }
}

@Composable
private fun Rapido(t: String, modifier: Modifier, alPulsar: () -> Unit) {
    Box(modifier.clip(RoundedCornerShape(12.dp)).background(Cabina.vidrioAlto).clickable(onClick = alPulsar).padding(8.dp),
        contentAlignment = Alignment.Center) { Text(t, color = Cabina.texto, fontSize = 11.sp, textAlign = TextAlign.Center, maxLines = 2) }
}

@Composable
private fun Llamar(t: String, c: Color, modifier: Modifier, alPulsar: () -> Unit) {
    Box(modifier.height(52.dp).clip(RoundedCornerShape(14.dp)).background(c.copy(alpha = 0.25f)).border(1.5.dp, c, RoundedCornerShape(14.dp))
        .clickable(onClick = alPulsar), contentAlignment = Alignment.Center) {
        Text(t, color = c, fontWeight = FontWeight.Black, fontSize = 15.sp)
    }
}

@Composable
private fun campoDiario() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor = Cabina.acento, unfocusedBorderColor = Cabina.borde, focusedTextColor = Cabina.texto,
    unfocusedTextColor = Cabina.texto, cursorColor = Cabina.acento, focusedPlaceholderColor = Cabina.suave, unfocusedPlaceholderColor = Cabina.suave,
)
