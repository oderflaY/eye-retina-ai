package com.temrarios.ojoscan.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
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
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.temrarios.ojoscan.api.AnalisisCorazon
import com.temrarios.ojoscan.api.DatosPerfil
import com.temrarios.ojoscan.api.HistorialCorazon
import com.temrarios.ojoscan.api.OjoApi
import com.temrarios.ojoscan.api.PeticionCorazon
import com.temrarios.ojoscan.api.explicar
import com.temrarios.ojoscan.historial.EstadoHistorial
import com.temrarios.ojoscan.historial.FuenteHistorial
import com.temrarios.ojoscan.historial.paraServidor
import com.temrarios.ojoscan.platform.rememberVoz
import com.temrarios.ojoscan.pulso.EstadoPulso
import com.temrarios.ojoscan.pulso.MonitorPulso
import com.temrarios.ojoscan.pulso.Muestra
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/** Síntomas que la persona puede marcar (clave del servidor → texto). */
private val SINTOMAS = listOf(
    "dolor_pecho" to "Dolor de pecho", "falta_aire" to "Falta de aire", "mareo" to "Mareo",
    "desmayo" to "Desmayo", "palpitaciones" to "Palpitaciones",
)

fun colorRiesgo(nivel: String): Color = when (nivel) {
    "urgente" -> Colores.mal
    "alto", "alta" -> Color(0xFFFF7A45)
    "vigilar", "media" -> Colores.atencion
    "ok" -> Colores.bien
    else -> Colores.cian
}

/** Pulso a 1 Hz (como lo entrena la IA): el último valor de cada segundo de los últimos [segundos]. */
fun serie1Hz(m: List<Muestra>, segundos: Int = 300): List<Int> {
    if (m.isEmpty()) return emptyList()
    val desde = m.last().tiempoMs - segundos * 1000L
    return m.filter { it.tiempoMs >= desde && it.lpm in 30..230 }
        .groupBy { it.tiempoMs / 1000 }.entries.sortedBy { it.key }.map { it.value.last().lpm }
}

private fun historialCorazon(e: EstadoHistorial): HistorialCorazon? =
    (e as? EstadoHistorial.Listo)?.resumen?.paraServidor()?.let {
        HistorialCorazon(it.pulsoReposo, it.pulsoReposo7d, it.pulsoReposoBase, it.spo2Media, it.spo2Min, it.spo2Bajas, it.suenoHoras)
    }

/**
 * Pantalla principal: telemetría de la pulsera Xiaomi + IA cardiaca para prevenir accidentes.
 * Con la pulsera conectada analiza cada 30 s los últimos 5 minutos y avisa (también con voz) si sube el riesgo.
 */
@Composable
fun PantallaCorazon(
    alVolver: () -> Unit,
    servidor: String,
    pulso: MonitorPulso,
    historial: FuenteHistorial,
    perfil: DatosPerfil,
    alEscaneos: () -> Unit,
    alTusDatos: () -> Unit,
) {
    val estado by pulso.estado.collectAsState()
    val muestras by pulso.muestras.collectAsState()
    val estadoHist by historial.estado.collectAsState()
    val alcance = rememberCoroutineScope()
    val voz = rememberVoz()
    val uri = LocalUriHandler.current

    var analisis by remember { mutableStateOf<AnalisisCorazon?>(null) }
    var cargando by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var sintomas by remember { mutableStateOf(setOf<String>()) }
    var enMovimiento by remember { mutableStateOf(false) }
    var vigilancia by remember { mutableStateOf(true) }

    val serie = serie1Hz(muestras)
    val midiendo = estado as? EstadoPulso.Midiendo

    fun analizar() {
        if (cargando) return
        cargando = true
        alcance.launch {
            val api = OjoApi(servidor)
            val antes = analisis?.riesgo
            runCatching {
                api.analizarCorazon(PeticionCorazon(
                    serie = serie1Hz(pulso.muestras.value), edad = perfil.edad, sexo = perfil.sexo,
                    actividad = if (enMovimiento) "movimiento" else "reposo", sintomas = sintomas.toList(),
                    historial = historialCorazon(historial.estado.value),
                ))
            }.fold(
                onSuccess = { r ->
                    analisis = r; error = null
                    // aviso hablado solo cuando el riesgo SUBE a alto/urgente
                    val orden = listOf("sin_datos", "ok", "vigilar", "alto", "urgente")
                    if (r.riesgo in listOf("alto", "urgente") && orden.indexOf(r.riesgo) > orden.indexOf(antes ?: "ok"))
                        r.alertas.firstOrNull()?.let { voz.decir("${r.titulo}. ${it.titulo}. ${it.accion}") }
                },
                onFailure = { error = explicar(it) },
            )
            api.cerrar()
            cargando = false
        }
    }
    val analizarActual by rememberUpdatedState(::analizar)

    // Vigilancia continua mientras la pulsera mide.
    LaunchedEffect(vigilancia, midiendo != null) {
        if (!vigilancia || midiendo == null) return@LaunchedEffect
        delay(15_000)
        while (true) {
            if (serie1Hz(pulso.muestras.value).size >= 60) analizarActual()
            delay(30_000)
        }
    }
    // Sin pulsera en vivo, al menos compara el historial de 30 días.
    LaunchedEffect(estadoHist) { if (estadoHist is EstadoHistorial.Listo && analisis == null) analizarActual() }

    FondoEspacial {
        Column(
            Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            // ── Encabezado ─────────────────────────────────────────────
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("◂ ASTROPULSE", style = Hud.etiqueta.copy(fontSize = 11.sp),
                        modifier = Modifier.clickable(onClick = alVolver).padding(vertical = 6.dp))
                    Text("MISIÓN CORAZÓN // TELEMETRÍA XIAOMI", style = Hud.etiqueta.copy(color = Colores.textoSuave, fontSize = 10.sp))
                    TextoDegradado("OjoScan", 38)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Led(if (midiendo != null) Colores.bien else Colores.textoSuave)
                    Text(if (midiendo != null) "EN ÓRBITA" else "SIN SEÑAL", style = Hud.etiqueta.copy(
                        fontSize = 10.sp, color = if (midiendo != null) Colores.bien else Colores.textoSuave))
                }
            }

            // ── Planeta corazón ────────────────────────────────────────
            val lpm = midiendo?.lpm?.takeIf { it > 0 }
            val riesgo = analisis?.riesgo ?: "sin_datos"
            Box(Modifier.fillMaxWidth().height(290.dp), contentAlignment = Alignment.Center) {
                PlanetaCorazon(lpm, colorRiesgo(riesgo), Modifier.size(280.dp))
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(lpm?.toString() ?: "--", color = Colores.texto, fontSize = 64.sp, fontWeight = FontWeight.Black)
                    Text("LATIDOS / MIN", style = Hud.etiqueta.copy(fontSize = 10.sp, color = Colores.textoSuave))
                }
            }
            Text(
                when (val e = estado) {
                    is EstadoPulso.Midiendo -> e.nombre + (if (e.contacto == false) " · ajusta la pulsera: no toca la piel" else " · señal en vivo por Bluetooth")
                    is EstadoPulso.Buscando -> e.detalle
                    is EstadoPulso.Conectando -> "Conectando con ${e.nombre}…"
                    EstadoPulso.SinPermiso -> "Falta el permiso de Bluetooth (Dispositivos cercanos)."
                    EstadoPulso.BluetoothApagado -> "Enciende el Bluetooth para recibir tu pulso."
                    is EstadoPulso.SinTransmision -> "${e.nombre} no transmite: activa \"Transmitir frecuencia cardíaca\" en la pulsera."
                    is EstadoPulso.Error -> e.mensaje
                    EstadoPulso.Inactivo -> "Tu Xiaomi Smart Band manda el pulso a la IA en tiempo real."
                },
                color = if (midiendo?.contacto == false || estado is EstadoPulso.Error || estado is EstadoPulso.SinTransmision) Colores.atencion else Colores.textoSuave,
                fontSize = 13.sp, textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth(),
            )
            if (midiendo == null && estado !is EstadoPulso.Buscando && estado !is EstadoPulso.Conectando)
                BotonNeon("Conectar pulsera", pulso::iniciar, Modifier.fillMaxWidth())

            // ── Alerta urgente ─────────────────────────────────────────
            analisis?.takeIf { it.riesgo == "urgente" }?.let { a ->
                PanelUrgente(a) { uri.openUri("tel:911") }
            }

            // ── Señal ──────────────────────────────────────────────────
            if (serie.isNotEmpty()) PanelVidrio(Modifier.fillMaxWidth()) {
                TituloHud("01", "Señal de telemetría")
                TrazoPulso(serie.takeLast(180), Modifier.fillMaxWidth().height(120.dp))
                val ult = serie.takeLast(180)
                Text("Últimos ${ult.size} s · mín ${ult.min()} · prom ${ult.average().roundToInt()} · máx ${ult.max()} lpm",
                    style = Hud.dato.copy(fontSize = 11.sp, color = Colores.cian))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Segmento("En reposo", !enMovimiento, { enMovimiento = false }, Modifier.weight(1f))
                    Segmento("Haciendo ejercicio", enMovimiento, { enMovimiento = true }, Modifier.weight(1f))
                }
            }

            // ── IA cardiaca ────────────────────────────────────────────
            PanelVidrio(Modifier.fillMaxWidth(), resaltado = analisis != null) {
                TituloHud("02", "IA cardiaca")
                val a = analisis
                if (a == null) {
                    Text(
                        if (serie.size in 1 until 60) "Reuniendo señal: ${serie.size} de 60 segundos. Quédate quieto con la pulsera ajustada."
                        else "La IA compara tu pulso con miles de horas de latidos reales (PhysioNet) de personas sanas, " +
                            "con arritmias, extrasístoles, insuficiencia cardiaca y de Holters previos a un paro cardiaco.",
                        color = Colores.textoSuave, fontSize = 13.sp, lineHeight = 18.sp,
                    )
                } else {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Led(colorRiesgo(a.riesgo), 10.dp)
                        Text(a.titulo, color = colorRiesgo(a.riesgo), fontSize = 22.sp, fontWeight = FontWeight.Black)
                    }
                    a.ia?.takeIf { it.modelos.isNotEmpty() }?.let { ia ->
                        for (m in ia.modelos) BarraNeon(m.nombre, m.probabilidad.toFloat(), colorRiesgo(m.nivel),
                            detalle = "Detecta ${(m.validacion.sensibilidad * 100).roundToInt()} % de los casos · " +
                                "${(m.validacion.especificidad * 100).roundToInt()} % de acierto en sanos · ${m.validacion.personas} personas")
                        Text("Basado en ${ia.ventanas} lecturas de tu pulsera. Más tiempo midiendo = resultado más firme.",
                            color = Colores.textoSuave, fontSize = 11.sp)
                    } ?: Text("Sin señal en vivo: el análisis usa solo tu historial de 30 días.", color = Colores.textoSuave, fontSize = 12.sp)
                }
                error?.let { Text(it, color = Colores.mal, fontSize = 12.sp) }
                InterruptorNeon("Vigilancia continua", vigilancia, { vigilancia = it },
                    detalle = "Analiza cada 30 s mientras la pulsera mide y te avisa en voz alta si sube el riesgo.")
                BotonContorno(if (cargando) "Analizando…" else "Analizar ahora", { analizar() }, Modifier.fillMaxWidth())
            }

            // ── Alertas ────────────────────────────────────────────────
            analisis?.alertas?.filter { analisis?.riesgo != "urgente" || it.nivel != "urgente" }?.takeIf { it.isNotEmpty() }?.let { lista ->
                PanelVidrio(Modifier.fillMaxWidth()) {
                    TituloHud("03", "Prevención")
                    for (al in lista) Alerta(al.nivel, al.titulo, al.texto, al.accion)
                }
            }

            // ── Síntomas ───────────────────────────────────────────────
            PanelVidrio(Modifier.fillMaxWidth()) {
                TituloHud("04", "¿Sientes algo ahora?")
                for (fila in SINTOMAS.chunked(3)) Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for ((k, t) in fila) Segmento(t, k in sintomas, {
                        sintomas = if (k in sintomas) sintomas - k else sintomas + k
                        analizar()
                    }, Modifier.weight(1f))
                    repeat(3 - fila.size) { Spacer(Modifier.weight(1f)) }
                }
                Text("Si tienes dolor de pecho, falta de aire o te desmayaste, no esperes a la app: llama al 911.",
                    color = Colores.textoSuave, fontSize = 11.sp, lineHeight = 15.sp)
            }

            // ── Comparación ────────────────────────────────────────────
            analisis?.comparacion?.takeIf { it.isNotEmpty() }?.let { lista ->
                PanelVidrio(Modifier.fillMaxWidth()) {
                    TituloHud("05", "Tú contra tu referencia")
                    for (c in lista) FilaComparacion(c.nombre, c.tuyo, c.referencia, c.nivel, c.texto)
                    if (perfil.edad == null) Text("Agrega tu edad en Tus datos para compararte con el límite de tu edad.",
                        color = Colores.textoSuave, fontSize = 11.sp)
                }
            }

            // ── Historial 30 días ──────────────────────────────────────
            PanelHistorial(historial)
            HistorialOrbital(estadoHist)

            BotonContorno("Tus datos (edad, presión…)", alTusDatos, Modifier.fillMaxWidth())
            BotonContorno("Escaneos: ojo, anemia, rostro, encía", alEscaneos, Modifier.fillMaxWidth())
            Text(analisis?.aviso?.takeIf { it.isNotBlank() }
                ?: "Herramienta de prevención, no un diagnóstico. Una pulsera no sustituye un electrocardiograma.",
                color = Colores.textoSuave.copy(alpha = 0.7f), fontSize = 10.sp, lineHeight = 13.sp,
                textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth())
            Spacer(Modifier.height(8.dp))
        }
    }
}

/** Planeta-corazón: late al ritmo real de la pulsera, con anillos orbitales y una luna. */
@Composable
private fun PlanetaCorazon(lpm: Int?, acento: Color, modifier: Modifier) {
    val latido = remember { Animatable(1f) }
    LaunchedEffect(lpm) {
        if (lpm == null) { latido.snapTo(1f); return@LaunchedEffect }
        val periodo = (60_000 / lpm.coerceIn(30, 220)).toLong()
        while (true) {
            latido.animateTo(1.07f, tween(110))
            latido.animateTo(1f, tween(220, easing = FastOutSlowInEasing))
            delay((periodo - 330).coerceAtLeast(40))
        }
    }
    val t = rememberInfiniteTransition()
    val giro by t.animateFloat(0f, 360f, infiniteRepeatable(tween(14000, easing = LinearEasing)))
    val giro2 by t.animateFloat(360f, 0f, infiniteRepeatable(tween(26000, easing = LinearEasing)))
    val respira by t.animateFloat(0.9f, 1.1f, infiniteRepeatable(tween(3000), RepeatMode.Reverse))
    val nucleo = if (lpm == null) Colores.violeta else Colores.mal
    Canvas(modifier) {
        val c = Offset(size.width / 2, size.height / 2)
        val r = size.minDimension / 2
        val rp = r * 0.5f * latido.value
        // halo
        drawCircle(Brush.radialGradient(listOf(nucleo.copy(alpha = 0.35f), Color.Transparent), c, r * respira), r * respira, c)
        // anillos orbitales inclinados
        rotate(-18f, c) {
            drawOval(acento.copy(alpha = 0.55f), Offset(c.x - r * 0.95f, c.y - r * 0.26f), Size(r * 1.9f, r * 0.52f),
                style = Stroke(1.5.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(14f, 10f))))
            val a = giro * PI.toFloat() / 180f
            val luna = Offset(c.x + cos(a) * r * 0.95f, c.y + sin(a) * r * 0.26f)
            if (sin(a) > -0.2f || cos(a).let { it * it } > 0.5f) {
                drawCircle(Colores.estrella.copy(alpha = 0.25f), 9.dp.toPx(), luna)
                drawCircle(Colores.estrella, 4.dp.toPx(), luna)
            }
        }
        rotate(24f, c) {
            drawOval(Colores.cian.copy(alpha = 0.35f), Offset(c.x - r * 0.8f, c.y - r * 0.18f), Size(r * 1.6f, r * 0.36f), style = Stroke(1.dp.toPx()))
        }
        // graduación exterior (como un radar)
        rotate(giro2, c) {
            for (i in 0 until 60) {
                val ang = i * 6f * PI.toFloat() / 180f
                val l = if (i % 5 == 0) 9.dp.toPx() else 4.dp.toPx()
                drawLine(acento.copy(alpha = if (i % 5 == 0) 0.7f else 0.25f),
                    Offset(c.x + cos(ang) * (r - l), c.y + sin(ang) * (r - l)), Offset(c.x + cos(ang) * r, c.y + sin(ang) * r), 1.2.dp.toPx())
            }
        }
        // planeta
        drawCircle(Brush.radialGradient(listOf(nucleo.copy(alpha = 0.95f), nucleo.copy(alpha = 0.5f), Color(0xFF1A0B2E)),
            Offset(c.x - rp * 0.35f, c.y - rp * 0.4f), rp * 1.6f), rp, c)
        // meridianos
        for (k in 1..3) {
            val w = rp * 2 * k / 4f
            drawOval(Color.White.copy(alpha = 0.08f), Offset(c.x - w / 2, c.y - rp), Size(w, rp * 2), style = Stroke(1.dp.toPx()))
        }
        drawOval(Color.White.copy(alpha = 0.08f), Offset(c.x - rp, c.y - rp * 0.3f), Size(rp * 2, rp * 0.6f), style = Stroke(1.dp.toPx()))
        drawCircle(acento.copy(alpha = 0.8f), rp, c, style = Stroke(1.5.dp.toPx()))
    }
}

/** Trazo del pulso como una línea de telemetría sobre una cuadrícula de constelación. */
@Composable
private fun TrazoPulso(serie: List<Int>, modifier: Modifier) {
    Canvas(modifier) {
        if (serie.size < 2) return@Canvas
        val lo = (serie.min() - 5).toFloat()
        val hi = (serie.max() + 5).toFloat()
        fun y(v: Int) = size.height * (1f - (v - lo) / (hi - lo))
        val dx = size.width / (serie.size - 1)
        for (i in 0..6) for (j in 0..3) drawCircle(Colores.textoSuave.copy(alpha = 0.18f), 1.dp.toPx(),
            Offset(size.width * i / 6f, size.height * j / 3f))
        val linea = Path().apply { serie.forEachIndexed { i, v -> if (i == 0) moveTo(0f, y(v)) else lineTo(i * dx, y(v)) } }
        val relleno = Path().apply { addPath(linea); lineTo(size.width, size.height); lineTo(0f, size.height); close() }
        drawPath(relleno, Brush.verticalGradient(listOf(Colores.magenta.copy(alpha = 0.28f), Color.Transparent)))
        drawPath(linea, Brush.horizontalGradient(listOf(Colores.violeta, Colores.magenta, Colores.cian)),
            style = Stroke(2.dp.toPx(), cap = StrokeCap.Round))
        drawCircle(Colores.estrella, 3.5.dp.toPx(), Offset(size.width, y(serie.last())))
        drawCircle(Colores.cian.copy(alpha = 0.3f), 8.dp.toPx(), Offset(size.width, y(serie.last())))
    }
}

@Composable
private fun PanelUrgente(a: AnalisisCorazon, alLlamar: () -> Unit) {
    val t = rememberInfiniteTransition()
    val pulso by t.animateFloat(0.4f, 1f, infiniteRepeatable(tween(700), RepeatMode.Reverse))
    val forma = RoundedCornerShape(22.dp)
    val al = a.alertas.firstOrNull { it.nivel == "urgente" }
    Column(
        Modifier.fillMaxWidth().clip(forma).background(Colores.mal.copy(alpha = 0.18f))
            .border(2.dp, Colores.mal.copy(alpha = pulso), forma).padding(18.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("ALERTA ROJA", style = Hud.etiqueta.copy(color = Colores.mal))
        Text(al?.titulo ?: a.titulo, color = Colores.texto, fontSize = 22.sp, fontWeight = FontWeight.Black)
        al?.let {
            Text(it.texto, color = Colores.texto, fontSize = 14.sp, lineHeight = 19.sp)
            Text(it.accion, color = Colores.atencion, fontSize = 14.sp, lineHeight = 19.sp, fontWeight = FontWeight.SemiBold)
        }
        androidx.compose.material3.TextButton(alLlamar,
            Modifier.fillMaxWidth().height(60.dp).clip(RoundedCornerShape(50)).background(Colores.mal)) {
            Text("LLAMAR AL 911", color = Color.White, fontWeight = FontWeight.Black, fontSize = 18.sp, letterSpacing = 3.sp)
        }
    }
}

@Composable
private fun Alerta(nivel: String, titulo: String, texto: String, accion: String) {
    val color = colorRiesgo(nivel)
    Row(Modifier.fillMaxWidth()) {
        Box(Modifier.width(3.dp).height(74.dp).clip(RoundedCornerShape(2.dp)).background(color))
        Spacer(Modifier.width(12.dp))
        Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(titulo, color = color, fontWeight = FontWeight.Bold, fontSize = 15.sp)
            Text(texto, color = Colores.texto, fontSize = 12.sp, lineHeight = 16.sp)
            Text("→ $accion", color = Colores.textoSuave, fontSize = 12.sp, lineHeight = 16.sp)
        }
    }
}

@Composable
private fun FilaComparacion(nombre: String, tuyo: String, referencia: String, nivel: String, texto: String) {
    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Text(nombre.uppercase(), style = Hud.etiqueta.copy(fontSize = 9.sp, color = Colores.textoSuave))
        Row(verticalAlignment = Alignment.Bottom) {
            Text(tuyo, color = colorRiesgo(nivel), fontSize = 20.sp, fontWeight = FontWeight.Bold)
            Text("  vs  $referencia", color = Colores.textoSuave, fontSize = 13.sp)
        }
        Text(texto, color = Colores.textoSuave, fontSize = 11.sp, lineHeight = 15.sp)
    }
}

/** Pulso en reposo de cada día como una órbita de puntos (los últimos 30 días). */
@Composable
private fun HistorialOrbital(e: EstadoHistorial) {
    val r = (e as? EstadoHistorial.Listo)?.resumen ?: return
    val dias = r.pulsoReposo.sortedBy { it.dia }.takeLast(30)
    if (dias.size < 3) return
    AnimatedVisibility(true, enter = fadeIn() + expandVertically()) {
        PanelVidrio(Modifier.fillMaxWidth()) {
            TituloHud("07", "Tu pulso en reposo, día por día")
            Canvas(Modifier.fillMaxWidth().height(110.dp)) {
                val vs = dias.map { it.valor.toFloat() }
                val lo = vs.min() - 3
                val hi = vs.max() + 3
                val dx = size.width / (vs.size - 1)
                val linea = Path()
                vs.forEachIndexed { i, v ->
                    val p = Offset(i * dx, size.height * (1f - (v - lo) / (hi - lo)))
                    if (i == 0) linea.moveTo(p.x, p.y) else linea.lineTo(p.x, p.y)
                }
                drawPath(linea, Colores.violeta.copy(alpha = 0.6f), style = Stroke(1.dp.toPx()))
                vs.forEachIndexed { i, v ->
                    val p = Offset(i * dx, size.height * (1f - (v - lo) / (hi - lo)))
                    val reciente = i >= vs.size - 7
                    drawCircle((if (reciente) Colores.magenta else Colores.cian).copy(alpha = 0.3f), 6.dp.toPx(), p)
                    drawCircle(if (reciente) Colores.magenta else Colores.estrella, 2.5.dp.toPx(), p)
                }
            }
            Text("${dias.first().dia} → ${dias.last().dia} · rosa = últimos 7 días · ${vs(dias)}",
                style = Hud.dato.copy(fontSize = 10.sp, color = Colores.textoSuave))
        }
    }
}

private fun vs(d: List<com.temrarios.ojoscan.historial.Diario>) =
    "mín ${d.minOf { it.valor }.roundToInt()} · máx ${d.maxOf { it.valor }.roundToInt()} lpm"
