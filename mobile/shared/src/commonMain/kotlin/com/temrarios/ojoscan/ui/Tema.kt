package com.temrarios.ojoscan.ui

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/** Paleta: espacio profundo + neón (cian = IA, violeta = acento, verde/ámbar/rojo = estado). */
object Colores {
    val fondo = Color(0xFF04060F)
    val fondo2 = Color(0xFF0A1028)
    val vidrio = Color(0x1A9FE8FF)
    val vidrioFuerte = Color(0x2E7FD8FF)
    val cian = Color(0xFF00E5FF)
    val violeta = Color(0xFF8B5CFF)
    val magenta = Color(0xFFFF3DCB)
    val texto = Color(0xFFE6F6FF)
    val textoSuave = Color(0xFF8FA6C2)
    val bien = Color(0xFF00FFA3)
    val atencion = Color(0xFFFFC23D)
    val mal = Color(0xFFFF4D6D)
    val velo = Color(0xCC02040B)

    // compatibilidad con nombres anteriores
    val turquesa = cian
    val turquesaClaro = cian

    val degradado = Brush.linearGradient(listOf(cian, violeta, magenta))
    val degradadoBorde = Brush.linearGradient(listOf(cian.copy(alpha = 0.8f), violeta.copy(alpha = 0.3f), magenta.copy(alpha = 0.7f)))
}

/** Tipografía tipo HUD: etiquetas en monoespaciada con espaciado amplio. */
object Hud {
    val etiqueta = TextStyle(fontFamily = FontFamily.Monospace, fontSize = 11.sp, letterSpacing = 2.5.sp,
        fontWeight = FontWeight.SemiBold, color = Colores.cian)
    val dato = TextStyle(fontFamily = FontFamily.Monospace, fontSize = 13.sp, letterSpacing = 1.sp,
        color = Colores.texto)
}

@Composable
fun TemaOjo(contenido: @Composable () -> Unit) {
    // Siempre oscuro: la interfaz es un HUD sobre fondo espacial.
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Colores.cian, onPrimary = Colores.fondo,
            secondary = Colores.violeta, tertiary = Colores.magenta,
            background = Colores.fondo, surface = Colores.fondo2, onSurface = Colores.texto,
            onBackground = Colores.texto, surfaceVariant = Color(0xFF16203D), onSurfaceVariant = Colores.textoSuave,
        ),
        content = contenido,
    )
}

/** Fondo animado: degradado radial que respira, rejilla en perspectiva y partículas que suben. */
@Composable
fun FondoEspacial(modifier: Modifier = Modifier, contenido: @Composable BoxScope.() -> Unit) {
    val t = rememberInfiniteTransition()
    val fase by t.animateFloat(0f, 1f, infiniteRepeatable(tween(14000, easing = LinearEasing)))
    val respira by t.animateFloat(0.85f, 1.15f, infiniteRepeatable(tween(5000, easing = FastOutSlowInEasing), RepeatMode.Reverse))
    Box(
        modifier.fillMaxSize().background(Colores.fondo).drawBehind {
            val w = size.width
            val h = size.height
            drawRect(Brush.radialGradient(listOf(Colores.violeta.copy(alpha = 0.28f), Color.Transparent),
                center = Offset(w * 0.85f, h * 0.08f), radius = w * 0.9f * respira))
            drawRect(Brush.radialGradient(listOf(Colores.cian.copy(alpha = 0.18f), Color.Transparent),
                center = Offset(w * 0.1f, h * 0.55f), radius = w * 0.8f * (2f - respira)))
            // rejilla en perspectiva (suelo) que avanza
            val horizonte = h * 0.72f
            val lineas = Colores.cian.copy(alpha = 0.10f)
            for (i in 0..14) {
                val p = ((i + fase * 2f) % 14f) / 14f
                val y = horizonte + (h - horizonte) * p * p
                drawLine(lineas.copy(alpha = 0.04f + 0.12f * p), Offset(0f, y), Offset(w, y), 1.dp.toPx())
            }
            for (i in -8..8) {
                val x0 = w / 2 + i * w * 0.04f
                val x1 = w / 2 + i * w * 0.22f
                drawLine(lineas, Offset(x0, horizonte), Offset(x1, h), 1.dp.toPx())
            }
            // partículas
            for (i in 0 until 40) {
                val semilla = (i * 7919) % 1000 / 1000f
                val x = w * ((i * 0.6180339f) % 1f)
                val y = h * (1f - ((semilla + fase * (0.3f + semilla)) % 1f))
                val a = 0.08f + 0.27f * ((i * 37) % 10) / 10f
                drawCircle((if (i % 3 == 0) Colores.violeta else Colores.cian).copy(alpha = a), 1.2.dp.toPx() + semilla * 1.5f, Offset(x, y))
            }
        },
        content = contenido,
    )
}

/** Panel de vidrio esmerilado con borde degradado de neón. */
@Composable
fun PanelVidrio(
    modifier: Modifier = Modifier,
    resaltado: Boolean = false,
    contenido: @Composable ColumnScope.() -> Unit,
) {
    val forma = RoundedCornerShape(22.dp)
    Column(
        modifier.clip(forma)
            .background(Brush.verticalGradient(listOf(if (resaltado) Colores.vidrioFuerte else Colores.vidrio, Color(0x0D8B5CFF))))
            .border(1.dp, if (resaltado) Colores.degradado else Colores.degradadoBorde, forma)
            .padding(18.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        content = contenido,
    )
}

/** Título de sección: índice + etiqueta HUD + línea que se desvanece. */
@Composable
fun TituloHud(indice: String, texto: String) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Text(indice, style = Hud.etiqueta.copy(color = Colores.magenta))
        Text(texto.uppercase(), style = Hud.etiqueta)
        Box(Modifier.weight(1f).height(1.dp).background(Brush.horizontalGradient(listOf(Colores.cian.copy(alpha = 0.6f), Color.Transparent))))
    }
}

/** Botón principal: píldora con degradado, halo que pulsa y escala al presionar. */
@Composable
fun BotonNeon(texto: String, alPulsar: () -> Unit, modifier: Modifier = Modifier, habilitado: Boolean = true) {
    val t = rememberInfiniteTransition()
    val halo by t.animateFloat(0.35f, 0.9f, infiniteRepeatable(tween(1600), RepeatMode.Reverse))
    val brillo by t.animateFloat(-1f, 2f, infiniteRepeatable(tween(2600, easing = LinearEasing)))
    val interaccion = remember { MutableInteractionSource() }
    val presionado by interaccion.collectIsPressedAsState()
    val escala by animateFloatAsState(if (presionado) 0.96f else 1f)
    val forma = RoundedCornerShape(50)
    Box(
        modifier.height(62.dp).scale(escala)
            .drawBehind {
                if (habilitado) {
                    for (k in 1..3) {
                        val m = k * 6.dp.toPx()
                        drawRoundRect(Colores.cian.copy(alpha = halo * 0.10f / k), Offset(-m, -m),
                            Size(size.width + 2 * m, size.height + 2 * m), androidx.compose.ui.geometry.CornerRadius(size.height))
                    }
                }
            }
            .clip(forma)
            .background(if (habilitado) Brush.horizontalGradient(listOf(Colores.cian, Colores.violeta, Colores.magenta))
                        else Brush.horizontalGradient(listOf(Color(0xFF2A3350), Color(0xFF2A3350))))
            .drawBehind {
                // destello que recorre el botón
                val x = size.width * brillo
                drawRect(Brush.linearGradient(listOf(Color.Transparent, Color.White.copy(alpha = 0.35f), Color.Transparent),
                    start = Offset(x - size.height, 0f), end = Offset(x + size.height, size.height)))
            }
            .clickable(interaccion, null, enabled = habilitado, onClick = alPulsar),
        contentAlignment = Alignment.Center,
    ) {
        Text(texto.uppercase(), color = Colores.fondo, fontWeight = FontWeight.Black, letterSpacing = 3.sp, fontSize = 16.sp)
    }
}

/** Botón secundario: contorno de neón. */
@Composable
fun BotonContorno(texto: String, alPulsar: () -> Unit, modifier: Modifier = Modifier) {
    val forma = RoundedCornerShape(50)
    Box(
        modifier.height(54.dp).clip(forma).background(Colores.vidrio).border(1.dp, Colores.degradadoBorde, forma)
            .clickable(onClick = alPulsar),
        contentAlignment = Alignment.Center,
    ) {
        Text(texto.uppercase(), color = Colores.cian, fontWeight = FontWeight.Bold, letterSpacing = 2.sp, fontSize = 14.sp)
    }
}

/** Opción seleccionable tipo "segmento" de un panel de control. */
@Composable
fun Segmento(texto: String, activo: Boolean, alPulsar: () -> Unit, modifier: Modifier = Modifier, detalle: String? = null) {
    val forma = RoundedCornerShape(14.dp)
    val alfa by animateFloatAsState(if (activo) 1f else 0f)
    Column(
        modifier.clip(forma)
            .background(Brush.verticalGradient(listOf(Colores.cian.copy(alpha = 0.04f + 0.16f * alfa), Colores.violeta.copy(alpha = 0.10f * alfa))))
            .border(1.dp, if (activo) Colores.degradado else Brush.linearGradient(listOf(Color(0x335C7BA8), Color(0x335C7BA8))), forma)
            .clickable(onClick = alPulsar)
            .padding(horizontal = 12.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Text(texto, color = if (activo) Colores.texto else Colores.textoSuave, fontWeight = if (activo) FontWeight.Bold else FontWeight.Medium,
            fontSize = 14.sp, textAlign = TextAlign.Center)
        detalle?.let { Text(it, color = Colores.textoSuave, fontSize = 10.sp, textAlign = TextAlign.Center, lineHeight = 13.sp) }
    }
}

/** Interruptor de neón. */
@Composable
fun InterruptorNeon(texto: String, activo: Boolean, alCambiar: (Boolean) -> Unit, detalle: String? = null) {
    val x by animateFloatAsState(if (activo) 1f else 0f)
    Row(Modifier.fillMaxWidth().clickable { alCambiar(!activo) }, verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(texto, color = Colores.texto, fontSize = 15.sp)
            detalle?.let { Text(it, color = Colores.textoSuave, fontSize = 11.sp, lineHeight = 14.sp) }
        }
        Box(
            Modifier.size(52.dp, 28.dp).clip(CircleShape)
                .background(Brush.horizontalGradient(listOf(Colores.cian.copy(alpha = 0.15f + 0.6f * x), Colores.violeta.copy(alpha = 0.15f + 0.6f * x))))
                .border(1.dp, Colores.cian.copy(alpha = 0.4f + 0.6f * x), CircleShape),
        ) {
            Box(
                Modifier.padding(3.dp).size(22.dp).align(Alignment.CenterStart)
                    .drawBehind {
                        val dx = (52.dp.toPx() - 28.dp.toPx()) * x
                        drawCircle(Colores.cian.copy(alpha = 0.35f * x), size.minDimension, Offset(size.width / 2 + dx, size.height / 2))
                        drawCircle(if (activo) Color.White else Colores.textoSuave, size.minDimension / 2, Offset(size.width / 2 + dx, size.height / 2))
                    },
            )
        }
    }
}

/** Indicador de estado: punto que late. */
@Composable
fun Led(color: Color, tam: Dp = 8.dp) {
    val t = rememberInfiniteTransition()
    val a by t.animateFloat(0.3f, 1f, infiniteRepeatable(tween(900), RepeatMode.Reverse))
    Canvas(Modifier.size(tam * 2)) {
        drawCircle(color.copy(alpha = 0.25f * a), size.minDimension / 2)
        drawCircle(color.copy(alpha = 0.5f + 0.5f * a), size.minDimension / 4)
    }
}

/** Medidor circular animado con marcas de graduación y valor al centro. */
@Composable
fun AnilloConfianza(valor: Float, color: Color, etiqueta: String, modifier: Modifier = Modifier.size(150.dp)) {
    var objetivo by remember { mutableStateOf(0f) }
    LaunchedEffect(valor) { objetivo = valor }
    val v by animateFloatAsState(objetivo, tween(1400, easing = FastOutSlowInEasing))
    val t = rememberInfiniteTransition()
    val giro by t.animateFloat(0f, 360f, infiniteRepeatable(tween(9000, easing = LinearEasing)))
    Box(modifier, contentAlignment = Alignment.Center) {
        Canvas(Modifier.fillMaxSize()) {
            val r = size.minDimension / 2
            val c = Offset(size.width / 2, size.height / 2)
            for (i in 0 until 60) {
                val ang = (i * 6f - 90f) * PI.toFloat() / 180f
                val largo = if (i % 5 == 0) 8.dp.toPx() else 4.dp.toPx()
                val enc = i / 60f <= v
                drawLine(if (enc) color.copy(alpha = 0.9f) else Colores.textoSuave.copy(alpha = 0.25f),
                    Offset(c.x + cos(ang) * (r - largo), c.y + sin(ang) * (r - largo)),
                    Offset(c.x + cos(ang) * r, c.y + sin(ang) * r), 2.dp.toPx())
            }
            val g = 14.dp.toPx()
            val tl = Offset(g, g)
            val sz = Size(size.width - 2 * g, size.height - 2 * g)
            drawArc(Colores.textoSuave.copy(alpha = 0.15f), 0f, 360f, false, tl, sz, style = Stroke(6.dp.toPx()))
            drawArc(color.copy(alpha = 0.25f), -90f, 360f * v, false, tl, sz, style = Stroke(14.dp.toPx(), cap = StrokeCap.Round))
            drawArc(Brush.sweepGradient(listOf(color.copy(alpha = 0.6f), color, Color.White)), -90f, 360f * v, false, tl, sz,
                style = Stroke(6.dp.toPx(), cap = StrokeCap.Round))
            // órbita decorativa
            val g2 = 26.dp.toPx()
            drawArc(Colores.violeta.copy(alpha = 0.5f), giro, 50f, false, Offset(g2, g2),
                Size(size.width - 2 * g2, size.height - 2 * g2), style = Stroke(1.5.dp.toPx(), cap = StrokeCap.Round))
            drawArc(Colores.cian.copy(alpha = 0.4f), giro + 180f, 30f, false, Offset(g2, g2),
                Size(size.width - 2 * g2, size.height - 2 * g2), style = Stroke(1.5.dp.toPx(), cap = StrokeCap.Round))
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("${(v * 100).roundToInt()}%", color = Colores.texto, fontSize = 34.sp, fontWeight = FontWeight.Black)
            Text(etiqueta.uppercase(), style = Hud.etiqueta.copy(fontSize = 9.sp, color = color))
        }
    }
}

/** Barra de probabilidad con brillo y animación de llenado; [marcas] dibuja líneas de referencia (0..1). */
@Composable
fun BarraNeon(nombre: String, p: Float, color: Color, marcas: List<Float> = emptyList(), detalle: String? = null) {
    var objetivo by remember { mutableStateOf(0f) }
    LaunchedEffect(p) { objetivo = p }
    val v by animateFloatAsState(objetivo, tween(1100, easing = FastOutSlowInEasing))
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(nombre, Modifier.weight(1f), color = Colores.texto, fontSize = 13.sp)
            Text("${(v * 100).roundToInt()}%", style = Hud.dato.copy(color = color, fontWeight = FontWeight.Bold))
        }
        Canvas(Modifier.fillMaxWidth().height(10.dp)) {
            val rad = androidx.compose.ui.geometry.CornerRadius(size.height / 2)
            drawRoundRect(Color(0x2216A0FF), cornerRadius = rad)
            // segmentos
            val n = 24
            for (i in 1 until n) {
                val x = size.width * i / n
                drawLine(Colores.fondo.copy(alpha = 0.6f), Offset(x, 0f), Offset(x, size.height), 1.dp.toPx())
            }
            val w = size.width * v.coerceIn(0f, 1f)
            drawRoundRect(color.copy(alpha = 0.25f), Offset(0f, -2.dp.toPx()), Size(w, size.height + 4.dp.toPx()), rad)
            drawRoundRect(Brush.horizontalGradient(listOf(color.copy(alpha = 0.5f), color), endX = w.coerceAtLeast(1f)),
                size = Size(w, size.height), cornerRadius = rad)
            if (w > 4f) drawCircle(Color.White, 2.5.dp.toPx(), Offset(w - 3.dp.toPx(), size.height / 2))
            for (m in marcas) {
                val x = size.width * m
                drawLine(Colores.texto.copy(alpha = 0.8f), Offset(x, -3.dp.toPx()), Offset(x, size.height + 3.dp.toPx()), 1.5.dp.toPx())
            }
        }
        detalle?.let { Text(it, color = Colores.textoSuave, fontSize = 11.sp) }
    }
}

/** Texto con degradado neón. */
@Composable
fun TextoDegradado(texto: String, tam: Int, modifier: Modifier = Modifier) {
    Text(texto, modifier, style = TextStyle(brush = Colores.degradado, fontSize = tam.sp, fontWeight = FontWeight.Black, letterSpacing = 1.sp))
}
