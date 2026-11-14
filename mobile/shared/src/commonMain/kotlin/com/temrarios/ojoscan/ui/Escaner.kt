package com.temrarios.ojoscan.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.height
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.graphics.drawscope.rotate
import com.temrarios.ojoscan.guia.Lectura
import com.temrarios.ojoscan.guia.Umbrales
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.temrarios.ojoscan.Ajustes
import com.temrarios.ojoscan.guia.Aviso
import com.temrarios.ojoscan.guia.Guia
import com.temrarios.ojoscan.guia.Modo
import com.temrarios.ojoscan.guia.MotorGuia
import com.temrarios.ojoscan.guia.Ojo
import com.temrarios.ojoscan.platform.CamaraOjo
import com.temrarios.ojoscan.platform.Captura
import com.temrarios.ojoscan.platform.Recorte
import com.temrarios.ojoscan.platform.rememberVoz
import kotlinx.coroutines.delay

@Composable
fun PantallaEscaner(ajustes: Ajustes, alVolver: () -> Unit, alTerminar: (List<Recorte>) -> Unit) {
    val motor = remember(ajustes.modo) { MotorGuia(ajustes.modo) }
    val voz = rememberVoz()
    // De cerca solo cabe un ojo por foto: "ambos" = primero el izquierdo, luego el derecho.
    val orden = remember(ajustes.ojo) {
        if (ajustes.ojo == Ojo.AMBOS) listOf(Ojo.IZQUIERDO, Ojo.DERECHO) else listOf(ajustes.ojo)
    }
    var indice by remember { mutableIntStateOf(0) }
    val hechos = remember { mutableListOf<Recorte>() }
    val ojoActual = orden[indice]
    var guia by remember { mutableStateOf(Guia(Aviso.SIN_OJO, 0f)) }
    var pedirFoto by remember { mutableIntStateOf(0) }
    var revisando by remember { mutableStateOf(false) }
    var repetir by remember { mutableStateOf<String?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var ultima by remember { mutableStateOf<Lectura?>(null) }

    DisposableEffect(Unit) { onDispose { voz.callar() } }
    // El motivo de repetir se muestra unos segundos y luego vuelve la guía normal.
    LaunchedEffect(repetir) {
        if (repetir != null) {
            delay(2500)
            repetir = null
            motor.reiniciar()
        }
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        CamaraOjo(
            modifier = Modifier.fillMaxSize(),
            ojo = ojoActual,
            modo = ajustes.modo,
            trasera = ajustes.trasera,
            pedirFoto = pedirFoto,
            alLeer = { lectura ->
                ultima = lectura
                if (!revisando && repetir == null) {
                    val g = motor.evaluar(lectura)
                    guia = g
                    if (ajustes.voz) g.hablar?.let(voz::decir)
                    if (g.capturar) {
                        revisando = true
                        pedirFoto++
                    }
                }
            },
            alCapturar = { c ->
                revisando = false
                when (c) {
                    is Captura.Exito -> {
                        hechos += c.recortes
                        if (indice + 1 < orden.size) {
                            indice++
                            val siguiente = "Bien. Ahora tu ojo ${orden[indice].nombre}"
                            repetir = siguiente
                            if (ajustes.voz) voz.decir(siguiente)
                        } else {
                            if (ajustes.voz) voz.decir("Listo")
                            alTerminar(hechos.toList())
                        }
                    }
                    is Captura.Repetir -> {
                        repetir = c.motivo
                        if (ajustes.voz) voz.decir(c.motivo)
                    }
                }
            },
            alError = { error = it },
        )

        VisorHud(guia, revisando, Modifier.fillMaxSize())

        Column(Modifier.fillMaxSize().safeDrawingPadding().padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            // ── Barra superior ──
            Row(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(Colores.velo)
                    .border(1.dp, Colores.degradadoBorde, RoundedCornerShape(18.dp)).padding(horizontal = 6.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = alVolver) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, "Volver", tint = Colores.cian)
                }
                Column(Modifier.weight(1f)) {
                    Text(if (ajustes.modo == Modo.ANEMIA) "PROTOCOLO · ANEMIA" else "PROTOCOLO · OJO", style = Hud.etiqueta.copy(fontSize = 9.sp))
                    Text(
                        (if (ojoActual == Ojo.IZQUIERDO) "Ojo izquierdo" else "Ojo derecho") +
                            (if (orden.size > 1) "  ${indice + 1}/2" else ""),
                        color = Colores.texto, fontWeight = FontWeight.Bold, fontSize = 16.sp,
                    )
                }
                Led(if (error != null) Colores.mal else Colores.bien)
                Text("EN VIVO", style = Hud.etiqueta.copy(fontSize = 9.sp, color = if (error != null) Colores.mal else Colores.bien))
                Spacer(Modifier.size(8.dp))
            }
            if (ajustes.modo == Modo.ANEMIA) {
                Text(
                    "Jala tu párpado de abajo con un dedo y mira hacia arriba: que se vea la parte rosada de adentro.",
                    color = Colores.texto, fontSize = 14.sp, textAlign = TextAlign.Center, lineHeight = 18.sp,
                    modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(Colores.velo)
                        .border(1.dp, Colores.magenta.copy(alpha = 0.6f), RoundedCornerShape(16.dp)).padding(12.dp),
                )
            }
            Spacer(Modifier.weight(1f))

            // ── Telemetría ──
            Telemetria(ultima, ajustes.modo)

            // ── Estado ──
            val texto = error ?: repetir ?: if (revisando) "Analizando la foto…" else guia.aviso.texto
            val color = when {
                error != null || repetir != null -> Colores.mal
                revisando || guia.aviso.bueno -> Colores.bien
                guia.aviso == Aviso.SIN_OJO -> Colores.mal
                else -> Colores.atencion
            }
            val colorAnim by animateColorAsState(color)
            Row(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(Colores.velo)
                    .border(1.5.dp, colorAnim.copy(alpha = 0.8f), RoundedCornerShape(20.dp)).padding(18.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (revisando) {
                    CircularProgressIndicator(Modifier.size(26.dp), color = Colores.bien, strokeWidth = 3.dp)
                } else {
                    Led(colorAnim, 10.dp)
                }
                Spacer(Modifier.size(12.dp))
                Column(Modifier.weight(1f)) {
                    Text("ESTADO DEL SISTEMA", style = Hud.etiqueta.copy(fontSize = 9.sp, color = colorAnim))
                    Text(texto, color = Colores.texto, fontSize = 21.sp, fontWeight = FontWeight.Bold, lineHeight = 25.sp)
                }
            }
            Text(
                "Captura automática cuando todo está en verde.",
                style = Hud.dato.copy(fontSize = 10.sp, color = Colores.textoSuave),
                textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

/** Lecturas en vivo de la cámara como medidores HUD. */
@Composable
private fun Telemetria(l: Lectura?, modo: Modo) {
    val u = remember { Umbrales() }
    val ok = l?.ojoVisible == true
    val luz = l?.brillo ?: 0f
    val nit = l?.nitidez ?: 0f
    val ap = l?.apertura ?: 0f
    val est = 1f - ((l?.giro ?: 0f) / (u.giroMax * 2f)).coerceIn(0f, 1f)
    val medidores = buildList {
        add(Triple("LUZ", (luz / 255f), ok && luz in u.brilloMin..u.brilloMax))
        add(Triple("FOCO", (nit / (u.nitidezMin * 4f)).coerceIn(0f, 1f), ok && nit >= u.nitidezMin))
        add(Triple("PULSO", est, (l?.giro ?: 0f) < u.giroMax))
        if (modo == Modo.OJO) add(Triple("APERTURA", (ap / 0.4f).coerceIn(0f, 1f), ok && ap >= u.aperturaMin))
        add(Triple("IA", l?.confianza ?: 0f, ok))
    }
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(Colores.velo)
            .border(1.dp, Colores.cian.copy(alpha = 0.25f), RoundedCornerShape(16.dp)).padding(horizontal = 12.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        for ((nombre, v, bien) in medidores) {
            val c = if (bien) Colores.bien else if (l == null) Colores.textoSuave else Colores.atencion
            val va by animateFloatAsState(v.coerceIn(0f, 1f))
            Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Text(nombre, style = Hud.etiqueta.copy(fontSize = 8.sp, letterSpacing = 1.sp, color = c))
                Canvas(Modifier.fillMaxWidth().height(5.dp)) {
                    drawRoundRect(Color(0x33FFFFFF), cornerRadius = androidx.compose.ui.geometry.CornerRadius(4f))
                    drawRoundRect(c, size = Size(size.width * va, size.height), cornerRadius = androidx.compose.ui.geometry.CornerRadius(4f))
                }
            }
        }
    }
}

/**
 * Visor: velo oscuro con ventana en forma de ojo (centrada, donde el detector espera el ojo),
 * corchetes de mira, anillo de graduación que gira, línea de barrido y progreso de la captura.
 */
@Composable
private fun VisorHud(guia: Guia, revisando: Boolean, modifier: Modifier) {
    val color by animateColorAsState(
        when {
            revisando || guia.aviso.bueno -> Colores.bien
            guia.aviso == Aviso.SIN_OJO -> Colores.cian
            else -> Colores.atencion
        }
    )
    val progreso by animateFloatAsState(guia.progreso)
    val t = rememberInfiniteTransition()
    val barrido by t.animateFloat(0f, 1f, infiniteRepeatable(tween(2200, easing = LinearEasing), RepeatMode.Reverse))
    val giro by t.animateFloat(0f, 360f, infiniteRepeatable(tween(12000, easing = LinearEasing)))
    val pulso by t.animateFloat(0f, 1f, infiniteRepeatable(tween(1400, easing = LinearEasing)))
    Canvas(modifier.graphicsLayer(compositingStrategy = CompositingStrategy.Offscreen)) {
        val ancho = size.width * 0.86f
        val alto = ancho * 0.68f
        val ovalo = Rect(Offset((size.width - ancho) / 2, size.height * 0.5f - alto / 2), Size(ancho, alto))
        val c = ovalo.center
        drawRect(Colores.velo)
        drawOval(Color.Black, ovalo.topLeft, ovalo.size, blendMode = BlendMode.Clear)
        // rejilla tenue dentro de la ventana
        val paso = 28.dp.toPx()
        clipPath(androidx.compose.ui.graphics.Path().apply { addOval(ovalo) }) {
            var x = ovalo.left
            while (x < ovalo.right) { drawLine(color.copy(alpha = 0.07f), Offset(x, ovalo.top), Offset(x, ovalo.bottom)); x += paso }
            var y = ovalo.top
            while (y < ovalo.bottom) { drawLine(color.copy(alpha = 0.07f), Offset(ovalo.left, y), Offset(ovalo.right, y)); y += paso }
        }
        // contorno con halo
        drawOval(color.copy(alpha = 0.25f), ovalo.topLeft, ovalo.size, style = Stroke(10.dp.toPx()))
        drawOval(color, ovalo.topLeft, ovalo.size, style = Stroke(2.dp.toPx()))
        // onda que se expande cuando todo está bien
        if (guia.aviso.bueno) {
            val e = 1f + 0.12f * pulso
            drawOval(color.copy(alpha = 0.5f * (1f - pulso)), Offset(c.x - ancho * e / 2, c.y - alto * e / 2), Size(ancho * e, alto * e),
                style = Stroke(2.dp.toPx()))
        }
        // barrido
        if (!guia.aviso.bueno && !revisando) {
            val yb = ovalo.top + ovalo.height * barrido
            drawLine(Brush.horizontalGradient(listOf(Color.Transparent, color, Color.Transparent), ovalo.left, ovalo.right),
                Offset(ovalo.left + 12.dp.toPx(), yb), Offset(ovalo.right - 12.dp.toPx(), yb), 2.dp.toPx())
        }
        // corchetes de mira
        val m = 16.dp.toPx()
        val l = 34.dp.toPx()
        val caja = Rect(ovalo.left - m, ovalo.top - m, ovalo.right + m, ovalo.bottom + m)
        val trazo = 3.dp.toPx()
        for ((p, dx, dy) in listOf(Triple(caja.topLeft, 1f, 1f), Triple(caja.topRight, -1f, 1f),
                                   Triple(caja.bottomLeft, 1f, -1f), Triple(caja.bottomRight, -1f, -1f))) {
            drawLine(color, p, Offset(p.x + l * dx, p.y), trazo, cap = StrokeCap.Round)
            drawLine(color, p, Offset(p.x, p.y + l * dy), trazo, cap = StrokeCap.Round)
        }
        // anillo de graduación que gira alrededor
        val rg = ancho / 2 + 30.dp.toPx()
        rotate(giro, c) {
            for (i in 0 until 48) {
                if (i % 12 in 5..7) continue
                val a = i * 7.5f * PI.toFloat() / 180f
                val largo = if (i % 4 == 0) 9.dp.toPx() else 4.dp.toPx()
                drawLine(color.copy(alpha = 0.45f), Offset(c.x + cos(a) * rg, c.y + sin(a) * rg * 0.72f),
                    Offset(c.x + cos(a) * (rg + largo), c.y + sin(a) * (rg + largo) * 0.72f), 1.5.dp.toPx())
            }
        }
        // cruz central
        val cc = 8.dp.toPx()
        drawLine(color.copy(alpha = 0.7f), Offset(c.x - cc, c.y), Offset(c.x + cc, c.y), 1.dp.toPx())
        drawLine(color.copy(alpha = 0.7f), Offset(c.x, c.y - cc), Offset(c.x, c.y + cc), 1.dp.toPx())
        // progreso de la captura
        if (progreso > 0f) {
            val g = 8.dp.toPx()
            drawArc(Colores.bien.copy(alpha = 0.3f), -90f, 360f * progreso, false, Offset(ovalo.left - g, ovalo.top - g),
                Size(ovalo.width + 2 * g, ovalo.height + 2 * g), style = Stroke(12.dp.toPx(), cap = StrokeCap.Round))
            drawArc(Colores.bien, -90f, 360f * progreso, false, Offset(ovalo.left - g, ovalo.top - g),
                Size(ovalo.width + 2 * g, ovalo.height + 2 * g), style = Stroke(4.dp.toPx(), cap = StrokeCap.Round))
        }
    }
}
