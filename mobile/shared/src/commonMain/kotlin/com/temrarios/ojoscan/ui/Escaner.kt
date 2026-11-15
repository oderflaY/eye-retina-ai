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
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
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
import com.temrarios.ojoscan.Paso
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
fun PantallaEscaner(
    ajustes: Ajustes,
    paso: Paso,
    /** "PASO 2/4" en el chequeo completo; null en una medición suelta. */
    numero: String?,
    alVolver: () -> Unit,
    alTerminar: (List<Recorte>) -> Unit,
    pulso: com.temrarios.ojoscan.pulso.MonitorPulso? = null,
) {
    val modo = paso.modo
    // 3 s seguidos en verde antes de la foto (antes 0.9 s: no daba tiempo de prepararse)
    val motor = remember(modo) { MotorGuia(modo, Umbrales(sostenerMs = SOSTENER_MS)) }
    val voz = rememberVoz()
    // De cerca solo cabe un ojo por foto: "ambos" = primero el izquierdo, luego el derecho.
    val orden = remember(paso.ojo, modo) {
        if (modo.fotoLibre) listOf(Ojo.AMBOS)
        else if (paso.ojo == Ojo.AMBOS) listOf(Ojo.IZQUIERDO, Ojo.DERECHO) else listOf(paso.ojo)
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
    val rostro = modo.fotoLibre  // foto libre con cuenta regresiva (rostro o palma)
    val palma = modo == Modo.PALMA
    val encia = modo == Modo.ENCIA
    var cuenta by remember { mutableIntStateOf(0) }  // cuenta regresiva del modo rostro
    // Ojo / anemia: la guía no arranca hasta que la persona dice "Estoy listo" (y otra vez para el segundo ojo),
    // y luego da 3 segundos para acomodarse.
    var listoUsuario by remember(indice) { mutableStateOf(false) }
    var preparando by remember(indice) { mutableIntStateOf(0) }
    LaunchedEffect(preparando) {
        if (preparando > 0) {
            if (ajustes.voz) voz.decir(if (preparando == 3) "Prepárate. 3" else "$preparando")
            delay(1000)
            if (preparando == 1) motor.reiniciar()
            preparando--
        }
    }

    LaunchedEffect(cuenta) {
        if (cuenta > 0) {
            if (ajustes.voz) voz.decir("$cuenta")
            delay(1000)
            if (cuenta == 1) {
                cuenta = 0
                revisando = true
                pedirFoto++
            } else cuenta--
        }
    }

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
            modo = modo,
            trasera = ajustes.trasera,
            pedirFoto = pedirFoto,
            alLeer = { lectura ->
                ultima = lectura
                if (!rostro && !revisando && repetir == null && listoUsuario && preparando == 0) {
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

        if (palma) MarcoPalma(revisando, cuenta, Modifier.fillMaxSize())
        else if (encia) MarcoEncia(revisando, cuenta, Modifier.fillMaxSize())
        else if (rostro) MarcoRostro(revisando, cuenta, Modifier.fillMaxSize())
        else VisorHud(guia, revisando, ultima, modo, espejo = !ajustes.trasera, Modifier.fillMaxSize())

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
                    Text((numero?.let { "$it · " } ?: "PROTOCOLO · ") + modo.titulo.uppercase(), style = Hud.etiqueta.copy(fontSize = 9.sp))
                    Text(
                        if (palma) "Palma de la mano" else if (encia) "Encía" else if (rostro) "Cara completa" else (if (ojoActual == Ojo.IZQUIERDO) "Ojo izquierdo" else "Ojo derecho") +
                            (if (orden.size > 1) "  ${indice + 1}/2" else ""),
                        color = Colores.texto, fontWeight = FontWeight.Bold, fontSize = 16.sp,
                    )
                }
                pulso?.let { PulsoEnVivo(it); Spacer(Modifier.size(8.dp)) }
                Led(if (error != null) Colores.mal else Colores.bien)
                Text("EN VIVO", style = Hud.etiqueta.copy(fontSize = 9.sp, color = if (error != null) Colores.mal else Colores.bien))
                Spacer(Modifier.size(8.dp))
            }
            if (encia) {
                Text(
                    "Junta los dientes y levanta el labio de arriba (con un dedo si hace falta): que se vea la encía rosada.",
                    color = Colores.texto, fontSize = 14.sp, textAlign = TextAlign.Center, lineHeight = 18.sp,
                    modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(Colores.velo)
                        .border(1.dp, Colores.magenta.copy(alpha = 0.6f), RoundedCornerShape(16.dp)).padding(12.dp),
                )
            }
            if (modo == Modo.ANEMIA) {
                Text(
                    "Jala tu párpado de abajo con un dedo y mira hacia arriba: que se vea la parte rosada de adentro.",
                    color = Colores.texto, fontSize = 14.sp, textAlign = TextAlign.Center, lineHeight = 18.sp,
                    modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(Colores.velo)
                        .border(1.dp, Colores.magenta.copy(alpha = 0.6f), RoundedCornerShape(16.dp)).padding(12.dp),
                )
            }
            Spacer(Modifier.weight(1f))

            // ── Telemetría ──
            Telemetria(ultima, modo)

            // ── Estado ──
            val texto = error ?: repetir ?: when {
                revisando -> "Analizando la foto…"
                !rostro && !listoUsuario -> "Lee los pasos y toca «Estoy listo»"
                !rostro && preparando > 0 -> "Prepárate… $preparando"
                !rostro && guia.aviso == Aviso.LISTO ->
                    "¡Perfecto! Quieto… ${kotlin.math.ceil((1f - guia.progreso) * SOSTENER_MS / 1000f).toInt().coerceAtLeast(1)}"
                rostro && cuenta > 0 -> "Quieto… $cuenta"
                palma -> "Pon la palma abierta dentro del recuadro, con luz de día"
                encia -> "Dientes juntos y sonríe enseñando la encía de arriba"
                rostro -> "Pon tu cara dentro del óvalo, de frente y sin lentes oscuros"
                guia.aviso == Aviso.CENTRA -> "Lleva tu ojo a la cruz: sigue la flecha"
                guia.aviso == Aviso.ACERCATE -> "Acércate un poco (mira la barra)"
                guia.aviso == Aviso.ALEJATE -> "Aléjate un poco (mira la barra)"
                !ajustes.trasera && guia.aviso == Aviso.ABRE_OJO -> "Abre bien el ojo y mira el punto"
                else -> guia.aviso.texto
            }
            val color = when {
                error != null || repetir != null -> Colores.mal
                revisando || (!rostro && guia.aviso.bueno) || cuenta > 0 -> Colores.bien
                rostro -> Colores.cian
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
            if (rostro) {
                BotonNeon(if (cuenta > 0) "Capturando…" else "Tomar foto", { if (cuenta == 0 && !revisando) cuenta = 3 }, Modifier.fillMaxWidth())
            } else if (!listoUsuario) {
                Column(
                    Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(Colores.velo)
                        .border(1.dp, Colores.cian.copy(alpha = 0.5f), RoundedCornerShape(18.dp)).padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Text(if (indice > 0) "AHORA TU OJO ${ojoActual.nombre.uppercase()}" else "ANTES DE EMPEZAR",
                        style = Hud.etiqueta.copy(fontSize = 10.sp, color = Colores.cian))
                    val pasos = if (modo == Modo.ANEMIA) listOf(
                        "Busca luz blanca de día (ventana), sin flash ni foco amarillo.",
                        "Lávate las manos y jala tu párpado de abajo con un dedo.",
                        "Acerca el teléfono a unos 10 cm del ojo ${ojoActual.nombre}.",
                        "Mira hacia arriba, al punto, sin parpadear.",
                    ) else listOf(
                        "Quítate los lentes y busca luz de frente (ventana o foco), sin flash.",
                        "Acerca el teléfono a unos 10 cm del ojo ${ojoActual.nombre}.",
                        "Abre bien el ojo y mira el punto de arriba.",
                        "Cuando todo esté en verde, quédate quieto 3 segundos.",
                    )
                    pasos.forEachIndexed { i, t -> Text("${i + 1}. $t", color = Colores.texto, fontSize = 14.sp, lineHeight = 18.sp) }
                    BotonNeon("Estoy listo", { listoUsuario = true; preparando = 3 }, Modifier.fillMaxWidth())
                }
            } else Text(
                "Captura automática: cuando todo esté en verde, quédate quieto 3 segundos.",
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
    val medidores = if (modo.fotoLibre) listOf(
        Triple("LUZ", (luz / 255f), luz in u.brilloMin..u.brilloMax),
        Triple("PULSO", est, (l?.giro ?: 0f) < u.giroMax),
    ) else buildList {
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
private fun VisorHud(guia: Guia, revisando: Boolean, lectura: Lectura?, modo: Modo, espejo: Boolean, modifier: Modifier) {
    val u = remember { Umbrales() }
    val medidor = rememberTextMeasurer()
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
        // ── Anemia: dónde debe quedar el párpado rosado (abajo del ojo) ──
        if (modo == Modo.ANEMIA) {
            val zona = Rect(c.x - ancho * 0.30f, c.y + alto * 0.12f, c.x + ancho * 0.30f, c.y + alto * 0.42f)
            drawArc(Colores.magenta.copy(alpha = 0.18f), 0f, 180f, true, Offset(zona.left, zona.top - zona.height), Size(zona.width, zona.height * 2))
            drawArc(Colores.magenta.copy(alpha = 0.8f), 0f, 180f, false, Offset(zona.left, zona.top - zona.height), Size(zona.width, zona.height * 2),
                style = Stroke(2.dp.toPx(), pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(12f, 9f))))
            val t = medidor.measure("párpado rosado aquí", TextStyle(color = Colores.magenta, fontSize = 11.sp, fontWeight = FontWeight.Bold))
            drawText(t, topLeft = Offset(c.x - t.size.width / 2, zona.bottom + 4.dp.toPx()))
        }

        // ── Dónde ve la cámara el ojo ahora (mismo recorte que la vista previa: cuadro 3:4 llenando la pantalla) ──
        val l0 = lectura
        if (l0 != null && l0.ojoVisible && !revisando) {
            val esc = maxOf(size.width / 3f, size.height / 4f)
            val fw = 3f * esc; val fh = 4f * esc
            val ex = (size.width - fw) / 2 + (if (espejo) 1f - l0.ojoX else l0.ojoX) * fw
            val ey = (size.height - fh) / 2 + l0.ojoY * fh
            val ojo = Offset(ex, ey)
            val anchoVisto = (l0.anchoOjo * fw).coerceIn(24.dp.toPx(), size.width)
            val centrado = kotlin.math.abs(l0.ojoX - 0.5f) <= u.desvioCentro && kotlin.math.abs(l0.ojoY - 0.5f) <= u.desvioCentro
            val cm = if (centrado) Colores.bien else Colores.atencion
            drawOval(cm.copy(alpha = 0.9f), Offset(ex - anchoVisto / 2, ey - anchoVisto * 0.25f), Size(anchoVisto, anchoVisto * 0.5f),
                style = Stroke(2.dp.toPx(), pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(14f, 10f))))
            drawCircle(cm, 4.dp.toPx(), ojo)
            // flecha del ojo hacia la cruz cuando falta centrar
            val d = c - ojo
            val dist = d.getDistance()
            if (!centrado && dist > 30.dp.toPx()) {
                val dir = d / dist
                val ini = ojo + dir * 14.dp.toPx()
                val fin = c - dir * 14.dp.toPx()
                drawLine(cm, ini, fin, 4.dp.toPx(), cap = StrokeCap.Round)
                val pun = 16.dp.toPx()
                val nrm = Offset(-dir.y, dir.x)
                val ruta = androidx.compose.ui.graphics.Path().apply {
                    moveTo(fin.x + dir.x * 6.dp.toPx(), fin.y + dir.y * 6.dp.toPx())
                    lineTo(fin.x - dir.x * pun + nrm.x * pun * 0.6f, fin.y - dir.y * pun + nrm.y * pun * 0.6f)
                    lineTo(fin.x - dir.x * pun - nrm.x * pun * 0.6f, fin.y - dir.y * pun - nrm.y * pun * 0.6f)
                    close()
                }
                drawPath(ruta, cm)
            }
            // barra de distancia a la derecha del óvalo: franja verde = distancia buena
            val bx = size.width - 14.dp.toPx()
            val bTop = ovalo.top; val bAlto = ovalo.height
            val maxA = u.anchoMax * 1.6f
            fun yDe(a: Float) = bTop + bAlto * (1f - (a / maxA).coerceIn(0f, 1f))
            drawLine(Color(0x55FFFFFF), Offset(bx, bTop), Offset(bx, bTop + bAlto), 6.dp.toPx(), cap = StrokeCap.Round)
            drawLine(Colores.bien.copy(alpha = 0.7f), Offset(bx, yDe(u.anchoMax)), Offset(bx, yDe(u.anchoMin)), 6.dp.toPx(), cap = StrokeCap.Round)
            val enRango = l0.anchoOjo in u.anchoMin..u.anchoMax
            val ym = yDe(l0.anchoOjo)
            drawCircle(if (enRango) Colores.bien else Colores.atencion, 7.dp.toPx(), Offset(bx, ym))
            drawCircle(Color.White, 7.dp.toPx(), Offset(bx, ym), style = Stroke(1.5.dp.toPx()))
            val tc = medidor.measure("CERCA", TextStyle(color = Colores.textoSuave, fontSize = 8.sp))
            drawText(tc, topLeft = Offset(bx - tc.size.width / 2, bTop - tc.size.height - 4.dp.toPx()))
            val tl = medidor.measure("LEJOS", TextStyle(color = Colores.textoSuave, fontSize = 8.sp))
            drawText(tl, topLeft = Offset(bx - tl.size.width / 2, bTop + bAlto + 4.dp.toPx()))
        }

        // ── Punto para fijar la mirada (con la cámara de enfrente: arriba del óvalo, rumbo a la cámara) ──
        if (espejo) {
            val pm = Offset(c.x, ovalo.top - 64.dp.toPx())
            val listo = guia.aviso.bueno || revisando
            val cp = if (listo) Colores.bien else Color.White
            val r0 = 7.dp.toPx()
            drawCircle(cp.copy(alpha = 0.35f * (1f - pulso)), r0 * (1.6f + 1.8f * pulso), pm)
            drawCircle(cp.copy(alpha = 0.6f), r0 * 1.9f, pm, style = Stroke(1.5.dp.toPx()))
            drawCircle(if (listo) Colores.bien else Colores.magenta, r0, pm)
            drawCircle(Color.White, r0 * 0.35f, pm)
            val txt = if (modo == Modo.ANEMIA) "MIRA EL PUNTO (hacia arriba)" else "MIRA EL PUNTO"
            val tm = medidor.measure(txt, TextStyle(color = cp, fontSize = 11.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp))
            drawText(tm, topLeft = Offset(pm.x - tm.size.width / 2, pm.y + r0 * 2.4f))
        }

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

/** Óvalo guía para la cara completa, con barrido y esquinas de mira. */
@Composable
private fun MarcoRostro(revisando: Boolean, cuenta: Int, modifier: Modifier) {
    val t = rememberInfiniteTransition()
    val barrido by t.animateFloat(0f, 1f, infiniteRepeatable(tween(2200, easing = LinearEasing), RepeatMode.Reverse))
    val color by animateColorAsState(if (revisando || cuenta > 0) Colores.bien else Colores.violeta)
    Canvas(modifier) {
        val w = size.width * 0.72f
        val h = w * 1.32f
        val tl = Offset((size.width - w) / 2, size.height * 0.44f - h / 2)
        val ovalo = androidx.compose.ui.geometry.Rect(tl, androidx.compose.ui.geometry.Size(w, h))
        val ruta = androidx.compose.ui.graphics.Path().apply { addOval(ovalo) }
        clipPath(ruta, clipOp = androidx.compose.ui.graphics.ClipOp.Difference) { drawRect(Colores.fondo.copy(alpha = 0.55f)) }
        drawOval(color, tl, ovalo.size, style = Stroke(3.dp.toPx()))
        drawOval(color.copy(alpha = 0.25f), Offset(tl.x - 8.dp.toPx(), tl.y - 8.dp.toPx()),
            androidx.compose.ui.geometry.Size(w + 16.dp.toPx(), h + 16.dp.toPx()), style = Stroke(1.dp.toPx()))
        clipPath(ruta) {
            val y = tl.y + h * barrido
            drawLine(color.copy(alpha = 0.8f), Offset(tl.x, y), Offset(tl.x + w, y), 2.dp.toPx())
            // línea media: ayuda a alinear la cara (la asimetría se mide respecto a ella)
            drawLine(color.copy(alpha = 0.25f), Offset(size.width / 2, tl.y), Offset(size.width / 2, tl.y + h), 1.dp.toPx())
        }
    }
}

/** Recuadro guía para la palma abierta: ventana con esquinas de mira, barrido y la zona que se mide. */
@Composable
private fun MarcoPalma(revisando: Boolean, cuenta: Int, modifier: Modifier) {
    val t = rememberInfiniteTransition()
    val barrido by t.animateFloat(0f, 1f, infiniteRepeatable(tween(2000, easing = LinearEasing), RepeatMode.Reverse))
    val color by animateColorAsState(if (revisando || cuenta > 0) Colores.bien else Colores.atencion)
    Canvas(modifier) {
        val w = size.width * 0.78f
        val h = w * 1.25f
        val tl = Offset((size.width - w) / 2, size.height * 0.45f - h / 2)
        val caja = androidx.compose.ui.geometry.Rect(tl, androidx.compose.ui.geometry.Size(w, h))
        val r = 28.dp.toPx()
        val ruta = androidx.compose.ui.graphics.Path().apply {
            addRoundRect(androidx.compose.ui.geometry.RoundRect(caja, androidx.compose.ui.geometry.CornerRadius(r)))
        }
        clipPath(ruta, clipOp = androidx.compose.ui.graphics.ClipOp.Difference) { drawRect(Colores.fondo.copy(alpha = 0.55f)) }
        drawPath(ruta, color, style = Stroke(3.dp.toPx()))
        // zona que se mide: el centro de la palma (abajo de los dedos)
        val zc = Offset(caja.center.x, caja.top + h * 0.62f)
        val zr = w * 0.24f
        drawCircle(color.copy(alpha = 0.18f), zr, zc)
        drawCircle(color.copy(alpha = 0.7f), zr, zc, style = Stroke(1.5.dp.toPx(),
            pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(10f, 8f))))
        clipPath(ruta) {
            val y = tl.y + h * barrido
            drawLine(color.copy(alpha = 0.8f), Offset(tl.x, y), Offset(tl.x + w, y), 2.dp.toPx())
        }
        val l = 34.dp.toPx()
        val m = 12.dp.toPx()
        val ext = androidx.compose.ui.geometry.Rect(caja.left - m, caja.top - m, caja.right + m, caja.bottom + m)
        for ((p, dx, dy) in listOf(Triple(ext.topLeft, 1f, 1f), Triple(ext.topRight, -1f, 1f),
                                   Triple(ext.bottomLeft, 1f, -1f), Triple(ext.bottomRight, -1f, -1f))) {
            drawLine(color, p, Offset(p.x + l * dx, p.y), 3.dp.toPx(), cap = StrokeCap.Round)
            drawLine(color, p, Offset(p.x, p.y + l * dy), 3.dp.toPx(), cap = StrokeCap.Round)
        }
    }
}


/** Marco para la encía: ventana ancha en forma de boca con la línea de los dientes y la zona que se mide. */
@Composable
private fun MarcoEncia(revisando: Boolean, cuenta: Int, modifier: Modifier) {
    val t = rememberInfiniteTransition()
    val barrido by t.animateFloat(0f, 1f, infiniteRepeatable(tween(2000, easing = LinearEasing), RepeatMode.Reverse))
    val color by animateColorAsState(if (revisando || cuenta > 0) Colores.bien else Colores.magenta)
    val medidor = rememberTextMeasurer()
    Canvas(modifier) {
        val w = size.width * 0.84f
        val h = w * 0.55f
        val tl = Offset((size.width - w) / 2, size.height * 0.45f - h / 2)
        val caja = Rect(tl, Size(w, h))
        val ruta = androidx.compose.ui.graphics.Path().apply {
            addRoundRect(androidx.compose.ui.geometry.RoundRect(caja, androidx.compose.ui.geometry.CornerRadius(h / 2)))
        }
        clipPath(ruta, clipOp = androidx.compose.ui.graphics.ClipOp.Difference) { drawRect(Colores.fondo.copy(alpha = 0.55f)) }
        drawPath(ruta, color, style = Stroke(3.dp.toPx()))
        // línea donde se juntan los dientes (a media altura) y la encía de arriba encima
        val yd = caja.top + h * 0.62f
        drawLine(color.copy(alpha = 0.7f), Offset(caja.left + h * 0.3f, yd), Offset(caja.right - h * 0.3f, yd), 2.dp.toPx(),
            pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(12f, 9f)))
        val zona = Rect(caja.left + w * 0.22f, caja.top + h * 0.12f, caja.right - w * 0.22f, caja.top + h * 0.36f)
        drawRoundRect(color.copy(alpha = 0.18f), zona.topLeft, zona.size, androidx.compose.ui.geometry.CornerRadius(12.dp.toPx()))
        val et = medidor.measure("encía de arriba aquí", TextStyle(color = color, fontSize = 11.sp, fontWeight = FontWeight.Bold))
        drawText(et, topLeft = Offset(caja.center.x - et.size.width / 2, zona.center.y - et.size.height / 2))
        val dt = medidor.measure("dientes juntos", TextStyle(color = color.copy(alpha = 0.8f), fontSize = 10.sp))
        drawText(dt, topLeft = Offset(caja.center.x - dt.size.width / 2, yd + 4.dp.toPx()))
        clipPath(ruta) {
            val y = tl.y + h * barrido
            drawLine(color.copy(alpha = 0.6f), Offset(tl.x, y), Offset(tl.x + w, y), 2.dp.toPx())
        }
    }
}


/** Tiempo que el ojo debe quedarse bien (todo en verde) antes de la foto automática. */
private const val SOSTENER_MS = 3000L
