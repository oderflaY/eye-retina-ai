package com.temrarios.ojoscan.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.BorderStroke
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bloodtype
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.PanTool
import androidx.compose.material.icons.filled.RemoveRedEye
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.temrarios.ojoscan.Ajustes
import com.temrarios.ojoscan.api.OjoApi
import com.temrarios.ojoscan.api.explicar
import com.temrarios.ojoscan.guia.Modo
import com.temrarios.ojoscan.guia.Ojo
import kotlinx.coroutines.launch
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

private enum class Enlace { SIN_PROBAR, PROBANDO, OK, PC_APAGADA, FALLO }

@Composable
fun PantallaInicio(alVolver: () -> Unit, ajustes: Ajustes, alCambiar: (Ajustes) -> Unit, alEscanear: () -> Unit) {
    val alcance = rememberCoroutineScope()
    var enlace by remember { mutableStateOf(Enlace.SIN_PROBAR) }
    var detalleEnlace by remember { mutableStateOf<String?>(null) }
    var sistemaAbierto by remember { mutableStateOf(false) }

    fun probar() {
        enlace = Enlace.PROBANDO
        detalleEnlace = null
        alcance.launch {
            val api = OjoApi(ajustes.servidor)
            runCatching { api.salud() }.fold(
                onSuccess = { s ->
                    fun estado(k: String) = if (s.models[k]?.available == true) "EN LÍNEA" else "NO DISPONIBLE"
                    if (s.pcConectada == false) {
                        enlace = Enlace.PC_APAGADA
                        detalleEnlace = "Servidor en línea, pero la computadora con la IA está apagada. " +
                            "Las fotos guardadas le llegarán cuando se conecte."
                    } else {
                        enlace = Enlace.OK
                        detalleEnlace = "NÚCLEO v${s.version} · OJO ${estado("eye_analysis")} · ANEMIA ${estado("anemia")}"
                    }
                },
                onFailure = { enlace = Enlace.FALLO; detalleEnlace = explicar(it) },
            )
            api.cerrar()
        }
    }

    val sinObjetivo = ajustes.completo || ajustes.modo.fotoLibre
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
                    Text("ESCANEOS COMPLEMENTARIOS", style = Hud.etiqueta.copy(color = Colores.textoSuave, fontSize = 10.sp))
                    TextoDegradado("OjoScan", 40)
                }
                EstadoEnlace(enlace) { probar() }
            }

            // ── Iris holográfico ───────────────────────────────────────
            Box(Modifier.fillMaxWidth().height(250.dp), contentAlignment = Alignment.Center) {
                IrisHolografico(ajustes.modo, Modifier.size(240.dp))
            }
            Text(
                if (ajustes.completo) "Todas las mediciones seguidas y un perfil médico que las combina."
                else when (ajustes.modo) {
                    Modo.OJO -> "Visión computacional que analiza tu ojo en segundos."
                    Modo.ANEMIA -> "Tamizaje de anemia por el color del párpado interno."
                    Modo.ROSTRO -> "Tu cara completa: asimetría facial, ojos amarillos y tus dos ojos a la vez."
                    Modo.PALMA -> "Tamizaje de anemia por el color de la palma de tu mano."
                    Modo.ENCIA -> "Tamizaje experimental de anemia por el color de tu encía."
                },
                color = Colores.textoSuave, fontSize = 14.sp, modifier = Modifier.fillMaxWidth(),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
            )

            // ── Protocolo ──────────────────────────────────────────────
            PanelVidrio(Modifier.fillMaxWidth()) {
                TituloHud("01", "Protocolo de escaneo")
                TarjetaCompleto(ajustes.completo) { alCambiar(ajustes.copy(completo = true)) }
                Text("O UNA MEDICIÓN SUELTA", style = Hud.etiqueta.copy(fontSize = 9.sp, color = Colores.textoSuave))
                for (fila in Modo.entries.chunked(2)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        for (m in fila) {
                            TarjetaModo(m, m.icono(), !ajustes.completo && ajustes.modo == m, Modifier.weight(1f)) {
                                alCambiar(ajustes.copy(completo = false, modo = m))
                            }
                        }
                    }
                }
                Text(
                    if (ajustes.completo) "5 pasos, unos 4 minutos: tus datos, tus dos ojos de cerca, el párpado, tu cara y tu encía. " +
                        "Cada medición confirma o descarta a las otras. Puedes saltar un paso si no puedes hacerlo."
                    else when (ajustes.modo) {
                        Modo.OJO -> "Mira de frente con el ojo bien abierto. La IA guía tu cámara."
                        Modo.ANEMIA -> "Jala tu párpado de abajo y mira hacia arriba. Mejor con luz blanca de día."
                        Modo.ROSTRO -> "Cara de frente y relajada, a un brazo de distancia, con luz de día. Sin lentes ni fleco sobre los ojos."
                        Modo.PALMA -> "Mano abierta con la palma hacia la cámara, dedos un poco separados, con luz de día y sin flash."
                        Modo.ENCIA -> "Dientes juntos y sonrisa grande levantando el labio de arriba: que se vea la encía. Luz de día."
                    },
                    color = Colores.textoSuave, fontSize = 12.sp, lineHeight = 16.sp,
                )
            }

            // ── Objetivo ───────────────────────────────────────────────
            if (!sinObjetivo) PanelVidrio(Modifier.fillMaxWidth()) {
                TituloHud("02", "Objetivo")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (o in Ojo.entries) {
                        Segmento(o.nombre.replaceFirstChar { it.uppercase() }, ajustes.ojo == o,
                            { alCambiar(ajustes.copy(ojo = o)) }, Modifier.weight(1f))
                    }
                }
                Text("Es TU ojo, no el lado de la pantalla.", color = Colores.textoSuave, fontSize = 11.sp)
            }

            // ── Sensor ─────────────────────────────────────────────────
            PanelVidrio(Modifier.fillMaxWidth()) {
                TituloHud(if (sinObjetivo) "02" else "03", "Sensor óptico")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Segmento("Frontal", !ajustes.trasera, { alCambiar(ajustes.copy(trasera = false)) }, Modifier.weight(1f),
                        detalle = "Te ves en pantalla")
                    Segmento("Trasera", ajustes.trasera, { alCambiar(ajustes.copy(trasera = true)) }, Modifier.weight(1f),
                        detalle = "Máxima resolución")
                }
                InterruptorNeon("Asistente de voz", ajustes.voz, { alCambiar(ajustes.copy(voz = it)) },
                    detalle = if (ajustes.trasera) "Con la trasera no ves la pantalla: deja la voz encendida." else "Te dice cómo acomodar la cámara.")
            }

            // ── Sistema (plegable) ─────────────────────────────────────
            PanelVidrio(Modifier.fillMaxWidth()) {
                Row(Modifier.fillMaxWidth().clickable { sistemaAbierto = !sistemaAbierto }, verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.weight(1f)) { TituloHud(if (sinObjetivo) "03" else "04", "Sistema") }
                    Text(if (sistemaAbierto) "  ▲" else "  ▼", color = Colores.cian, fontSize = 12.sp)
                }
                AnimatedVisibility(sistemaAbierto, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        OutlinedTextField(
                            value = ajustes.servidor,
                            onValueChange = { alCambiar(ajustes.copy(servidor = it.trim())) },
                            label = { Text("Nodo de la IA") },
                            singleLine = true,
                            textStyle = Hud.dato,
                            shape = RoundedCornerShape(14.dp),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Colores.cian, unfocusedBorderColor = Colores.cian.copy(alpha = 0.3f),
                                focusedLabelColor = Colores.cian, cursorColor = Colores.cian,
                            ),
                            modifier = Modifier.fillMaxWidth(),
                        )
                        BotonContorno(if (enlace == Enlace.PROBANDO) "Sincronizando…" else "Probar enlace", { probar() }, Modifier.fillMaxWidth())
                        InterruptorNeon("Donar mis escaneos al entrenamiento", ajustes.guardar, { alCambiar(ajustes.copy(guardar = it)) },
                            detalle = if (ajustes.guardar) "Aceptas que el recorte de tu ojo se guarde en el servidor del proyecto para " +
                                "entrenar el modelo. Puedes desactivarlo cuando quieras." else "Desactivado: tus fotos solo se analizan.")
                    }
                }
                detalleEnlace?.let {
                    Text(it, style = Hud.dato.copy(fontSize = 11.sp, color = when (enlace) {
                        Enlace.OK -> Colores.bien
                        Enlace.PC_APAGADA -> Colores.atencion
                        else -> Colores.mal
                    }))
                }
            }

            BotonNeon(if (ajustes.completo) "Iniciar chequeo completo" else "Iniciar escaneo", alEscanear, Modifier.fillMaxWidth())
            Text(
                "Herramienta de investigación. No es un dispositivo médico ni sustituye la consulta con un especialista.",
                color = Colores.textoSuave.copy(alpha = 0.7f), fontSize = 10.sp, lineHeight = 13.sp,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center, modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(8.dp))
        }
    }
}

@Composable
private fun EstadoEnlace(enlace: Enlace, alPulsar: () -> Unit) {
    val (color, texto) = when (enlace) {
        Enlace.SIN_PROBAR -> Colores.textoSuave to "ENLACE"
        Enlace.PROBANDO -> Colores.cian to "SYNC…"
        Enlace.OK -> Colores.bien to "EN LÍNEA"
        Enlace.PC_APAGADA -> Colores.atencion to "IA OFF"
        Enlace.FALLO -> Colores.mal to "SIN RED"
    }
    Row(
        Modifier.clickable(onClick = alPulsar).padding(4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Led(color)
        Text(texto, style = Hud.etiqueta.copy(color = color, fontSize = 10.sp))
    }
}

/** Opción principal: las 4 mediciones seguidas. */
@Composable
private fun TarjetaCompleto(activo: Boolean, alPulsar: () -> Unit) {
    val forma = RoundedCornerShape(18.dp)
    Row(
        Modifier.fillMaxWidth().bordeModo(activo, forma).clickable(onClick = alPulsar).padding(14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text("RECOMENDADO", style = Hud.etiqueta.copy(fontSize = 8.sp, color = if (activo) Colores.bien else Colores.textoSuave))
            Text("Chequeo completo", color = if (activo) Colores.texto else Colores.textoSuave, fontWeight = FontWeight.Bold, fontSize = 18.sp)
            Text("Tus datos + ojos + anemia + rostro + encía + palma → tu perfil médico", color = Colores.textoSuave, fontSize = 11.sp, lineHeight = 14.sp)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            for (m in Modo.entries) Icon(m.icono(), null, tint = if (activo) Colores.cian else Colores.textoSuave, modifier = Modifier.size(18.dp))
        }
    }
}

@Composable
private fun TarjetaModo(
    modo: Modo,
    icono: androidx.compose.ui.graphics.vector.ImageVector,
    activo: Boolean,
    modifier: Modifier,
    alPulsar: () -> Unit,
) {
    val forma = RoundedCornerShape(18.dp)
    Column(
        modifier.height(132.dp)
            .bordeModo(activo, forma)
            .clickable(onClick = alPulsar)
            .padding(horizontal = 11.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.SpaceBetween,
    ) {
        Icon(icono, null, tint = if (activo) Colores.cian else Colores.textoSuave, modifier = Modifier.size(28.dp))
        Column {
            Text(modo.titulo, color = if (activo) Colores.texto else Colores.textoSuave, fontWeight = FontWeight.Bold, fontSize = 16.sp)
            Text(modo.detalle, color = Colores.textoSuave, fontSize = 10.sp, lineHeight = 12.sp, maxLines = 3)
        }
    }
}

private fun Modifier.bordeModo(activo: Boolean, forma: RoundedCornerShape): Modifier =
    clip(forma)
        .background(
            Brush.verticalGradient(
                if (activo) listOf(Colores.cian.copy(alpha = 0.22f), Colores.violeta.copy(alpha = 0.18f))
                else listOf(Colores.vidrio, Color.Transparent)
            )
        )
        .border(BorderStroke(if (activo) 1.5.dp else 1.dp, if (activo) Colores.degradado else Colores.degradadoBorde), forma)

/** Iris holográfico: anillos que giran en sentidos opuestos, fibras del iris, pupila que late y barrido. */
@Composable
private fun IrisHolografico(modo: Modo, modifier: Modifier) {
    val t = rememberInfiniteTransition()
    val giro by t.animateFloat(0f, 360f, infiniteRepeatable(tween(16000, easing = LinearEasing)))
    val giro2 by t.animateFloat(360f, 0f, infiniteRepeatable(tween(9000, easing = LinearEasing)))
    val pupila by t.animateFloat(0.26f, 0.34f, infiniteRepeatable(tween(2200, easing = FastOutSlowInEasing), RepeatMode.Reverse))
    val barrido by t.animateFloat(0f, 1f, infiniteRepeatable(tween(2800, easing = LinearEasing)))
    val principal = when (modo) { Modo.OJO -> Colores.cian; Modo.ANEMIA -> Colores.magenta; Modo.ROSTRO -> Colores.violeta; Modo.PALMA -> Colores.atencion; Modo.ENCIA -> Colores.magenta }
    Canvas(modifier) {
        val c = Offset(size.width / 2, size.height / 2)
        val r = size.minDimension / 2
        // halo
        drawCircle(Brush.radialGradient(listOf(principal.copy(alpha = 0.35f), Color.Transparent), c, r * 1.1f), r * 1.1f, c)
        // anillo exterior con graduación
        rotate(giro, c) {
            for (i in 0 until 72) {
                val a = i * 5f * PI.toFloat() / 180f
                val l = if (i % 6 == 0) 10.dp.toPx() else 4.dp.toPx()
                drawLine(principal.copy(alpha = if (i % 6 == 0) 0.9f else 0.35f),
                    Offset(c.x + cos(a) * (r - l), c.y + sin(a) * (r - l)), Offset(c.x + cos(a) * r, c.y + sin(a) * r), 1.5.dp.toPx())
            }
        }
        // anillos de datos
        rotate(giro2, c) {
            drawCircle(Colores.violeta.copy(alpha = 0.7f), r * 0.82f, c,
                style = Stroke(2.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(30f, 14f, 6f, 14f))))
            drawArc(Colores.magenta, 20f, 70f, false, Offset(c.x - r * 0.74f, c.y - r * 0.74f), Size(r * 1.48f, r * 1.48f),
                style = Stroke(3.dp.toPx(), cap = StrokeCap.Round))
            drawArc(principal, 200f, 110f, false, Offset(c.x - r * 0.74f, c.y - r * 0.74f), Size(r * 1.48f, r * 1.48f),
                style = Stroke(3.dp.toPx(), cap = StrokeCap.Round))
        }
        // iris: fibras radiales
        val ri = r * 0.62f
        drawCircle(Brush.radialGradient(listOf(Colores.fondo, principal.copy(alpha = 0.45f), Colores.violeta.copy(alpha = 0.6f), Colores.fondo), c, ri), ri, c)
        rotate(giro * 0.3f, c) {
            for (i in 0 until 90) {
                val a = i * 4f * PI.toFloat() / 180f
                val r0 = ri * (0.36f + 0.05f * ((i * 13) % 5))
                drawLine(principal.copy(alpha = 0.18f + 0.25f * ((i * 7) % 4) / 4f),
                    Offset(c.x + cos(a) * r0, c.y + sin(a) * r0), Offset(c.x + cos(a) * ri * 0.96f, c.y + sin(a) * ri * 0.96f), 1.dp.toPx())
            }
        }
        drawCircle(principal.copy(alpha = 0.9f), ri, c, style = Stroke(1.5.dp.toPx()))
        // pupila
        drawCircle(Color.Black, ri * pupila * 1.6f, c)
        drawCircle(principal.copy(alpha = 0.6f), ri * pupila * 1.6f, c, style = Stroke(1.dp.toPx()))
        drawCircle(Color.White.copy(alpha = 0.85f), ri * 0.07f, Offset(c.x - ri * 0.18f, c.y - ri * 0.2f))
        // barrido vertical
        val y = c.y - ri + 2 * ri * barrido
        clipPath(androidx.compose.ui.graphics.Path().apply { addOval(androidx.compose.ui.geometry.Rect(c, ri)) }) {
        drawLine(Brush.horizontalGradient(listOf(Color.Transparent, principal, Color.Transparent), c.x - r, c.x + r),
            Offset(c.x - r * 0.9f, y), Offset(c.x + r * 0.9f, y), 2.dp.toPx())
        drawRect(Brush.verticalGradient(listOf(Color.Transparent, principal.copy(alpha = 0.12f)), y - 40.dp.toPx(), y),
            Offset(c.x - r * 0.9f, y - 40.dp.toPx()), Size(r * 1.8f, 40.dp.toPx()))
        }
    }
}
