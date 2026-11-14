package com.temrarios.ojoscan.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
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

        GuiaOvalo(guia, Modifier.fillMaxSize())

        Column(Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = alVolver) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, "Volver", tint = Color.White)
                }
                Text(
                    (if (ojoActual == Ojo.IZQUIERDO) "Ojo izquierdo" else "Ojo derecho") +
                        (if (orden.size > 1) " (${indice + 1} de 2)" else "") +
                        (if (ajustes.modo == Modo.ANEMIA) " · anemia" else ""),
                    color = Color.White,
                    style = MaterialTheme.typography.titleMedium,
                )
            }
            if (ajustes.modo == Modo.ANEMIA) {
                Text(
                    "Con un dedo jala tu párpado de abajo hacia abajo y mira hacia arriba, " +
                        "que se vea la parte rosada de adentro.",
                    color = Color.White,
                    style = MaterialTheme.typography.titleMedium,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth().background(Colores.velo, RoundedCornerShape(16.dp)).padding(12.dp),
                )
            }
            Spacer(Modifier.weight(1f))
            val texto = error ?: repetir ?: if (revisando) "Revisando la foto…" else guia.aviso.texto
            val color = when {
                error != null || repetir != null -> Colores.mal
                revisando || guia.aviso.bueno -> Colores.bien
                guia.aviso == Aviso.SIN_OJO -> Colores.mal
                else -> Colores.atencion
            }
            Row(
                Modifier.fillMaxWidth().background(Colores.velo, RoundedCornerShape(20.dp)).padding(20.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
            ) {
                if (revisando) {
                    CircularProgressIndicator(Modifier.size(24.dp), color = Colores.bien, strokeWidth = 3.dp)
                    Spacer(Modifier.size(12.dp))
                }
                Text(
                    texto,
                    color = color,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                )
            }
            Text(
                "La foto se toma sola cuando todo está bien.",
                color = Color.White.copy(alpha = 0.8f),
                style = MaterialTheme.typography.bodySmall,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
            )
        }
    }
}

/** Velo oscuro con una ventana en forma de ojo, centrada (donde el detector espera el ojo), y un anillo de progreso. */
@Composable
private fun GuiaOvalo(guia: Guia, modifier: Modifier) {
    val color by animateColorAsState(
        when {
            guia.aviso.bueno -> Colores.bien
            guia.aviso == Aviso.SIN_OJO -> Colores.mal
            else -> Colores.atencion
        }
    )
    val progreso by animateFloatAsState(guia.progreso)
    Canvas(modifier.graphicsLayer(compositingStrategy = CompositingStrategy.Offscreen)) {
        val ancho = size.width * 0.86f
        val alto = ancho * 0.68f
        val ovalo = Rect(Offset((size.width - ancho) / 2, size.height * 0.5f - alto / 2), Size(ancho, alto))
        drawRect(Colores.velo)
        drawOval(Color.Black, ovalo.topLeft, ovalo.size, blendMode = BlendMode.Clear)
        drawOval(color, ovalo.topLeft, ovalo.size, style = Stroke(6.dp.toPx()))
        if (progreso > 0f) {
            val m = 10.dp.toPx()
            drawArc(
                Colores.bien, -90f, 360f * progreso, false,
                Offset(ovalo.left - m, ovalo.top - m), Size(ovalo.width + 2 * m, ovalo.height + 2 * m),
                style = Stroke(8.dp.toPx()),
            )
        }
    }
}
