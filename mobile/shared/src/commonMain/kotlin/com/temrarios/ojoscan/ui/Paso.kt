package com.temrarios.ojoscan.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bloodtype
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.PanTool
import androidx.compose.material.icons.filled.RemoveRedEye
import androidx.compose.material.icons.filled.SentimentVerySatisfied
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.temrarios.ojoscan.Paso
import com.temrarios.ojoscan.guia.Modo

fun Modo.icono(): ImageVector = when (this) {
    Modo.OJO -> Icons.Filled.RemoveRedEye
    Modo.ANEMIA -> Icons.Filled.Bloodtype
    Modo.ROSTRO -> Icons.Filled.Face
    Modo.PALMA -> Icons.Filled.PanTool
    Modo.ENCIA -> Icons.Filled.SentimentVerySatisfied
}

/** Nombre del paso tal como se ve en el chequeo completo. */
fun Paso.nombre(): String = when (modo) {
    Modo.OJO -> "Tus dos ojos de cerca"
    Modo.ANEMIA -> "Párpado interno"
    Modo.ROSTRO -> "Tu cara completa"
    Modo.PALMA -> "Palma de la mano"
    Modo.ENCIA -> "Tu encía"
}

/** Qué aporta cada paso al pronóstico. */
private fun Modo.paraQue(): String = when (this) {
    Modo.OJO -> "Ojo rojo, pterigión y catarata. Primero el izquierdo y luego el derecho."
    Modo.ANEMIA -> "El color de adentro del párpado: es la primera señal de anemia."
    Modo.ROSTRO -> "Asimetría de la cara, ojos amarillos y una segunda mirada a tus ojos."
    Modo.PALMA -> "Confirma la anemia con otra parte del cuerpo."
    Modo.ENCIA -> "El color de la encía: otra mucosa que palidece con la anemia. Experimental: apoya al párpado."
}

private fun Modo.instrucciones(): String = when (this) {
    Modo.OJO -> "Acerca la cámara a tu ojo, míralo de frente y ábrelo bien. La foto se toma sola cuando todo esté en verde."
    Modo.ANEMIA -> "Jala tu párpado de abajo con un dedo y mira hacia arriba, que se vea la parte rosada. Mejor con luz blanca de día."
    Modo.ROSTRO -> "Cara de frente y relajada, a un brazo de distancia. Sin lentes ni fleco sobre los ojos. Pulsa «Tomar foto»."
    Modo.PALMA -> "Mano abierta con la palma hacia la cámara y los dedos un poco separados. Luz de día, sin flash."
    Modo.ENCIA -> "Cara de frente a un brazo de distancia. Junta los dientes y sonríe levantando el labio de arriba " +
        "(con un dedo si hace falta) para que se vea la encía rosada. Luz de día, sin flash. Pulsa «Tomar foto»."
}

/** Antes de cada paso del chequeo completo: qué sigue, para qué sirve y cómo hacerlo. */
@Composable
fun PantallaPaso(pasos: List<Paso>, indice: Int, alEmpezar: () -> Unit, alSaltar: () -> Unit, alCancelar: () -> Unit) {
    val paso = pasos[indice]
    FondoEspacial {
        Column(
            Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            Text("CHEQUEO COMPLETO // PASO ${indice + 2} DE ${pasos.size + 1}", style = Hud.etiqueta.copy(color = Colores.textoSuave, fontSize = 10.sp))
            TextoDegradado(paso.nombre(), 32)

            // Línea de progreso con los pasos
            PanelVidrio(Modifier.fillMaxWidth()) {
                for ((i, p) in pasos.withIndex()) {
                    val color = when {
                        i < indice -> Colores.bien
                        i == indice -> Colores.cian
                        else -> Colores.textoSuave.copy(alpha = 0.5f)
                    }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            Modifier.size(34.dp).background(color.copy(alpha = if (i == indice) 0.2f else 0.08f), CircleShape)
                                .border(1.5.dp, color, CircleShape),
                            contentAlignment = Alignment.Center,
                        ) {
                            if (i < indice) Text("✓", color = color, fontWeight = FontWeight.Black)
                            else Icon(p.modo.icono(), null, tint = color, modifier = Modifier.size(18.dp))
                        }
                        Spacer(Modifier.size(12.dp))
                        Text(p.nombre(), color = if (i == indice) Colores.texto else Colores.textoSuave,
                            fontWeight = if (i == indice) FontWeight.Bold else FontWeight.Normal, fontSize = 15.sp)
                    }
                }
            }

            PanelVidrio(Modifier.fillMaxWidth(), resaltado = true) {
                TituloHud("◆", "Para qué sirve")
                Text(paso.modo.paraQue(), color = Colores.texto, fontSize = 15.sp, lineHeight = 20.sp)
                TituloHud("▶", "Cómo hacerlo")
                Text(paso.modo.instrucciones(), color = Colores.textoSuave, fontSize = 14.sp, lineHeight = 19.sp)
            }

            BotonNeon("Empezar paso ${indice + 2}", alEmpezar, Modifier.fillMaxWidth())
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                BotonContorno("Cancelar", alCancelar, Modifier.weight(1f))
                BotonContorno(if (indice + 1 < pasos.size) "Saltar paso" else "Saltar y ver resultado", alSaltar, Modifier.weight(1.3f))
            }
            Text(
                "Mientras más pasos completes, más fiable es el pronóstico: cada medición confirma o descarta a las otras.",
                color = Colores.textoSuave.copy(alpha = 0.8f), fontSize = 11.sp, lineHeight = 14.sp,
                textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(8.dp))
        }
    }
}
