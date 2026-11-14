package com.temrarios.ojoscan.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.temrarios.ojoscan.Ajustes
import com.temrarios.ojoscan.api.AnalisisOjo
import com.temrarios.ojoscan.api.AnalisisAnemia
import com.temrarios.ojoscan.api.MetaCaptura
import com.temrarios.ojoscan.api.OjoApi
import com.temrarios.ojoscan.api.explicar
import com.temrarios.ojoscan.guia.Modo
import com.temrarios.ojoscan.guia.Ojo
import com.temrarios.ojoscan.platform.Recorte
import com.temrarios.ojoscan.platform.decodificarImagen
import kotlin.math.roundToInt

private sealed interface Envio {
    data object Enviando : Envio
    class Listo(val a: AnalisisOjo, val guardado: String?) : Envio
    class ListoAnemia(val a: AnalisisAnemia, val guardado: String?) : Envio
    class Fallo(val mensaje: String, val guardado: String?) : Envio
}

@Composable
fun PantallaResultado(ajustes: Ajustes, recortes: List<Recorte>, alRepetir: () -> Unit, alInicio: () -> Unit) {
    FondoEspacial {
        Column(
            Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(if (ajustes.modo == Modo.ANEMIA) "REPORTE // TAMIZAJE DE ANEMIA" else "REPORTE // ANÁLISIS OCULAR", style = Hud.etiqueta)
            TextoDegradado("Resultados", 34)
            for (r in recortes) TarjetaOjo(ajustes, r)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                BotonContorno("Inicio", alInicio, Modifier.weight(1f))
                BotonNeon("Re-escanear", alRepetir, Modifier.weight(1.3f))
            }
            Spacer(Modifier.height(8.dp))
        }
    }
}

@Composable
private fun TarjetaOjo(ajustes: Ajustes, r: Recorte) {
    var envio by remember(r) { mutableStateOf<Envio>(Envio.Enviando) }
    LaunchedEffect(r) {
        val api = OjoApi(ajustes.servidor)
        val guardado = if (ajustes.guardar) {
            runCatching {
                api.guardarCaptura(r.jpeg, MetaCaptura(r.ojo.nombre, r.caja, r.puntos, r.apertura, consentimiento = true))
            }.fold({ "Escaneo donado al entrenamiento" }, { "No se pudo guardar: ${explicar(it)}" })
        } else null
        envio = if (ajustes.modo == Modo.ANEMIA) {
            runCatching { api.analizarAnemia(r.jpeg) }
                .fold({ Envio.ListoAnemia(it, guardado) }, { Envio.Fallo(explicar(it), guardado) })
        } else {
            runCatching { api.analizarOjo(r.jpeg) }
                .fold({ Envio.Listo(it, guardado) }, { Envio.Fallo(explicar(it), guardado) })
        }
        api.cerrar()
    }

    PanelVidrio(Modifier.fillMaxWidth(), resaltado = envio !is Envio.Enviando) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text((if (r.ojo == Ojo.IZQUIERDO) "OJO IZQUIERDO" else "OJO DERECHO"), style = Hud.etiqueta, modifier = Modifier.weight(1f))
            r.apertura?.let { Text("APERTURA ${(it * 100).roundToInt()}", style = Hud.dato.copy(fontSize = 10.sp, color = Colores.textoSuave)) }
        }
        FotoEscaneada(r, analizando = envio is Envio.Enviando)
        AnimatedContent(envio, transitionSpec = { fadeIn(tween(500)) togetherWith fadeOut(tween(200)) }) { e ->
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                when (e) {
                    Envio.Enviando -> Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(Modifier.size(22.dp), color = Colores.cian, strokeWidth = 2.dp)
                        Spacer(Modifier.size(12.dp))
                        Column {
                            Text("PROCESANDO EN LA RED NEURONAL", style = Hud.etiqueta.copy(fontSize = 10.sp))
                            Text("Enviando al núcleo de IA…", color = Colores.textoSuave, fontSize = 13.sp)
                        }
                    }
                    is Envio.Listo -> Analisis(e.a)
                    is Envio.ListoAnemia -> Anemia(e.a)
                    is Envio.Fallo -> {
                        Text("ENLACE INTERRUMPIDO", style = Hud.etiqueta.copy(color = Colores.mal))
                        Text(e.mensaje, color = Colores.atencion, fontSize = 14.sp)
                    }
                }
                val guardado = when (e) {
                    is Envio.Listo -> e.guardado
                    is Envio.ListoAnemia -> e.guardado
                    is Envio.Fallo -> e.guardado
                    else -> null
                }
                guardado?.let { Text(it, style = Hud.dato.copy(fontSize = 10.sp, color = Colores.textoSuave)) }
            }
        }
    }
}

/** La foto del ojo con los puntos que encontró la IA y una línea de barrido mientras se analiza. */
@Composable
private fun FotoEscaneada(r: Recorte, analizando: Boolean) {
    val imagen = remember(r) { decodificarImagen(r.jpeg) } ?: return
    val t = rememberInfiniteTransition()
    val barrido by t.animateFloat(0f, 1f, infiniteRepeatable(tween(1600, easing = LinearEasing), RepeatMode.Reverse))
    val latido by t.animateFloat(0.4f, 1f, infiniteRepeatable(tween(900), RepeatMode.Reverse))
    val forma = RoundedCornerShape(16.dp)
    Box(Modifier.fillMaxWidth().aspectRatio(r.ancho.toFloat() / r.alto).clip(forma).border(1.dp, Colores.degradadoBorde, forma)) {
        Image(imagen, "Foto del ojo", Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
        Canvas(Modifier.fillMaxSize()) {
            val k = size.width / r.ancho
            drawRect(Brush.verticalGradient(listOf(Color.Transparent, Colores.fondo.copy(alpha = 0.55f))))
            // puntos de referencia
            for ((nombre, p) in r.puntos) {
                if (p.size < 2) continue
                val o = Offset(p[0] * k, p[1] * k)
                val c = if (nombre == "pupil" || nombre == "iris") Colores.magenta else Colores.cian
                drawCircle(c.copy(alpha = 0.3f * latido), 9.dp.toPx(), o)
                drawCircle(c, 3.dp.toPx(), o)
            }
            // caja del ojo con esquinas de mira
            r.caja?.takeIf { it.size == 4 }?.let { (x, y, w, h) ->
                val l = 14.dp.toPx()
                val tl = Offset(x * k, y * k)
                val br = Offset((x + w) * k, (y + h) * k)
                val s = 2.dp.toPx()
                drawLine(Colores.cian, tl, Offset(tl.x + l, tl.y), s); drawLine(Colores.cian, tl, Offset(tl.x, tl.y + l), s)
                drawLine(Colores.cian, br, Offset(br.x - l, br.y), s); drawLine(Colores.cian, br, Offset(br.x, br.y - l), s)
                drawLine(Colores.cian, Offset(br.x, tl.y), Offset(br.x - l, tl.y), s); drawLine(Colores.cian, Offset(br.x, tl.y), Offset(br.x, tl.y + l), s)
                drawLine(Colores.cian, Offset(tl.x, br.y), Offset(tl.x + l, br.y), s); drawLine(Colores.cian, Offset(tl.x, br.y), Offset(tl.x, br.y - l), s)
            }
            if (analizando) {
                val y = size.height * barrido
                drawRect(Brush.verticalGradient(listOf(Color.Transparent, Colores.cian.copy(alpha = 0.25f)), y - 50.dp.toPx(), y),
                    Offset(0f, y - 50.dp.toPx()), androidx.compose.ui.geometry.Size(size.width, 50.dp.toPx()))
                drawLine(Colores.cian, Offset(0f, y), Offset(size.width, y), 2.dp.toPx())
            }
        }
        Text("IA · ${r.puntos.size} PUNTOS", style = Hud.etiqueta.copy(fontSize = 9.sp, color = Colores.texto),
            modifier = Modifier.align(Alignment.BottomStart).padding(10.dp))
    }
}

@Composable
private fun Analisis(a: AnalisisOjo) {
    val normal = a.principal == "normal"
    val color = when {
        normal -> Colores.bien
        a.confianza < 0.6 -> Colores.atencion
        else -> Colores.mal
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        AnilloConfianza(a.confianza.toFloat(), color, "confianza", Modifier.size(132.dp))
        Spacer(Modifier.size(14.dp))
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("DIAGNÓSTICO IA", style = Hud.etiqueta.copy(fontSize = 9.sp, color = color))
            Text(a.titulo, color = Colores.texto, fontSize = 20.sp, fontWeight = FontWeight.Black, lineHeight = 24.sp)
        }
    }
    if (a.confianza < 0.6) {
        Text("El modelo no está seguro. Repite la foto con buena luz y el ojo bien abierto.",
            color = Colores.atencion, fontSize = 12.sp)
    }
    if (a.descripcion.isNotBlank()) Text(a.descripcion, color = Colores.textoSuave, fontSize = 14.sp, lineHeight = 19.sp)
    if (a.hallazgos.isNotEmpty()) TituloHud("Σ", "Probabilidades")
    for (h in a.hallazgos) BarraNeon(h.nombre, h.probabilidad.toFloat(), if (h.clave == "normal") Colores.bien else Colores.atencion)
    if (a.aviso.isNotBlank()) Text(a.aviso, color = Colores.textoSuave.copy(alpha = 0.7f), fontSize = 10.sp, lineHeight = 13.sp)
}

@Composable
private fun Anemia(a: AnalisisAnemia) {
    val color = when {
        !a.valida || a.resultado == "no_concluyente" -> Colores.atencion
        a.resultado == "palidez" -> Colores.mal
        else -> Colores.bien
    }
    val p = a.probabilidad
    if (a.valida && p != null) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            AnilloConfianza(p.toFloat(), color, "prob. anemia", Modifier.size(132.dp))
            Spacer(Modifier.size(14.dp))
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("VEREDICTO", style = Hud.etiqueta.copy(fontSize = 9.sp, color = color))
                Text(a.titulo, color = Colores.texto, fontSize = 20.sp, fontWeight = FontWeight.Black, lineHeight = 24.sp)
            }
        }
    } else {
        Text(a.titulo, color = color, fontSize = 20.sp, fontWeight = FontWeight.Black)
    }
    if (a.descripcion.isNotBlank()) Text(a.descripcion, color = Colores.textoSuave, fontSize = 14.sp, lineHeight = 19.sp)
    if (a.valida && p != null) {
        val zona = a.zonaGris?.takeIf { it.size == 2 }
        BarraNeon(
            "Probabilidad de anemia", p.toFloat(), color,
            marcas = zona?.map { it.toFloat() } ?: listOfNotNull(a.umbral?.toFloat()),
            detalle = zona?.let { "Veredicto claro por debajo de ${(it[0] * 100).roundToInt()} % o desde ${(it[1] * 100).roundToInt()} %; entre las marcas no se da veredicto." }
                ?: a.umbral?.let { "Se marca anemia a partir de ${(it * 100).roundToInt()} %." },
        )
    }
    a.advertenciaLuz?.let { Text(it, color = Colores.atencion, fontSize = 12.sp) }
    a.validacion?.let { v ->
        val datos = listOfNotNull(
            v.sensibilidad?.let { "DETECCIÓN" to "${(it * 100).roundToInt()}%" },
            v.especificidad?.let { "ACIERTO SANOS" to "${(it * 100).roundToInt()}%" },
            v.cobertura?.let { "VEREDICTO" to "${(it * 100).roundToInt()}%" },
            v.n?.let { "MUESTRAS" to "$it" },
        )
        if (datos.isNotEmpty()) {
            TituloHud("◆", "Fiabilidad del modelo")
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for ((k, valor) in datos) {
                    Column(
                        Modifier.weight(1f).clip(RoundedCornerShape(12.dp)).background(Colores.vidrio).padding(vertical = 8.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        Text(valor, color = Colores.cian, fontWeight = FontWeight.Black, fontSize = 16.sp)
                        Text(k, style = Hud.etiqueta.copy(fontSize = 7.sp, letterSpacing = 1.sp, color = Colores.textoSuave), textAlign = TextAlign.Center)
                    }
                }
            }
        }
    }
    if (a.aviso.isNotBlank()) Text(a.aviso, color = Colores.textoSuave.copy(alpha = 0.7f), fontSize = 10.sp, lineHeight = 13.sp)
}
