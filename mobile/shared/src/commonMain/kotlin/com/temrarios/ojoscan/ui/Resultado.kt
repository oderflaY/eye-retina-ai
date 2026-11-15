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
import com.temrarios.ojoscan.Medicion
import com.temrarios.ojoscan.api.AnalisisPerfil
import com.temrarios.ojoscan.api.DatosPerfil
import com.temrarios.ojoscan.guia.Nivel
import com.temrarios.ojoscan.guia.Pronostico
import com.temrarios.ojoscan.guia.ResultadosChequeo
import com.temrarios.ojoscan.guia.pronosticar
import androidx.compose.runtime.mutableStateListOf
import com.temrarios.ojoscan.api.AnalisisOjo
import com.temrarios.ojoscan.api.AnalisisAnemia
import com.temrarios.ojoscan.api.AnalisisRostro
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
    class ListoRostro(val a: AnalisisRostro, val guardado: String?) : Envio
    class Fallo(val mensaje: String, val guardado: String?) : Envio
}

/** Una foto a analizar con el modo con que se tomó. */
private class Pieza(val modo: Modo, val r: Recorte)

private fun nombrePieza(p: Pieza) = when (p.modo) {
    Modo.OJO -> "Ojo ${p.r.ojo.nombre} de cerca"
    Modo.ANEMIA -> "Párpado"
    Modo.ROSTRO -> "Rostro"
    Modo.PALMA -> "Palma"
    Modo.ENCIA -> "Encía"
}

@Composable
fun PantallaResultado(
    ajustes: Ajustes, mediciones: List<Medicion>, perfil: DatosPerfil?, alRepetir: () -> Unit, alInicio: () -> Unit,
    /** Cada tamizaje de anemia válido se manda al tablero AstroPulse. */
    alAnemia: (Modo, AnalisisAnemia) -> Unit = { _, _ -> },
) {
    var perfilRes by remember(mediciones) { mutableStateOf<Result<AnalisisPerfil>?>(null) }
    val piezas = remember(mediciones) { mediciones.flatMap { m -> m.recortes.map { Pieza(m.modo, it) } } }
    val envios = remember(mediciones) { mutableStateListOf<Envio>().apply { repeat(piezas.size) { add(Envio.Enviando) } } }
    // Una por una: la PC con la IA atiende de a una y el intermediario corta a los 55 s.
    LaunchedEffect(mediciones) {
        val api = OjoApi(ajustes.servidor)
        // El perfil escrito primero: no tiene fotos y responde en un instante.
        if (perfil != null) perfilRes = runCatching { api.analizarPerfil(perfil) }
        for ((i, p) in piezas.withIndex()) {
            envios[i] = analizar(api, ajustes, p)
            (envios[i] as? Envio.ListoAnemia)?.a?.takeIf { it.valida }?.let { alAnemia(p.modo, it) }
        }
        api.cerrar()
    }
    val listos = envios.count { it !is Envio.Enviando } + (if (perfil != null && perfilRes != null) 1 else 0)
    val total = piezas.size + (if (perfil != null) 1 else 0)
    val completo = mediciones.size > 1 || ajustes.completo

    FondoEspacial {
        Column(
            Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(if (completo) "REPORTE // CHEQUEO COMPLETO" else when (mediciones.firstOrNull()?.modo) {
                Modo.ANEMIA -> "REPORTE // TAMIZAJE DE ANEMIA"
                Modo.ROSTRO -> "REPORTE // ROSTRO + OJOS"
                Modo.PALMA -> "REPORTE // ANEMIA POR LA PALMA"
                Modo.ENCIA -> "REPORTE // ANEMIA POR LA ENCÍA (EXPERIMENTAL)"
                else -> "REPORTE // ANÁLISIS OCULAR"
            }, style = Hud.etiqueta)
            TextoDegradado(if (completo) "Pronóstico" else "Resultados", 34)
            if (perfil != null) PanelPerfil(perfil, perfilRes, Modifier.fillMaxWidth())
            if (piezas.isEmpty() && perfil == null) {
                PanelVidrio(Modifier.fillMaxWidth()) {
                    Text("No se completó ninguna medición.", color = Colores.atencion, fontSize = 16.sp)
                }
            } else if (completo) {
                if (listos < total) ProgresoAnalisis(piezas, envios, listos, total)
                else PanelPronostico(pronosticar(juntar(piezas, envios).let { r ->
                    val pf = perfilRes
                    when {
                        pf == null -> r
                        pf.isSuccess -> r.copy(perfil = pf.getOrNull())
                        else -> r.copy(fallidas = r.fallidas + "Tus datos (${explicar(pf.exceptionOrNull()!!).take(60)})")
                    }
                }), Modifier.fillMaxWidth())
                TituloHud("≡", "Detalle de cada medición")
            }
            for ((i, p) in piezas.withIndex()) TarjetaOjo(p.modo, p.r, envios[i])
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                BotonContorno("Inicio", alInicio, Modifier.weight(1f))
                BotonNeon(if (completo) "Repetir chequeo" else "Re-escanear", alRepetir, Modifier.weight(1.3f))
            }
            Spacer(Modifier.height(8.dp))
        }
    }
}

private suspend fun analizar(api: OjoApi, ajustes: Ajustes, p: Pieza): Envio {
    val r = p.r
    // Las fotos de cara completa y de la mano nunca se donan: son biométricas.
    val guardado = if (ajustes.guardar && !p.modo.fotoLibre) {
        runCatching {
            api.guardarCaptura(r.jpeg, MetaCaptura(r.ojo.nombre, r.caja, r.puntos, r.apertura, consentimiento = true))
        }.fold({ "Escaneo donado al entrenamiento" }, { "No se pudo guardar: ${explicar(it)}" })
    } else null
    return when (p.modo) {
        Modo.ENCIA -> runCatching { api.analizarEncia(r.jpeg) }.fold({ Envio.ListoAnemia(it, guardado) }, { Envio.Fallo(explicar(it), guardado) })
        Modo.PALMA -> runCatching { api.analizarPalma(r.jpeg) }.fold({ Envio.ListoAnemia(it, guardado) }, { Envio.Fallo(explicar(it), guardado) })
        Modo.ANEMIA -> runCatching { api.analizarAnemia(r.jpeg) }.fold({ Envio.ListoAnemia(it, guardado) }, { Envio.Fallo(explicar(it), guardado) })
        Modo.ROSTRO -> runCatching { api.analizarRostro(r.jpeg) }.fold({ Envio.ListoRostro(it, guardado) }, { Envio.Fallo(explicar(it), guardado) })
        Modo.OJO -> runCatching { api.analizarOjo(r.jpeg) }.fold({ Envio.Listo(it, guardado) }, { Envio.Fallo(explicar(it), guardado) })
    }
}

/** Respuestas del servidor → entrada del pronóstico. */
private fun juntar(piezas: List<Pieza>, envios: List<Envio>): ResultadosChequeo {
    var r = ResultadosChequeo()
    val fallidas = mutableListOf<String>()
    for ((p, e) in piezas.zip(envios)) {
        r = when (e) {
            is Envio.Listo -> if (p.r.ojo == Ojo.DERECHO) r.copy(ojoDerecho = e.a) else r.copy(ojoIzquierdo = e.a)
            is Envio.ListoAnemia -> when (p.modo) {
                Modo.PALMA -> r.copy(palma = e.a)
                Modo.ENCIA -> r.copy(encia = e.a)
                else -> r.copy(conjuntiva = e.a)
            }
            is Envio.ListoRostro -> r.copy(rostro = e.a)
            is Envio.Fallo -> { fallidas += "${nombrePieza(p)} (${e.mensaje.take(60)})"; r }
            Envio.Enviando -> r
        }
    }
    return r.copy(fallidas = fallidas)
}

@Composable
private fun ProgresoAnalisis(piezas: List<Pieza>, envios: List<Envio>, listos: Int, total: Int) {
    PanelVidrio(Modifier.fillMaxWidth(), resaltado = true) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(Modifier.size(22.dp), color = Colores.cian, strokeWidth = 2.dp)
            Spacer(Modifier.size(12.dp))
            Column {
                Text("CRUZANDO MEDICIONES · $listos/$total", style = Hud.etiqueta.copy(fontSize = 10.sp))
                Text("El pronóstico aparece cuando lleguen todas.", color = Colores.textoSuave, fontSize = 13.sp)
            }
        }
        for ((p, e) in piezas.zip(envios)) {
            val c = when (e) { Envio.Enviando -> Colores.textoSuave; is Envio.Fallo -> Colores.mal; else -> Colores.bien }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Led(c, 8.dp)
                Spacer(Modifier.size(10.dp))
                Text(nombrePieza(p), color = Colores.texto, fontSize = 13.sp, modifier = Modifier.weight(1f))
                Text(when (e) { Envio.Enviando -> "…"; is Envio.Fallo -> "falló"; else -> "listo" },
                    style = Hud.dato.copy(fontSize = 10.sp, color = c))
            }
        }
    }
}

/** Encabezado del perfil médico: quién es la persona y lo que salió de sus datos escritos. */
@Composable
private fun PanelPerfil(d: DatosPerfil, res: Result<AnalisisPerfil>?, modifier: Modifier) {
    PanelVidrio(modifier) {
        TituloHud("☰", "Perfil médico")
        Text(d.nombre.ifBlank { "Paciente" }, color = Colores.texto, fontSize = 20.sp, fontWeight = FontWeight.Black)
        Text(listOfNotNull(
            d.edad?.let { "$it años" }, when (d.sexo) { "M" -> "hombre"; "F" -> "mujer"; else -> null },
            d.peso?.let { "${fmt(it)} kg" }, d.estatura?.let { "${fmt(it)} cm" },
            d.sistolica?.let { s -> d.diastolica?.let { "${fmt(s)}/${fmt(it)} mmHg" } },
        ).joinToString(" · "), style = Hud.dato.copy(fontSize = 12.sp, color = Colores.textoSuave))
        when {
            res == null -> Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(Modifier.size(18.dp), color = Colores.cian, strokeWidth = 2.dp)
                Spacer(Modifier.size(10.dp))
                Text("Calculando tu perfil…", color = Colores.textoSuave, fontSize = 13.sp)
            }
            res.isFailure -> Text(explicar(res.exceptionOrNull()!!), color = Colores.atencion, fontSize = 13.sp)
            else -> {
                val a = res.getOrThrow()
                for (i in a.indicadores) {
                    val c = when (i.nivel) { "alta" -> Colores.mal; "media" -> Colores.atencion; else -> Colores.bien }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Led(c, 8.dp)
                        Spacer(Modifier.size(10.dp))
                        Column(Modifier.weight(1f)) {
                            Text(i.nombre.uppercase(), style = Hud.etiqueta.copy(fontSize = 8.sp, color = c))
                            Text(i.texto, color = Colores.texto, fontSize = 13.sp, lineHeight = 16.sp)
                            if (i.fuente.isNotBlank()) Text(i.fuente, style = Hud.dato.copy(fontSize = 8.sp, color = Colores.textoSuave))
                        }
                        Spacer(Modifier.size(8.dp))
                        Text(i.valor, color = c, fontWeight = FontWeight.Black, fontSize = 15.sp)
                    }
                }
                if (a.faltan.isNotEmpty()) Text("Para completarlo falta: " + a.faltan.joinToString(", ") + ".",
                    color = Colores.textoSuave, fontSize = 11.sp, lineHeight = 14.sp)
            }
        }
    }
}

private fun fmt(x: Double) = if (x % 1.0 == 0.0) x.toLong().toString() else ((x * 10).roundToInt() / 10.0).toString()

@Composable
private fun PanelPronostico(p: Pronostico, modifier: Modifier) {
    val color = when (p.nivel) { Nivel.ALTA -> Colores.mal; Nivel.MEDIA -> Colores.atencion; Nivel.OK -> Colores.bien }
    PanelVidrio(modifier, resaltado = true) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Led(color, 14.dp)
            Spacer(Modifier.size(12.dp))
            Column(Modifier.weight(1f)) {
                Text(when (p.nivel) { Nivel.ALTA -> "ATENCIÓN MÉDICA"; Nivel.MEDIA -> "VIGILAR"; Nivel.OK -> "TODO EN ORDEN" },
                    style = Hud.etiqueta.copy(fontSize = 10.sp, color = color))
                Text(p.resumen, color = Colores.texto, fontSize = 19.sp, fontWeight = FontWeight.Black, lineHeight = 23.sp)
            }
        }
        for (c in p.conclusiones) {
            val cc = when (c.nivel) { Nivel.ALTA -> Colores.mal; Nivel.MEDIA -> Colores.atencion; Nivel.OK -> Colores.bien }
            Column(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(cc.copy(alpha = 0.10f))
                    .border(1.dp, cc.copy(alpha = 0.5f), RoundedCornerShape(14.dp)).padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(c.area.uppercase(), style = Hud.etiqueta.copy(fontSize = 9.sp, color = cc))
                Text(c.titulo, color = Colores.texto, fontWeight = FontWeight.Bold, fontSize = 15.sp, lineHeight = 19.sp)
                if (c.detalle.isNotBlank()) Text(c.detalle, color = Colores.textoSuave, fontSize = 13.sp, lineHeight = 17.sp)
                Text("Basado en: " + c.fuentes.joinToString(" · "), style = Hud.dato.copy(fontSize = 9.sp, color = Colores.textoSuave))
            }
        }
        Text("Mediciones usadas (${p.usadas.size}): " + p.usadas.joinToString(", ").ifEmpty { "ninguna" },
            style = Hud.dato.copy(fontSize = 10.sp, color = Colores.textoSuave))
        if (p.fallidas.isNotEmpty()) Text("Sin resultado: " + p.fallidas.joinToString("; ") + ". Repite esos pasos para un pronóstico más completo.",
            color = Colores.atencion, fontSize = 11.sp, lineHeight = 14.sp)
        Text("Orientativo, hecho con datos públicos limitados. No es un diagnóstico ni sustituye la consulta médica.",
            color = Colores.textoSuave.copy(alpha = 0.7f), fontSize = 10.sp, lineHeight = 13.sp)
    }
}

@Composable
private fun TarjetaOjo(modo: Modo, r: Recorte, envio: Envio) {
    PanelVidrio(Modifier.fillMaxWidth(), resaltado = envio !is Envio.Enviando) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(when (modo) {
                Modo.PALMA -> "PALMA DE LA MANO"
                Modo.ENCIA -> "ENCÍA · EXPERIMENTAL"
                Modo.ROSTRO -> "ROSTRO COMPLETO"
                Modo.ANEMIA -> "PÁRPADO · OJO ${r.ojo.nombre.uppercase()}"
                Modo.OJO -> if (r.ojo == Ojo.DERECHO) "OJO DERECHO" else "OJO IZQUIERDO"
            }, style = Hud.etiqueta, modifier = Modifier.weight(1f))
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
                            Text("EN COLA PARA LA RED NEURONAL", style = Hud.etiqueta.copy(fontSize = 10.sp))
                            Text("Enviando al núcleo de IA…", color = Colores.textoSuave, fontSize = 13.sp)
                        }
                    }
                    is Envio.Listo -> Analisis(e.a)
                    is Envio.ListoAnemia -> Anemia(e.a)
                    is Envio.ListoRostro -> Rostro(e.a)
                    is Envio.Fallo -> {
                        Text("ENLACE INTERRUMPIDO", style = Hud.etiqueta.copy(color = Colores.mal))
                        Text(e.mensaje, color = Colores.atencion, fontSize = 14.sp)
                    }
                }
                val guardado = when (e) {
                    is Envio.Listo -> e.guardado
                    is Envio.ListoAnemia -> e.guardado
                    is Envio.ListoRostro -> e.guardado
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
        if (r.puntos.isNotEmpty()) Text("IA · ${r.puntos.size} PUNTOS", style = Hud.etiqueta.copy(fontSize = 9.sp, color = Colores.texto),
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

@Composable
private fun Rostro(a: AnalisisRostro) {
    if (!a.cara) {
        TituloHud("!", "Sin cara")
        Text(a.mensaje ?: a.error ?: "No se encontró una cara.", color = Colores.atencion, fontSize = 15.sp, lineHeight = 20.sp)
        return
    }
    for (aviso in a.avisosFoto) Text(aviso, color = Colores.atencion, fontSize = 12.sp)

    // Lectura combinada cara + ojos: lo más importante va primero
    TituloHud("⊕", "Lectura combinada")
    for (n in a.combinado) {
        val c = when (n.nivel) { "alta" -> Colores.mal; "media" -> Colores.atencion; else -> Colores.bien }
        Row(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(c.copy(alpha = 0.12f))
                .border(1.dp, c.copy(alpha = 0.6f), RoundedCornerShape(14.dp)).padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Led(c, 10.dp)
            Spacer(Modifier.size(10.dp))
            Text(n.texto, color = Colores.texto, fontSize = 14.sp, lineHeight = 19.sp)
        }
    }

    TituloHud("Σ", "Rostro")
    for (h in a.hallazgos) {
        val c = when (h.resultado) { "posible" -> Colores.mal; "no_concluyente" -> Colores.atencion; else -> Colores.bien }
        val veredicto = when (h.resultado) { "posible" -> "señales presentes"; "no_concluyente" -> "no concluyente"; else -> "sin señales" }
        BarraNeon("${h.nombre} · $veredicto", h.probabilidad.toFloat(), c, marcas = listOf(0.35f, 0.65f),
            detalle = h.validacion.let { v ->
                listOfNotNull(
                    (v["auc_por_persona"] ?: v["auc"])?.let { "AUC ${(it * 100).roundToInt()}%" },
                    v["sensibilidad"]?.let { "detecta ${(it * 100).roundToInt()}%" },
                    v["falsas_alarmas_caras_sanas"]?.let { "falsas alarmas ${(it * 100).roundToInt()}%" },
                ).joinToString(" · ")
            })
    }

    if (a.ojos.isNotEmpty()) {
        TituloHud("◉", "Ojos en la misma foto")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for ((lado, o) in a.ojos.entries.sortedBy { it.key }) {
                val c = if (o.principal == "normal") Colores.bien else Colores.atencion
                Column(
                    Modifier.weight(1f).clip(RoundedCornerShape(12.dp)).background(Colores.vidrio).padding(10.dp),
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text("OJO ${lado.uppercase()}", style = Hud.etiqueta.copy(fontSize = 8.sp, color = c))
                    Text(o.titulo, color = Colores.texto, fontWeight = FontWeight.Bold, fontSize = 13.sp, lineHeight = 16.sp)
                    Text("${(o.confianza * 100).roundToInt()}% conf.", style = Hud.dato.copy(fontSize = 10.sp, color = Colores.textoSuave))
                }
            }
        }
        Text("Desde lejos el ojo sale pequeño: la foto de cerca es la que manda.", color = Colores.textoSuave, fontSize = 11.sp)
    }
    if (a.aviso.isNotBlank()) Text(a.aviso, color = Colores.textoSuave.copy(alpha = 0.8f), fontSize = 11.sp, lineHeight = 14.sp)
}
