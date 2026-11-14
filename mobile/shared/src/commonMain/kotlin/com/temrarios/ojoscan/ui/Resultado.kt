package com.temrarios.ojoscan.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
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
    Surface(Modifier.fillMaxSize()) {
        Column(
            Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(if (ajustes.modo == Modo.ANEMIA) "Tamizaje de anemia" else "Análisis del ojo", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
            for (r in recortes) TarjetaOjo(ajustes, r)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedButton(onClick = alInicio, modifier = Modifier.weight(1f)) { Text("Inicio") }
                Button(onClick = alRepetir, modifier = Modifier.weight(1f)) { Text("Escanear otra vez") }
            }
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
            }.fold({ "Foto guardada para entrenar" }, { "No se pudo guardar: ${explicar(it)}" })
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

    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(
                if (r.ojo == Ojo.IZQUIERDO) "Ojo izquierdo" else "Ojo derecho",
                style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold,
            )
            val imagen = remember(r) { decodificarImagen(r.jpeg) }
            if (imagen != null) {
                Image(
                    imagen, "Foto del ojo",
                    Modifier.fillMaxWidth().aspectRatio(r.ancho.toFloat() / r.alto).clip(RoundedCornerShape(12.dp)),
                    contentScale = ContentScale.Crop,
                )
            }
            when (val e = envio) {
                Envio.Enviando -> Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
                    Text("  Analizando…")
                }
                is Envio.Listo -> {
                    Analisis(e.a)
                    e.guardado?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                }
                is Envio.ListoAnemia -> {
                    Anemia(e.a)
                    e.guardado?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                }
                is Envio.Fallo -> {
                    Text(e.mensaje, color = Colores.atencion)
                    e.guardado?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                }
            }
        }
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
    Text(a.titulo, color = color, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
    Text("Confianza ${(a.confianza * 100).roundToInt()} %", style = MaterialTheme.typography.bodyMedium)
    if (a.confianza < 0.6) {
        Text(
            "El modelo no está seguro. Repite la foto con buena luz y el ojo bien abierto.",
            style = MaterialTheme.typography.bodySmall, color = Colores.atencion,
        )
    }
    if (a.descripcion.isNotBlank()) Text(a.descripcion, style = MaterialTheme.typography.bodyMedium)
    for (h in a.hallazgos) Barra(h.nombre, h.probabilidad.toFloat(), if (h.clave == "normal") Colores.bien else Colores.atencion)
    if (a.aviso.isNotBlank()) {
        Text(a.aviso, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun Anemia(a: AnalisisAnemia) {
    val color = when {
        !a.valida -> Colores.atencion
        a.resultado == "palidez" -> Colores.mal
        else -> Colores.bien
    }
    Text(a.titulo, color = color, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
    if (a.descripcion.isNotBlank()) Text(a.descripcion, style = MaterialTheme.typography.bodyMedium)
    val p = a.probabilidad
    if (a.valida && p != null) {
        Barra("Probabilidad de anemia", p.toFloat(), if (a.resultado == "palidez") Colores.mal else Colores.bien)
        a.umbral?.let {
            Text("Se marca anemia a partir de ${(it * 100).roundToInt()} %.", style = MaterialTheme.typography.bodySmall)
        }
    }
    a.advertenciaLuz?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = Colores.atencion) }
    a.validacion?.let { v ->
        val partes = listOfNotNull(
            v.sensibilidad?.let { "detecta ${(it * 100).roundToInt()} % de las anemias" },
            v.especificidad?.let { "${(it * 100).roundToInt()} % de aciertos en personas sin anemia" },
            v.n?.let { "probado con $it fotos" },
        )
        if (partes.isNotEmpty()) {
            Text("Fiabilidad: " + partes.joinToString(", ") + ".", style = MaterialTheme.typography.bodySmall)
        }
    }
    if (a.aviso.isNotBlank()) {
        Text(a.aviso, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun Barra(nombre: String, p: Float, color: androidx.compose.ui.graphics.Color) {
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Row {
            Text(nombre, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
            Text("${(p * 100).roundToInt()} %", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
        }
        Box(
            Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant),
        ) {
            Box(Modifier.fillMaxHeight().fillMaxWidth(p.coerceIn(0f, 1f)).background(color))
        }
    }
}
