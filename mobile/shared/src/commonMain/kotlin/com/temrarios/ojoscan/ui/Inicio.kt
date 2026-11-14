package com.temrarios.ojoscan.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.RemoveRedEye
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.temrarios.ojoscan.Ajustes
import com.temrarios.ojoscan.api.OjoApi
import com.temrarios.ojoscan.api.explicar
import com.temrarios.ojoscan.guia.Modo
import com.temrarios.ojoscan.guia.Ojo
import kotlinx.coroutines.launch

@Composable
fun PantallaInicio(ajustes: Ajustes, alCambiar: (Ajustes) -> Unit, alEscanear: () -> Unit) {
    val alcance = rememberCoroutineScope()
    var prueba by remember { mutableStateOf<String?>(null) }

    Surface(Modifier.fillMaxSize()) {
        Column(
            Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.RemoveRedEye, null, Modifier.size(40.dp), tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.size(12.dp))
                Column {
                    Text("OjoScan", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
                    Text("Foto guiada del ojo para Eye-Retina AI", style = MaterialTheme.typography.bodyMedium)
                }
            }

            Seccion("¿Qué quieres revisar?") {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (m in Modo.entries) {
                        FilterChip(
                            selected = ajustes.modo == m,
                            onClick = { alCambiar(ajustes.copy(modo = m)) },
                            label = { Text(m.titulo) },
                        )
                    }
                }
                Text(
                    if (ajustes.modo == Modo.OJO) "Enrojecimiento, pterigión y catarata: mira de frente con el ojo bien abierto."
                    else "Palidez del párpado por dentro (signo de anemia): jala tu párpado de abajo y mira hacia arriba. " +
                        "Mejor con luz blanca de día.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            Seccion("¿Qué ojo vas a escanear?") {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (o in Ojo.entries) {
                        FilterChip(
                            selected = ajustes.ojo == o,
                            onClick = { alCambiar(ajustes.copy(ojo = o)) },
                            label = { Text(o.nombre.replaceFirstChar { it.uppercase() }) },
                        )
                    }
                }
                Text(
                    "Es tu ojo, no el lado de la pantalla.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            Seccion("Cámara") {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(!ajustes.trasera, { alCambiar(ajustes.copy(trasera = false)) }, { Text("Frontal") })
                    FilterChip(ajustes.trasera, { alCambiar(ajustes.copy(trasera = true)) }, { Text("Trasera (mejor calidad)") })
                }
                if (ajustes.trasera) {
                    Text(
                        "Con la trasera no ves la pantalla: deja la voz encendida o pide ayuda a otra persona.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Opcion("Indicaciones por voz", ajustes.voz) { alCambiar(ajustes.copy(voz = it)) }
            }

            Seccion("Servidor") {
                OutlinedTextField(
                    value = ajustes.servidor,
                    onValueChange = { alCambiar(ajustes.copy(servidor = it.trim())) },
                    label = { Text("Dirección del servidor") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedButton(onClick = {
                    prueba = "Probando…"
                    alcance.launch {
                        val api = OjoApi(ajustes.servidor)
                        prueba = runCatching { api.salud() }.fold(
                            onSuccess = { s ->
                                fun estado(k: String) = if (s.models[k]?.available == true) "listo" else "no disponible"
                                when (s.pcConectada) {
                                    false -> "Servidor en línea, pero la computadora con la IA está apagada. " +
                                        "Las fotos guardadas le llegarán cuando se conecte."
                                    else -> "Conectado (v${s.version}). Análisis del ojo: ${estado("eye_analysis")}. " +
                                        "Anemia: ${estado("anemia")}"
                                }
                            },
                            onFailure = { explicar(it) },
                        )
                        api.cerrar()
                    }
                }) { Text("Probar conexión") }
                prueba?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                Opcion("Guardar mis fotos para entrenar el modelo", ajustes.guardar) {
                    alCambiar(ajustes.copy(guardar = it))
                }
                if (ajustes.guardar) {
                    Text(
                        "Al activarlo aceptas que el recorte de tu ojo se guarde en el servidor del proyecto " +
                            "para entrenar el modelo. Puedes desactivarlo cuando quieras.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            Button(onClick = alEscanear, modifier = Modifier.fillMaxWidth().height(56.dp)) {
                Text("Escanear", style = MaterialTheme.typography.titleMedium)
            }
            Text(
                "Herramienta de investigación. No es un dispositivo médico ni sustituye la consulta con un oftalmólogo.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun Seccion(titulo: String, contenido: @Composable () -> Unit) {
    Card(colors = CardDefaults.cardColors(), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(titulo, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            contenido()
        }
    }
}

@Composable
private fun Opcion(texto: String, activo: Boolean, alCambiar: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(texto, Modifier.weight(1f))
        Switch(activo, alCambiar)
    }
}
