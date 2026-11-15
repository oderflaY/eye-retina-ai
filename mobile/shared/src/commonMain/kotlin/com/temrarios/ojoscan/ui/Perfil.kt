package com.temrarios.ojoscan.ui

import androidx.compose.animation.AnimatedVisibility
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.temrarios.ojoscan.api.DatosPerfil
import com.temrarios.ojoscan.pulso.MonitorPulso
import com.temrarios.ojoscan.historial.FuenteHistorial

/** Campo de texto que guarda el valor tal como se escribe y se convierte al final. */
private class Campo(inicial: Number?) {
    var texto by mutableStateOf(inicial?.let { if (it.toDouble() % 1.0 == 0.0) it.toLong().toString() else it.toString() } ?: "")
    val numero: Double? get() = texto.replace(',', '.').trim().toDoubleOrNull()?.takeIf { it > 0 }
}

/**
 * "Tus datos": lo que la cámara no ve. Con esto el servidor calcula IMC, grasa corporal y presión,
 * y el riesgo del corazón si hay dolor de pecho (modelos entrenados con ~/Deep_Learning).
 * Se guarda en el teléfono; el nombre nunca se envía.
 */
@Composable
fun PantallaPerfil(inicial: DatosPerfil, pulso: MonitorPulso, historial: FuenteHistorial, numeroPaso: String?, alGuardar: (DatosPerfil) -> Unit, alSaltar: () -> Unit, alCancelar: () -> Unit) {
    var nombre by remember { mutableStateOf(inicial.nombre) }
    var sexo by remember { mutableStateOf(inicial.sexo) }
    val edad = remember { Campo(inicial.edad) }
    val peso = remember { Campo(inicial.peso) }
    val estatura = remember { Campo(inicial.estatura) }
    val cuello = remember { Campo(inicial.cuello) }
    val abdomen = remember { Campo(inicial.abdomen) }
    val cadera = remember { Campo(inicial.cadera) }
    val sis = remember { Campo(inicial.sistolica) }
    val dia = remember { Campo(inicial.diastolica) }
    val colesterol = remember { Campo(inicial.colesterol) }
    val glucosa = remember { Campo(inicial.glucosa) }
    val fcMax = remember { Campo(inicial.fcMaxima) }
    var dolor by remember { mutableStateOf(inicial.dolorPecho) }
    var dolorEjercicio by remember { mutableStateOf(inicial.dolorEjercicio) }
    var labAbierto by remember { mutableStateOf(inicial.colesterol != null || inicial.glucosa != null) }

    fun armar() = DatosPerfil(
        nombre = nombre.trim(), edad = edad.numero?.toInt(), sexo = sexo,
        peso = peso.numero, estatura = estatura.numero, cuello = cuello.numero, abdomen = abdomen.numero, cadera = cadera.numero,
        sistolica = sis.numero, diastolica = dia.numero, dolorPecho = dolor, dolorEjercicio = dolorEjercicio && dolor != "ninguno",
        colesterol = colesterol.numero, glucosa = glucosa.numero, fcMaxima = fcMax.numero,
    )

    FondoEspacial {
        Column(
            Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text((numeroPaso?.let { "CHEQUEO COMPLETO // $it" } ?: "PERFIL MÉDICO"), style = Hud.etiqueta.copy(color = Colores.textoSuave, fontSize = 10.sp))
            TextoDegradado("Tus datos", 34)
            Text("Lo que la cámara no ve. Con esto se arma tu perfil médico. Solo edad y sexo son necesarios; " +
                "mientras más llenes, más completo. Se guarda en tu teléfono.", color = Colores.textoSuave, fontSize = 13.sp, lineHeight = 17.sp)

            PanelVidrio(Modifier.fillMaxWidth()) {
                TituloHud("01", "Persona")
                CampoTexto("Nombre (opcional, no se envía)", nombre, { nombre = it }, numerico = false)
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                    CampoNum("Edad", edad, "años", Modifier.weight(1f))
                    Segmento("Hombre", sexo == "M", { sexo = "M" }, Modifier.weight(1f))
                    Segmento("Mujer", sexo == "F", { sexo = "F" }, Modifier.weight(1f))
                }
            }

            PanelVidrio(Modifier.fillMaxWidth()) {
                TituloHud("02", "Cuerpo")
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    CampoNum("Peso", peso, "kg", Modifier.weight(1f))
                    CampoNum("Estatura", estatura, "cm", Modifier.weight(1f))
                }
                Text("Con cinta métrica, sin apretar (para la grasa corporal):", color = Colores.textoSuave, fontSize = 12.sp)
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    CampoNum("Cuello", cuello, "cm", Modifier.weight(1f))
                    CampoNum("Cintura", abdomen, "cm", Modifier.weight(1f))
                    CampoNum("Cadera", cadera, "cm", Modifier.weight(1f))
                }
                Text("Cuello: debajo de la manzana de Adán. Cintura: a la altura del ombligo. Cadera: en la parte más ancha.",
                    color = Colores.textoSuave.copy(alpha = 0.8f), fontSize = 11.sp, lineHeight = 14.sp)
            }

            PanelVidrio(Modifier.fillMaxWidth()) {
                TituloHud("03", "Presión arterial")
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    CampoNum("Alta (sistólica)", sis, "mmHg", Modifier.weight(1f))
                    CampoNum("Baja (diastólica)", dia, "mmHg", Modifier.weight(1f))
                }
                Text("Sentado, después de 5 minutos en reposo.", color = Colores.textoSuave.copy(alpha = 0.8f), fontSize = 11.sp)
            }

            PanelVidrio(Modifier.fillMaxWidth()) {
                TituloHud("04", "Dolor de pecho")
                for ((clave, titulo, detalle) in listOf(
                    Triple("ninguno", "No tengo", null),
                    Triple("tipico", "Opresivo", "Presión o peso detrás del pecho, aparece con el esfuerzo y se quita al descansar"),
                    Triple("atipico", "Parecido, pero no igual", "Tiene solo parte de lo anterior"),
                    Triple("no_anginoso", "Punzante o al moverme", "Como piquete, cambia al respirar, moverte o apretar la zona"),
                )) Segmento(titulo, dolor == clave, { dolor = clave }, Modifier.fillMaxWidth(), detalle = detalle)
                if (dolor != "ninguno") InterruptorNeon("Me da al hacer ejercicio o esfuerzo", dolorEjercicio, { dolorEjercicio = it })
                if (dolor == "tipico") Text("Si el dolor dura más de 15 minutos, te da en reposo o con falta de aire o sudor frío, llama al 911.",
                    color = Colores.mal, fontSize = 12.sp, lineHeight = 16.sp)
            }

            PanelVidrio(Modifier.fillMaxWidth()) {
                Row(Modifier.fillMaxWidth().clickable { labAbierto = !labAbierto }, verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.weight(1f)) { TituloHud("05", "Análisis de sangre (opcional)") }
                    Text(if (labAbierto) "  ▲" else "  ▼", color = Colores.cian, fontSize = 12.sp)
                }
                AnimatedVisibility(labAbierto) {
                    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                            CampoNum("Colesterol total", colesterol, "mg/dL", Modifier.weight(1f))
                            CampoNum("Glucosa ayunas", glucosa, "mg/dL", Modifier.weight(1f))
                        }
                        CampoNum("Pulso máximo en ejercicio (reloj o banda)", fcMax, "lpm", Modifier.fillMaxWidth())
                    }
                }
            }

            val listo = edad.numero != null && sexo != null
            BotonNeon(if (listo) "Guardar y continuar" else "Falta edad y sexo", { if (listo) alGuardar(armar()) }, Modifier.fillMaxWidth(), habilitado = listo)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                BotonContorno("Cancelar", alCancelar, Modifier.weight(1f))
                BotonContorno("Saltar", alSaltar, Modifier.weight(1f))
            }
            Text("Los modelos de grasa corporal y corazón se entrenaron con datos públicos pequeños. Es orientativo.",
                color = Colores.textoSuave.copy(alpha = 0.7f), fontSize = 10.sp, lineHeight = 13.sp,
                textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth())
            Spacer(Modifier.height(8.dp))
        }
    }
}

@Composable
private fun CampoNum(etiqueta: String, campo: Campo, unidad: String, modifier: Modifier) =
    CampoTexto(etiqueta, campo.texto, { campo.texto = it.filter { c -> c.isDigit() || c == '.' || c == ',' }.take(6) }, true, unidad, modifier)

@Composable
private fun CampoTexto(etiqueta: String, valor: String, alCambiar: (String) -> Unit, numerico: Boolean, unidad: String? = null, modifier: Modifier = Modifier.fillMaxWidth()) {
    OutlinedTextField(
        value = valor, onValueChange = alCambiar, singleLine = true,
        label = { Text(etiqueta, fontSize = 12.sp) },
        suffix = unidad?.let { { Text(it, fontSize = 11.sp, color = Colores.textoSuave) } },
        keyboardOptions = KeyboardOptions(keyboardType = if (numerico) KeyboardType.Decimal else KeyboardType.Text),
        textStyle = Hud.dato.copy(color = Colores.texto),
        shape = RoundedCornerShape(14.dp),
        colors = OutlinedTextFieldDefaults.colors(
            focusedBorderColor = Colores.cian, unfocusedBorderColor = Colores.cian.copy(alpha = 0.3f),
            focusedLabelColor = Colores.cian, cursorColor = Colores.cian,
        ),
        modifier = modifier,
    )
}
