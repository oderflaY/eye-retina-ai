package com.temrarios.ojoscan.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.temrarios.ojoscan.historial.EstadoHistorial
import com.temrarios.ojoscan.historial.FuenteHistorial
import com.temrarios.ojoscan.historial.paraServidor

private fun n(x: Double?, dec: Boolean = false) = x?.let { if (dec) ((it * 10).toInt() / 10.0).toString() else it.toInt().toString() } ?: "—"

/** Tarjeta "Historial de la pulsera": 30 días de Mi Fitness vía Health Connect. */
@Composable
fun PanelHistorial(fuente: FuenteHistorial) {
    val estado by fuente.estado.collectAsState()
    // Si ya diste permiso antes, se lee solo al abrir.
    LaunchedEffect(Unit) { if (estado == EstadoHistorial.Inactivo) fuente.conectar() }
    PanelVidrio(Modifier.fillMaxWidth()) {
        TituloHud("06", "Historial de la pulsera (30 días)")
        Text("Sueño, oxígeno, pasos y pulso que Mi Fitness guarda en Health Connect. Se resumen en tu teléfono.",
            color = Colores.textoSuave, fontSize = 12.sp, lineHeight = 16.sp)
        when (val e = estado) {
            is EstadoHistorial.Listo -> {
                val h = e.resumen.paraServidor()
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                    Dato("Pulso reposo", n(h.pulsoReposo), "lpm", Modifier.weight(1f))
                    Dato("Oxígeno", n(h.spo2Media, true), "%", Modifier.weight(1f))
                }
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                    Dato("Sueño", n(h.suenoHoras, true), "h/noche", Modifier.weight(1f))
                    Dato("Pasos", n(h.pasos), "al día", Modifier.weight(1f))
                }
                Text("${e.resumen.pulsoReposo.size} días de pulso · ${h.spo2N} lecturas de oxígeno · ${h.suenoNoches} noches · ${h.pasosDias} días de pasos",
                    style = Hud.dato.copy(fontSize = 10.sp, color = Colores.cian))
                BotonContorno("Volver a leer", { fuente.conectar() }, Modifier.fillMaxWidth())
            }
            EstadoHistorial.Leyendo -> Text("Leyendo tu historial…", color = Colores.cian, fontSize = 14.sp)
            EstadoHistorial.NoDisponible -> Text("Este teléfono no tiene Health Connect.", color = Colores.atencion, fontSize = 13.sp)
            else -> {
                val aviso = when (e) {
                    EstadoHistorial.SinPermiso -> "Falta permitir a OjoScan leer Health Connect."
                    EstadoHistorial.SinDatos -> "Health Connect está vacío: Mi Fitness todavía no le comparte tus datos."
                    is EstadoHistorial.Error -> e.mensaje
                    else -> null
                }
                aviso?.let { Text(it, color = Colores.atencion, fontSize = 13.sp, lineHeight = 17.sp) }
                if (e == EstadoHistorial.SinDatos) Text(
                    "En Mi Fitness: Perfil (Yo) → Health Connect / \"Apps de terceros\" → actívalo y permite todo. " +
                        "Espera a que sincronice y vuelve a leer.",
                    color = Colores.textoSuave, fontSize = 12.sp, lineHeight = 16.sp,
                )
                BotonNeon("Conectar historial", { fuente.conectar() }, Modifier.fillMaxWidth())
                if (e == EstadoHistorial.SinDatos || e == EstadoHistorial.SinPermiso)
                    BotonContorno("Abrir Health Connect", fuente::abrirAjustes, Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
private fun Dato(titulo: String, valor: String, unidad: String, modifier: Modifier) {
    Column(modifier) {
        Text(titulo.uppercase(), style = Hud.etiqueta.copy(fontSize = 9.sp, color = Colores.textoSuave))
        Row { Text(valor, color = Colores.texto, fontSize = 24.sp, fontWeight = FontWeight.Bold); Text(" $unidad", color = Colores.textoSuave, fontSize = 12.sp) }
    }
}
