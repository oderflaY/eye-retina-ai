package com.temrarios.ojoscan.ui

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.material3.Text
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.temrarios.ojoscan.pulso.EstadoPulso
import com.temrarios.ojoscan.pulso.MonitorPulso
import com.temrarios.ojoscan.pulso.resumir

/** Tarjeta "Pulsera" en Tus datos: conecta la Smart Band y muestra el pulso latido a latido. */
@Composable
fun PanelPulsera(pulso: MonitorPulso) {
    val estado by pulso.estado.collectAsState()
    val muestras by pulso.muestras.collectAsState()
    PanelVidrio(Modifier.fillMaxWidth()) {
        TituloHud("05", "Pulsera (en vivo)")
        Text("Tu pulso en tiempo real desde la pulsera por Bluetooth, sin la nube de Xiaomi. Se mide durante todo el chequeo.",
            color = Colores.textoSuave, fontSize = 12.sp, lineHeight = 16.sp)
        when (val e = estado) {
            is EstadoPulso.Midiendo -> {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                    Corazon(e.lpm > 0)
                    Column(Modifier.weight(1f)) {
                        Text(if (e.lpm > 0) "${e.lpm} lpm" else "Esperando latido…", color = Colores.texto, fontSize = 28.sp, fontWeight = FontWeight.Bold)
                        Text(e.nombre + (if (e.contacto == false) " · ¡no toca la piel!" else ""), color = if (e.contacto == false) Colores.atencion else Colores.textoSuave, fontSize = 12.sp)
                    }
                }
                val r = resumir(muestras)
                Text(
                    if (r == null) "Juntando lecturas (${muestras.size})… quédate quieto." else
                        "Reposo ${r.reposo} · mín ${r.minimo} · máx ${r.maximo} lpm · ${r.segundos} s" + (r.rmssd?.let { " · VFC $it ms" } ?: ""),
                    style = Hud.dato.copy(fontSize = 11.sp, color = Colores.cian),
                )
                BotonContorno("Desconectar", pulso::detener, Modifier.fillMaxWidth())
            }
            is EstadoPulso.Buscando -> Text(e.detalle, color = Colores.cian, fontSize = 14.sp)
            is EstadoPulso.Conectando -> Text("Conectando con ${e.nombre}…", color = Colores.cian, fontSize = 14.sp)
            else -> {
                val aviso = when (e) {
                    EstadoPulso.SinPermiso -> "Falta el permiso de Bluetooth (\"Dispositivos cercanos\")."
                    EstadoPulso.BluetoothApagado -> "Enciende el Bluetooth."
                    is EstadoPulso.SinTransmision -> "Encontré ${e.nombre}, pero no está transmitiendo el pulso."
                    is EstadoPulso.Error -> e.mensaje
                    else -> null
                }
                aviso?.let { Text(it, color = Colores.atencion, fontSize = 13.sp, lineHeight = 17.sp) }
                if (e is EstadoPulso.SinTransmision || e is EstadoPulso.Error) Text(
                    "En la pulsera: Ajustes → Frecuencia cardíaca → \"Transmitir frecuencia cardíaca\" (o en Mi Fitness: tu pulsera → " +
                        "Frecuencia cardíaca → Compartir frecuencia cardíaca). Déjala activa y vuelve a conectar.",
                    color = Colores.textoSuave, fontSize = 12.sp, lineHeight = 16.sp,
                )
                BotonNeon("Conectar pulsera", pulso::iniciar, Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
private fun Corazon(latiendo: Boolean) {
    val t = rememberInfiniteTransition()
    val s by t.animateFloat(1f, if (latiendo) 1.18f else 1f, infiniteRepeatable(tween(420), RepeatMode.Reverse))
    Icon(Icons.Filled.Favorite, null, tint = Colores.mal, modifier = Modifier.size(44.dp).scale(s))
}

/** Pastilla con el pulso en vivo encima de la cámara (solo si la pulsera está midiendo). */
@Composable
fun PulsoEnVivo(pulso: MonitorPulso, modifier: Modifier = Modifier) {
    val e by pulso.estado.collectAsState()
    val m = e as? EstadoPulso.Midiendo ?: return
    if (m.lpm <= 0) return
    Row(
        modifier.clip(RoundedCornerShape(14.dp)).background(Colores.velo)
            .border(1.dp, Colores.mal.copy(alpha = 0.6f), RoundedCornerShape(14.dp)).padding(horizontal = 10.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(Icons.Filled.Favorite, null, tint = Colores.mal, modifier = Modifier.size(14.dp))
        Spacer(Modifier.size(6.dp))
        Text("${m.lpm} lpm", color = Colores.texto, fontSize = 13.sp, fontWeight = FontWeight.Bold)
    }
}
