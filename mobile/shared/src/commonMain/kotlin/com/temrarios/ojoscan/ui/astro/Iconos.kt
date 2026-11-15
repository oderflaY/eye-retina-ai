package com.temrarios.ojoscan.ui.astro

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.addPathNodes
import androidx.compose.ui.unit.dp

/** Iconos de contorno (24×24, trazo 1.6) definidos como trayectorias SVG. Se tiñen con `tint`. */
object Ico {
    private fun svg(nombre: String, vararg trazos: String): ImageVector =
        ImageVector.Builder(nombre, 24.dp, 24.dp, 24f, 24f).apply {
            for (d in trazos) addPath(addPathNodes(d), stroke = SolidColor(Color.White), strokeLineWidth = 1.6f,
                strokeLineCap = StrokeCap.Round, strokeLineJoin = StrokeJoin.Round)
        }.build()

    val ppg = svg("ppg",
        "M19.5 12.6 12 20l-7.5-7.4A4.6 4.6 0 0 1 12 6.6a4.6 4.6 0 0 1 7.5 6Z",
        "M3.5 12h4l1.5-3 3 6 1.5-3h7")
    val pvt = svg("pvt", "M12 21a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z", "M12 9v4l2.5 2", "M10 2h4", "M12 2v3", "M19 6l1.5-1.5")
    val imu = svg("imu", "M3 3v18h18", "M7 15l3-6 3 4 3-8 3 6", "M3 21l4-4")
    val manual = svg("manual", "M4 19.5V5a2 2 0 0 1 2-2h14v16H6a2 2 0 0 0-2 2Z", "M4 19.5A2 2 0 0 0 6 21.5h14", "M12 7v6", "M9 10h6")
    val voz = svg("voz", "M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z", "M19 11a7 7 0 0 1-14 0", "M12 18v3", "M9 21h6")
    val sistema = svg("sistema", "M7 7h10v10H7z", "M10 10h4v4h-4z", "M9 3v4", "M15 3v4", "M9 17v4", "M15 17v4", "M3 9h4", "M3 15h4", "M17 9h4", "M17 15h4")
    val farmacia = svg("farmacia", "M10.5 20.5a5 5 0 0 1-7-7l7-7a5 5 0 0 1 7 7Z", "M7 10l7 7", "M17 3.5l1 2 2 1-2 1-1 2-1-2-2-1 2-1Z")
    val sos = svg("sos", "M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z", "M12 9v4", "M12 17h.01")
    val ia = svg("ia", "M5 8h14a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2Z", "M12 4v4", "M12 4h.01",
        "M9 13v1", "M15 13v1", "M1 13v2", "M23 13v2")
    val abajo = svg("abajo", "M6 9l6 6 6-6")
    val cerrar = svg("cerrar", "M18 6 6 18", "M6 6l12 12")
    val enviar = svg("enviar", "M22 2 11 13", "M22 2l-7 20-4-9-9-4Z")
    val satelite = svg("satelite", "M13 7 9 3 5 7l4 4", "M17 11l4 4-4 4-4-4", "M8 12l4 4 6-6-4-4Z", "M16 8l3-3", "M9 21a6 6 0 0 0-6-6")
    val play = svg("play", "M6 4l14 8-14 8Z")
    val stop = svg("stop", "M6 6h12v12H6z")
    val basura = svg("basura", "M3 6h18", "M8 6V4h8v2", "M19 6l-1 14H6L5 6")
    val corazon = svg("corazon", "M19.5 12.6 12 20l-7.5-7.4A4.6 4.6 0 0 1 12 6.6a4.6 4.6 0 0 1 7.5 6Z")
    val ojo = svg("ojo", "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z", "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z")
    val gota = svg("gota", "M12 2.7s-6 6.6-6 11.3a6 6 0 0 0 12 0c0-4.7-6-11.3-6-11.3Z", "M9.5 14.5a2.5 2.5 0 0 0 2.5 2.5")
    val candado = svg("candado", "M5 11h14v10H5z", "M8 11V7a4 4 0 0 1 8 0v4")
}
