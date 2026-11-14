package com.temrarios.ojoscan.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

object Colores {
    val turquesa = Color(0xFF0E8C8C)
    val turquesaClaro = Color(0xFF5FD3CF)
    val bien = Color(0xFF2ECC71)
    val atencion = Color(0xFFFFB020)
    val mal = Color(0xFFFF5A5F)
    val velo = Color(0xB3000000)
}

@Composable
fun TemaOjo(contenido: @Composable () -> Unit) {
    val oscuro = isSystemInDarkTheme()
    MaterialTheme(
        colorScheme = if (oscuro) {
            darkColorScheme(primary = Colores.turquesaClaro, secondary = Colores.turquesa)
        } else {
            lightColorScheme(primary = Colores.turquesa, secondary = Colores.turquesaClaro)
        },
        content = contenido,
    )
}
