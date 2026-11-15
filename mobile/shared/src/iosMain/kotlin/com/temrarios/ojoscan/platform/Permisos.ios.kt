package com.temrarios.ojoscan.platform

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect

/** En iPhone cada permiso se pide cuando se usa (cámara, Bluetooth). */
@Composable
actual fun PermisosAlIniciar(alTerminar: () -> Unit) {
    LaunchedEffect(Unit) { alTerminar() }
}
