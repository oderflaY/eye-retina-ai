package com.temrarios.ojoscan.platform

import androidx.compose.runtime.Composable

/**
 * Al abrir la app pide de una vez todo lo que hace falta: cámara, Bluetooth (pulsera) y Health Connect
 * (historial de Mi Fitness). Solo pregunta lo que falte. [alTerminar] se llama al final, con o sin permisos.
 */
@Composable
expect fun PermisosAlIniciar(alTerminar: () -> Unit)
