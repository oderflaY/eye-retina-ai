package com.temrarios.ojoscan.pulso

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import kotlinx.coroutines.flow.MutableStateFlow

/** En iPhone aún no se lee la pulsera (CoreBluetooth pendiente): el chequeo sigue sin pulso. */
@Composable
actual fun rememberMonitorPulso(): MonitorPulso = remember {
    object : MonitorPulso {
        override val estado = MutableStateFlow<EstadoPulso>(EstadoPulso.Inactivo)
        override val muestras = MutableStateFlow<List<Muestra>>(emptyList())
        override fun iniciar() { estado.value = EstadoPulso.Error("La pulsera aún no está disponible en iPhone.") }
        override fun detener() { estado.value = EstadoPulso.Inactivo }
        override fun reiniciarSesion() { muestras.value = emptyList() }
    }
}
