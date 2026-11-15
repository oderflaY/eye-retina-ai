package com.temrarios.ojoscan.historial

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import kotlinx.coroutines.flow.MutableStateFlow

/** En iPhone el historial vendría de HealthKit (pendiente). */
@Composable
actual fun rememberFuenteHistorial(): FuenteHistorial = remember {
    object : FuenteHistorial {
        override val estado = MutableStateFlow<EstadoHistorial>(EstadoHistorial.NoDisponible)
        override fun conectar(dias: Int) {}
        override fun leerSiHayPermiso(dias: Int) {}
        override fun abrirAjustes() {}
    }
}
