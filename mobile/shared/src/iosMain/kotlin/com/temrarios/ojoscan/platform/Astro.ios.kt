package com.temrarios.ojoscan.platform

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color

/*
 * iPhone: pendiente (Keychain + CryptoKit para la bóveda, AVAudioRecorder, CoreMotion y AVCapture con linterna).
 * Mientras tanto los módulos muestran "no disponible en este dispositivo" en vez de fingir datos.
 */

@Composable
actual fun rememberBoveda(): Boveda = remember {
    object : Boveda {
        override val disponible = false
        override fun guardar(nombre: String, datos: ByteArray): ArchivoBoveda = error("Bóveda cifrada no disponible en iPhone todavía")
        override fun leer(nombre: String): ByteArray? = null
        override fun listar(prefijo: String) = emptyList<ArchivoBoveda>()
        override fun borrar(nombre: String) {}
    }
}

@Composable
actual fun rememberGrabadora(): Grabadora = remember {
    object : Grabadora {
        override val disponible = false
        override fun iniciar() = false
        override fun nivel() = 0f
        override fun detener(): ByteArray? = null
        override fun reproducir(audio: ByteArray, alTerminar: () -> Unit) = alTerminar()
        override fun pararReproduccion() {}
    }
}

@Composable
actual fun EscuchaAcelerometro(activo: Boolean, alMuestra: (x: Double, y: Double, z: Double) -> Unit) {}

@Composable
actual fun CamaraPpg(modifier: Modifier, alCuadro: (horaMs: Long, rojo: Double, verde: Double) -> Unit) {
    Box(modifier.background(Color.Black))
}

@Composable
actual fun rememberDispositivo(): Dispositivo = remember {
    object : Dispositivo {
        override fun bateria(): Int? = null
        override fun cargando() = false
        override fun libreMb(): Long? = null
        override fun modelo() = "iPhone"
        override fun sensores() = listOf("Sensores de iPhone (pendiente)" to false)
    }
}

@Composable
actual fun rememberLectorAssets(): LectorAssets = remember {
    // Pendiente: leer del bundle de iOS (NSBundle). Sin base, el diario usa solo el semáforo.
    object : LectorAssets { override fun leer(nombre: String): String? = null }
}

// Pendiente en iOS: NSTimeZone.localTimeZone. Mientras, el diario agrupa por día UTC.
actual fun desfaseHorarioMs(horaMs: Long): Long = 0L

@androidx.compose.runtime.Composable
actual fun rememberNotificador(): Notificador = androidx.compose.runtime.remember {
    object : Notificador {
        override fun avisar(id: Int, titulo: String, texto: String, urgente: Boolean) {}
        override fun programarRecordatorios() {}
    }
}

@androidx.compose.runtime.Composable
actual fun rememberAnemiaLocal(): AnemiaLocal? = null

@androidx.compose.runtime.Composable
actual fun rememberCompartidor(): Compartidor = androidx.compose.runtime.remember {
    object : Compartidor {
        override val disponible = false
        override fun compartirPdf(nombre: String, lineas: List<com.temrarios.ojoscan.astro.LineaReporte>) {}
        override fun guardarRespaldo(nombre: String, json: String, clave: String, listo: (String?) -> Unit) = listo("No disponible en iPhone todavía")
        override fun abrirRespaldo(clave: String, listo: (String?, String?) -> Unit) = listo(null, "No disponible en iPhone todavía")
    }
}
