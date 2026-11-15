package com.temrarios.ojoscan.platform

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

data class ArchivoBoveda(val nombre: String, val bytes: Long, val horaMs: Long, val huella: String)

/** Almacén cifrado local (AES-256-GCM, llave que nunca sale del chip seguro del teléfono). */
interface Boveda {
    val disponible: Boolean
    fun guardar(nombre: String, datos: ByteArray): ArchivoBoveda
    fun leer(nombre: String): ByteArray?
    fun listar(prefijo: String): List<ArchivoBoveda>
    fun borrar(nombre: String)
}

@Composable
expect fun rememberBoveda(): Boveda

/** Grabadora de voz: el audio crudo nunca queda en disco sin cifrar. */
interface Grabadora {
    val disponible: Boolean
    fun iniciar(): Boolean
    /** Nivel 0..1 del micrófono mientras graba. */
    fun nivel(): Float
    /** Detiene y devuelve el audio (m4a) para cifrarlo. */
    fun detener(): ByteArray?
    fun reproducir(audio: ByteArray, alTerminar: () -> Unit)
    fun pararReproduccion()
}

@Composable
expect fun rememberGrabadora(): Grabadora

/** Acelerómetro (m/s², incluye gravedad) mientras [activo]. */
@Composable
expect fun EscuchaAcelerometro(activo: Boolean, alMuestra: (x: Double, y: Double, z: Double) -> Unit)

/** Cámara trasera con flash encendido para PPG: entrega el promedio de rojo y verde del centro de cada cuadro. */
@Composable
expect fun CamaraPpg(modifier: Modifier, alCuadro: (horaMs: Long, rojo: Double, verde: Double) -> Unit)

interface Dispositivo {
    fun bateria(): Int?
    fun cargando(): Boolean
    fun libreMb(): Long?
    fun modelo(): String
    fun sensores(): List<Pair<String, Boolean>>
}

@Composable
expect fun rememberDispositivo(): Dispositivo
