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

/** Archivos de texto empaquetados con la app (p. ej. la base de primeros auxilios). Leer fuera del hilo principal. */
interface LectorAssets {
    fun leer(nombre: String): String?
}

@Composable
expect fun rememberLectorAssets(): LectorAssets

/** Desfase de la zona horaria local respecto a UTC en esa hora (ms), para agrupar el diario por día local. */
expect fun desfaseHorarioMs(horaMs: Long): Long

/** Notificaciones del teléfono: alertas del historial y recordatorio diario de revisiones. */
interface Notificador {
    fun avisar(id: Int, titulo: String, texto: String, urgente: Boolean)
    /** Revisión diaria (≈10:00) aunque la app esté cerrada: avisa qué revisiones tocan. */
    fun programarRecordatorios()
}

@androidx.compose.runtime.Composable
expect fun rememberNotificador(): Notificador

/** Anemia por la conjuntiva en el teléfono (sin servidor). null si la plataforma no lo tiene. */
interface AnemiaLocal {
    suspend fun analizar(jpeg: ByteArray): com.temrarios.ojoscan.api.AnalisisAnemia?
}

@androidx.compose.runtime.Composable
expect fun rememberAnemiaLocal(): AnemiaLocal?

/** Compartir el resumen en PDF y guardar/restaurar el respaldo cifrado (el usuario elige dónde). */
interface Compartidor {
    val disponible: Boolean
    fun compartirPdf(nombre: String, lineas: List<com.temrarios.ojoscan.astro.LineaReporte>)
    /** Cifra [json] con [clave] y abre «Guardar como…». [listo] recibe el error o null. */
    fun guardarRespaldo(nombre: String, json: String, clave: String, listo: (String?) -> Unit)
    /** Abre «Elegir archivo…», descifra y entrega el JSON (o el error). */
    fun abrirRespaldo(clave: String, listo: (json: String?, error: String?) -> Unit)
}

@androidx.compose.runtime.Composable
expect fun rememberCompartidor(): Compartidor
