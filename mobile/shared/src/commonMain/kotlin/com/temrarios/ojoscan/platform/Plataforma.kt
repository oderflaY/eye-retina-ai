package com.temrarios.ojoscan.platform

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import com.temrarios.ojoscan.guia.Lectura
import com.temrarios.ojoscan.guia.Modo
import com.temrarios.ojoscan.guia.Ojo

/** Un ojo recortado de la foto en alta resolución. */
class Recorte(
    /** IZQUIERDO o DERECHO (de la persona). */
    val ojo: Ojo,
    val jpeg: ByteArray,
    val ancho: Int,
    val alto: Int,
    /** Caja del ojo en el recorte: x, y, ancho, alto (px). */
    val caja: List<Float>?,
    /** inner_corner, outer_corner, upper_lid, lower_lid, pupil, iris → [x, y] en px del recorte. */
    val puntos: Map<String, List<Float>>,
    /** Distancia entre párpados / ancho del ojo medida en la foto final. */
    val apertura: Float?,
)

/** Resultado de pedir la foto: los recortes, o por qué hay que repetirla. */
sealed interface Captura {
    class Exito(val recortes: List<Recorte>) : Captura
    class Repetir(val motivo: String) : Captura
}

/**
 * Cámara con análisis en vivo.
 * - [alLeer]: una [Lectura] por cuadro analizado (≈ 10-15 por segundo).
 * - [pedirFoto]: cada vez que aumenta, se toma una foto en alta resolución, se revisa de nuevo
 *   (ojo abierto, tamaño) y se recortan los ojos; el resultado llega por [alCapturar].
 * - [alError]: no hay cámara o no se dio permiso.
 */
@Composable
expect fun CamaraOjo(
    modifier: Modifier,
    ojo: Ojo,
    modo: Modo,
    trasera: Boolean,
    pedirFoto: Int,
    alLeer: (Lectura) -> Unit,
    alCapturar: (Captura) -> Unit,
    alError: (String) -> Unit,
)

/** Lector en voz alta (español). */
interface Voz {
    fun decir(texto: String)
    fun callar()
}

@Composable
expect fun rememberVoz(): Voz

/** JPEG/PNG → imagen para Compose. */
expect fun decodificarImagen(bytes: ByteArray): ImageBitmap?

/** Guardado local pequeño (texto por clave) que sobrevive al cerrar la app. Nada sale del teléfono. */
interface Almacen {
    fun leer(clave: String): String?
    fun guardar(clave: String, valor: String)
}

@Composable
expect fun rememberAlmacen(): Almacen
