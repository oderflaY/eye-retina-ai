package com.temrarios.ojoscan.platform

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import com.temrarios.ojoscan.guia.Modo
import com.temrarios.ojoscan.guia.Ojo
import java.io.ByteArrayOutputStream
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Revisa la foto en alta resolución y recorta el ojo.
 *
 * La guía en vivo usa cuadros pequeños; aquí se vuelve a buscar el ojo sobre la foto final (detector
 * de ojo en dos pasadas: primero el centro de la foto, luego un cuadro ajustado al ojo) y se
 * verifica que esté abierto y que salga grande. Si algo falla, se pide repetir en vez de mandar
 * una foto mala al sistema.
 *
 * Recorte cuadrado de máximo 512 px:
 * - [Modo.OJO]: 1.5 veces el ancho del ojo, centrado (el ojo llena la foto, como en el entrenamiento).
 * - [Modo.ANEMIA]: 1.7 veces y corrido hacia abajo, para que entre la conjuntiva del párpado
 *   inferior jalado y el blanco del ojo (referencia de color).
 */
object Recortador {
    private const val LADO_MAX = 3000      // la foto se reduce a esto antes de girarla (memoria)
    private const val RECORTE_MAX = 512
    private const val APERTURA_MIN = 0.14f   // en la foto final solo se rechaza un parpadeo claro
    private const val OJO_MIN_PX = 110f

    /** [foto] en la orientación del sensor; [rotacion] grados para dejarla derecha. */
    fun procesar(context: Context, foto: Bitmap, rotacion: Int, ojo: Ojo, modo: Modo): Captura {
        val escala = min(1f, LADO_MAX.toFloat() / max(foto.width, foto.height))
        val derecha = enderezar(foto, rotacion, escala)
        try {
            val d = DetectorOjo.de(context).detectarFino(derecha, cuadradoCentral(derecha.width, derecha.height))
            if (!d.hayOjo) return Captura.Repetir("No se encontró tu ojo en la foto, intentemos otra vez")
            if (d.ancho < OJO_MIN_PX) return Captura.Repetir("El ojo salió muy pequeño, acércate un poco más")
            val apertura = ClasificadorApertura.de(context).apertura(derecha, d)
            if (modo == Modo.OJO && apertura < APERTURA_MIN) {
                return Captura.Repetir("Parpadeaste justo en la foto. Abre bien el ojo y no te muevas")
            }
            return Captura.Exito(listOf(recortar(derecha, d, apertura, ojo, modo)))
        } catch (e: Exception) {
            return Captura.Repetir("No se pudo revisar la foto, intentemos otra vez")
        } finally {
            derecha.recycle()
        }
    }

    private fun recortar(img: Bitmap, d: DetectorOjo.Deteccion, apertura: Float, ojo: Ojo, modo: Modo): Recorte {
        val (factor, bajar) = if (modo == Modo.ANEMIA) 1.7f to 0.30f else 1.5f to 0f
        // Recorte cuadrado; si se sale de la foto, se recorre hacia adentro.
        var ladoPx = (max(d.ancho, d.caja.height()) * factor).roundToInt()
        ladoPx = min(ladoPx, min(img.width, img.height))
        val cx = d.centro.x
        val cy = d.centro.y + bajar * d.ancho
        val x0 = (cx - ladoPx / 2f).roundToInt().coerceIn(0, img.width - ladoPx)
        val y0 = (cy - ladoPx / 2f).roundToInt().coerceIn(0, img.height - ladoPx)
        val escala = min(1f, RECORTE_MAX.toFloat() / ladoPx)
        val final = (ladoPx * escala).roundToInt()
        val recorte = Bitmap.createScaledBitmap(Bitmap.createBitmap(img, x0, y0, ladoPx, ladoPx), final, final, true)

        fun rel(p: PointF) = listOf((p.x - x0) * escala, (p.y - y0) * escala)
        val nombres = listOf("inner_corner", "outer_corner", "upper_lid", "lower_lid", "pupil", "iris")
        val puntos = nombres.zip(d.puntos).associate { (n, p) -> n to rel(p) }
        val caja = listOf((d.caja.left - x0) * escala, (d.caja.top - y0) * escala,
            d.caja.width() * escala, d.caja.height() * escala)

        val bytes = ByteArrayOutputStream().use {
            recorte.compress(Bitmap.CompressFormat.JPEG, 92, it)
            it.toByteArray()
        }
        recorte.recycle()
        return Recorte(ojo, bytes, final, final, caja, puntos, apertura)
    }
}
