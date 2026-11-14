package com.temrarios.ojoscan.platform

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Detector de ojo en el teléfono (MobileNetV2 entrenado con ojos de cerca, `ojo_detector.tflite`).
 *
 * Funciona como un detector de caras, pero del ojo: recibe un cuadrado de la imagen y dice si hay
 * un ojo, su caja y 6 puntos (esquina interna, esquina externa, párpado superior, párpado
 * inferior, pupila y borde del iris). No necesita ver la cara completa.
 *
 * El intérprete no admite dos hilos a la vez: todas las llamadas pasan por [detectar], que está
 * sincronizado (la guía en vivo y la foto final comparten la misma instancia).
 */
class DetectorOjo private constructor(context: Context) {
    companion object {
        const val LADO = 224
        const val UMBRAL = 0.6f
        private const val MODELO = "ojo_detector.tflite"
        @Volatile private var unico: DetectorOjo? = null

        fun de(context: Context): DetectorOjo =
            unico ?: synchronized(this) { unico ?: DetectorOjo(context.applicationContext).also { unico = it } }
    }

    /** Resultado en coordenadas del bitmap de entrada. */
    class Deteccion(
        val confianza: Float,
        val caja: RectF,
        /** Esquina izquierda y derecha (en la imagen), párpado superior, párpado inferior, pupila, iris. */
        val puntos: List<PointF>,
    ) {
        val hayOjo get() = confianza >= UMBRAL
        val interna get() = puntos[0]
        val externa get() = puntos[1]
        /** Ancho del ojo de esquina a esquina (px). */
        val ancho get() = max(hypot(externa.x - interna.x, externa.y - interna.y), caja.width())
        val centro get() = PointF(caja.centerX(), caja.centerY())
        // La apertura NO se saca de estos puntos (no la siguen): ver ClasificadorApertura.
    }

    private val interprete: Interpreter
    private val entrada = ByteBuffer.allocateDirect(LADO * LADO * 3 * 4).order(ByteOrder.nativeOrder())
    private val pixeles = IntArray(LADO * LADO)
    private val tablero = Bitmap.createBitmap(LADO, LADO, Bitmap.Config.ARGB_8888)
    private val lienzo = Canvas(tablero)
    private val pincel = Paint(Paint.FILTER_BITMAP_FLAG)
    private var iCaja = 0
    private var iPuntos = 1
    private var iProb = 2

    init {
        val fd = context.assets.openFd(MODELO)
        val modelo: MappedByteBuffer = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interprete = Interpreter(modelo, Interpreter.Options().setNumThreads(4))
        // Las salidas vienen de un diccionario: se identifican por su tamaño (4, 12 y 1).
        for (i in 0 until interprete.outputTensorCount) {
            when (interprete.getOutputTensor(i).shape().last()) {
                4 -> iCaja = i
                12 -> iPuntos = i
                1 -> iProb = i
            }
        }
    }

    /**
     * Busca un ojo dentro de [region] (cuadrado) de [bmp]. Las coordenadas devueltas son del bitmap.
     * Las partes de la región que salen del bitmap quedan negras.
     */
    @Synchronized
    fun detectar(bmp: Bitmap, region: Rect): Deteccion {
        lienzo.drawColor(android.graphics.Color.BLACK)
        lienzo.drawBitmap(bmp, region, Rect(0, 0, LADO, LADO), pincel)
        tablero.getPixels(pixeles, 0, LADO, 0, 0, LADO, LADO)
        entrada.rewind()
        for (p in pixeles) {
            entrada.putFloat(((p shr 16) and 0xFF) / 255f)
            entrada.putFloat(((p shr 8) and 0xFF) / 255f)
            entrada.putFloat((p and 0xFF) / 255f)
        }
        entrada.rewind()
        val caja = Array(1) { FloatArray(4) }
        val puntos = Array(1) { FloatArray(12) }
        val prob = Array(1) { FloatArray(1) }
        interprete.runForMultipleInputsOutputs(arrayOf(entrada), mapOf(iCaja to caja, iPuntos to puntos, iProb to prob))

        val s = region.width().toFloat()
        val sy = region.height().toFloat()
        fun x(v: Float) = region.left + v * s
        fun y(v: Float) = region.top + v * sy
        val c = caja[0]
        val p = puntos[0]
        return Deteccion(
            confianza = prob[0][0],
            caja = RectF(x(c[0]), y(c[1]), x(c[0] + c[2]), y(c[1] + c[3])),
            puntos = List(6) { PointF(x(p[2 * it]), y(p[2 * it + 1])) },
        )
    }

    /**
     * Dos pasadas: primero en [region], luego en un cuadrado ajustado al ojo encontrado (más
     * resolución sobre el ojo → puntos más precisos). Para la foto final.
     */
    fun detectarFino(bmp: Bitmap, region: Rect): Deteccion {
        val d = detectar(bmp, region)
        if (!d.hayOjo) return d
        val lado = (d.ancho * 2.2f).roundToInt().coerceAtLeast(LADO)
        val cx = d.centro.x.roundToInt()
        val cy = d.centro.y.roundToInt()
        val r2 = Rect(cx - lado / 2, cy - lado / 2, cx + lado / 2, cy + lado / 2)
        val d2 = detectar(bmp, r2)
        return if (d2.hayOjo) d2 else d
    }
}

/** Cuadrado centrado del tamaño del lado corto del bitmap. */
fun cuadradoCentral(ancho: Int, alto: Int): Rect {
    val l = min(ancho, alto)
    val x = (ancho - l) / 2
    val y = (alto - l) / 2
    return Rect(x, y, x + l, y + l)
}
