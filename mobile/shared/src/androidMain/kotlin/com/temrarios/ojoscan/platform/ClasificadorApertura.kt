package com.temrarios.ojoscan.platform

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * ¿Qué tan abierto está el ojo? Modelo aparte (`ojo_apertura.tflite`, MobileNetV2 chico) porque los
 * puntos de párpado del detector no siguen la apertura real.
 *
 * Recibe el ojo que encontró [DetectorOjo] y devuelve la apertura (párpado a párpado / esquina a
 * esquina: abierto ≈0.3, cerrado ≈0.1). Si el modelo está seguro de que el ojo está cerrado, la
 * apertura se baja a [CERRADO] para que la guía pida "abre más el ojo".
 */
class ClasificadorApertura private constructor(context: Context) {
    companion object {
        const val LADO = 128
        /** Lado del recorte / ancho del ojo (a la mitad del rango de entrenamiento 1.25–3.6). */
        private const val ESCALA = 2.0f
        private const val PROB_CERRADO = 0.5f
        private const val CERRADO = 0.08f
        private const val MODELO = "ojo_apertura.tflite"
        @Volatile private var unico: ClasificadorApertura? = null

        fun de(context: Context): ClasificadorApertura =
            unico ?: synchronized(this) { unico ?: ClasificadorApertura(context.applicationContext).also { unico = it } }
    }

    private val interprete: Interpreter
    private val entrada = ByteBuffer.allocateDirect(LADO * LADO * 3 * 4).order(ByteOrder.nativeOrder())
    private val pixeles = IntArray(LADO * LADO)
    private val tablero = Bitmap.createBitmap(LADO, LADO, Bitmap.Config.ARGB_8888)
    private val lienzo = Canvas(tablero)
    private val pincel = Paint(Paint.FILTER_BITMAP_FLAG)
    private val salida = Array(1) { FloatArray(2) }   // [apertura, prob. cerrado]

    init {
        val fd = context.assets.openFd(MODELO)
        val modelo = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interprete = Interpreter(modelo, Interpreter.Options().setNumThreads(2))
    }

    /** Apertura del ojo [d] dentro de [bmp] (mismas coordenadas que la detección). */
    @Synchronized
    fun apertura(bmp: Bitmap, d: DetectorOjo.Deteccion): Float {
        val lado = min((d.ancho * ESCALA).roundToInt(), min(bmp.width, bmp.height)).coerceAtLeast(8)
        val x0 = (d.centro.x - lado / 2f).roundToInt().coerceIn(0, bmp.width - lado)
        val y0 = (d.centro.y - lado / 2f).roundToInt().coerceIn(0, bmp.height - lado)
        lienzo.drawBitmap(bmp, Rect(x0, y0, x0 + lado, y0 + lado), Rect(0, 0, LADO, LADO), pincel)
        tablero.getPixels(pixeles, 0, LADO, 0, 0, LADO, LADO)
        entrada.rewind()
        for (p in pixeles) {
            entrada.putFloat(((p shr 16) and 0xFF) / 255f)
            entrada.putFloat(((p shr 8) and 0xFF) / 255f)
            entrada.putFloat((p and 0xFF) / 255f)
        }
        entrada.rewind()
        interprete.run(entrada, salida)
        val (ap, cerrado) = salida[0]
        return if (cerrado >= PROB_CERRADO) min(ap, CERRADO) else ap
    }
}
