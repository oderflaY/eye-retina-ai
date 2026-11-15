package com.temrarios.ojoscan.platform

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import com.temrarios.ojoscan.api.AnalisisAnemia
import com.temrarios.ojoscan.astro.ColorConjuntiva
import com.temrarios.ojoscan.astro.ModeloAnemiaMovil
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.roundToInt

/**
 * Segmentador MobileNetV2-UNet (`conjuntiva_movil.tflite`, 256 px, entrada RGB 0..255) + color + logística
 * (`anemia_movil.json`). Misma tubería que el servidor: foto completa en letterbox, prueba con espejo,
 * umbral, mayor componente, medidas de color sin balance de blancos.
 */
class AnemiaLocalAndroid private constructor(ctx: Context) : AnemiaLocal {
    companion object {
        @Volatile private var unico: AnemiaLocalAndroid? = null
        fun de(ctx: Context) = unico ?: synchronized(this) { unico ?: AnemiaLocalAndroid(ctx.applicationContext).also { unico = it } }
    }

    private val modelo = ModeloAnemiaMovil.desdeJson(ctx.assets.open("anemia_movil.json").bufferedReader().readText())
    private val L = modelo.lado
    private val interprete: Interpreter = ctx.assets.openFd("conjuntiva_movil.tflite").let { fd ->
        Interpreter(FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength),
            Interpreter.Options().setNumThreads(4))
    }
    private val entrada = ByteBuffer.allocateDirect(L * L * 3 * 4).order(ByteOrder.nativeOrder())
    private val salida = Array(1) { Array(L) { Array(L) { FloatArray(1) } } }

    private fun correr(px: IntArray, espejo: Boolean): FloatArray {
        entrada.rewind()
        for (y in 0 until L) for (x in 0 until L) {
            val p = px[y * L + if (espejo) L - 1 - x else x]
            entrada.putFloat(((p shr 16) and 0xFF).toFloat()); entrada.putFloat(((p shr 8) and 0xFF).toFloat()); entrada.putFloat((p and 0xFF).toFloat())
        }
        entrada.rewind()
        interprete.run(entrada, salida)
        return FloatArray(L * L) { i -> val y = i / L; val x = i % L; salida[0][y][if (espejo) L - 1 - x else x][0] }
    }

    @Synchronized
    private fun analizarSync(jpeg: ByteArray): AnalisisAnemia? {
        val bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size) ?: return null
        val s = L.toFloat() / maxOf(bmp.width, bmp.height)
        val nw = (bmp.width * s).roundToInt().coerceIn(1, L); val nh = (bmp.height * s).roundToInt().coerceIn(1, L)
        val t = (L - nh) / 2; val l = (L - nw) / 2
        val lienzo = Bitmap.createBitmap(L, L, Bitmap.Config.ARGB_8888)
        Canvas(lienzo).apply { drawColor(Color.BLACK); drawBitmap(bmp, Rect(0, 0, bmp.width, bmp.height), Rect(l, t, l + nw, t + nh), Paint(Paint.FILTER_BITMAP_FLAG)) }
        val px = IntArray(L * L); lienzo.getPixels(px, 0, L, 0, 0, L, L)
        val a = correr(px, false); val b = correr(px, true)
        val m = BooleanArray(L * L) { i -> val y = i / L; val x = i % L
            y in t until t + nh && x in l until l + nw && (a[i] + b[i]) / 2 >= modelo.umbralMascara }
        val conj = ColorConjuntiva.mayorComponente(m, L, L)
        val n = conj.count { it }
        val util = nw * nh
        if (n < 80) return modelo.resultado(null, n.toDouble() / util)
        val rgb = FloatArray(n * 3); var j = 0
        for (i in conj.indices) if (conj[i]) { val p = px[i]
            rgb[j++] = ((p shr 16) and 0xFF) / 255f; rgb[j++] = ((p shr 8) and 0xFF) / 255f; rgb[j++] = (p and 0xFF) / 255f }
        return modelo.resultado(ColorConjuntiva.medir(rgb), n.toDouble() / util)
    }

    override suspend fun analizar(jpeg: ByteArray): AnalisisAnemia? = withContext(Dispatchers.Default) { runCatching { analizarSync(jpeg) }.getOrNull() }
}

@Composable
actual fun rememberAnemiaLocal(): AnemiaLocal? {
    val ctx = LocalContext.current
    return remember { runCatching { AnemiaLocalAndroid.de(ctx) }.getOrNull() }
}
