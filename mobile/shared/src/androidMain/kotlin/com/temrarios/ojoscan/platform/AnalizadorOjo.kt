package com.temrarios.ojoscan.platform

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.temrarios.ojoscan.guia.Lectura
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Analiza cada cuadro de la cámara (≈640×480) buscando el OJO, no la cara:
 * - el detector de ojo (TFLite) corre en el cuadrado central del cuadro y da caja + puntos;
 * - el canal de luminancia da el brillo y la nitidez de la zona del ojo;
 * - solo si no hay ojo cerca se pregunta a ML Kit si hay una cara a lo lejos (para decir
 *   "acércate" en vez de "acerca la cámara a tu ojo").
 */
class AnalizadorOjo(
    context: Context,
    private val giro: () -> Float,
    private val emitir: (Lectura) -> Unit,
) : ImageAnalysis.Analyzer {
    private val detector = DetectorOjo.de(context)
    private val aperturas = ClasificadorApertura.de(context)
    private val caras = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setMinFaceSize(0.15f)
            .build()
    )

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(img: ImageProxy) {
        val media = img.image
        if (media == null) {
            img.close()
            return
        }
        val t = SystemClock.elapsedRealtime()
        val rot = img.imageInfo.rotationDegrees
        val luma = Luma.copiar(img)
        val derecha = try {
            enderezar(img.toBitmap(), rot)
        } catch (e: Exception) {
            img.close()
            return
        }
        val d = detector.detectar(derecha, cuadradoCentral(derecha.width, derecha.height))
        if (d.hayOjo) {
            emitir(leer(d, aperturas.apertura(derecha, d), derecha.width, derecha.height, luma, rot, t))
            derecha.recycle()
            img.close()
            return
        }
        derecha.recycle()
        caras.process(InputImage.fromMediaImage(media, rot))
            .addOnSuccessListener { f -> emitir(Lectura(t, confianza = d.confianza, caraLejos = f.isNotEmpty(), giro = giro())) }
            .addOnFailureListener { emitir(Lectura(t, confianza = d.confianza, giro = giro())) }
            .addOnCompleteListener { img.close() }
    }

    private fun leer(d: DetectorOjo.Deteccion, apertura: Float, ancho: Int, alto: Int, luma: Luma, rot: Int, t: Long): Lectura {
        val zona = Rect(d.caja.left.toInt(), d.caja.top.toInt(), d.caja.right.toInt(), d.caja.bottom.toInt())
        val sensor = luma.aSensor(zona, rot)
        return Lectura(
            tiempoMs = t,
            ojoVisible = true,
            confianza = d.confianza,
            anchoOjo = d.ancho / ancho,
            ojoX = d.centro.x / ancho,
            ojoY = d.centro.y / alto,
            apertura = apertura,
            brillo = luma.brillo(sensor),
            nitidez = luma.nitidez(sensor),
            giro = giro(),
        )
    }

    fun cerrar() = caras.close()
}

/** Gira el bitmap para dejarlo derecho (vertical, como lo ve la persona). */
internal fun enderezar(b: Bitmap, rotacion: Int, escala: Float = 1f): Bitmap {
    if (rotacion == 0 && escala == 1f) return b
    val m = Matrix().apply {
        postScale(escala, escala)
        postRotate(rotacion.toFloat())
    }
    return Bitmap.createBitmap(b, 0, 0, b.width, b.height, m, true).also { if (it !== b) b.recycle() }
}

/** Copia del plano Y (luminancia) de un cuadro, en la orientación del sensor. */
class Luma(val datos: ByteArray, val ancho: Int, val alto: Int) {
    companion object {
        fun copiar(img: ImageProxy): Luma {
            val plano = img.planes[0]
            val buf = plano.buffer.duplicate()
            val paso = plano.rowStride
            val w = img.width
            val h = img.height
            val datos = ByteArray(w * h)
            for (y in 0 until h) {
                buf.position(y * paso)
                buf.get(datos, y * w, w)
            }
            return Luma(datos, w, h)
        }
    }

    private fun y(x: Int, yy: Int) = datos[yy * ancho + x].toInt() and 0xFF

    /**
     * Rectángulo en la imagen derecha (como la ve ML Kit) → rectángulo en el sensor.
     * rot = grados que hay que girar el sensor en sentido horario para verlo derecho.
     */
    fun aSensor(r: Rect, rot: Int): Rect {
        fun p(x: Int, y: Int): Pair<Int, Int> = when (rot) {
            90 -> y to (alto - 1 - x)
            180 -> (ancho - 1 - x) to (alto - 1 - y)
            270 -> (ancho - 1 - y) to x
            else -> x to y
        }
        val a = p(r.left, r.top)
        val b = p(r.right, r.bottom)
        return Rect(
            min(a.first, b.first).coerceIn(0, ancho - 1), min(a.second, b.second).coerceIn(0, alto - 1),
            max(a.first, b.first).coerceIn(0, ancho - 1), max(a.second, b.second).coerceIn(0, alto - 1),
        )
    }

    /** Brillo medio (0..255) muestreando cada 4 píxeles. */
    fun brillo(r: Rect): Float {
        var suma = 0L
        var n = 0
        var yy = r.top
        while (yy <= r.bottom) {
            var x = r.left
            while (x <= r.right) { suma += y(x, yy); n++; x += 4 }
            yy += 4
        }
        return if (n == 0) 0f else suma.toFloat() / n
    }

    /** Varianza del laplaciano (4 vecinos): foto movida o desenfocada → valor bajo. */
    fun nitidez(r: Rect): Float? {
        if (r.width() < 6 || r.height() < 6) return null
        var suma = 0.0
        var suma2 = 0.0
        var n = 0
        for (yy in r.top + 1 until r.bottom) for (x in r.left + 1 until r.right) {
            val l = (4 * y(x, yy) - y(x - 1, yy) - y(x + 1, yy) - y(x, yy - 1) - y(x, yy + 1)).toDouble()
            suma += l
            suma2 += l * l
            n++
        }
        if (n == 0) return null
        val m = suma / n
        return (suma2 / n - m * m).toFloat()
    }
}

/** Qué tan rápido gira el teléfono (rad/s), con caída suave para no perder sacudidas cortas. */
class Giroscopio(context: Context) : SensorEventListener {
    private val gestor = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val sensor = gestor.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    @Volatile private var valor = 0f

    fun iniciar() { sensor?.let { gestor.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) } }
    fun detener() = gestor.unregisterListener(this)
    fun magnitud(): Float = valor

    override fun onSensorChanged(e: SensorEvent) {
        val (x, y, z) = Triple(e.values[0], e.values[1], e.values[2])
        valor = max(sqrt(x * x + y * y + z * z), valor * 0.85f)
    }

    override fun onAccuracyChanged(s: Sensor?, a: Int) {}
}

internal fun PointF.dist2(o: PointF): Float = (x - o.x) * (x - o.x) + (y - o.y) * (y - o.y)
