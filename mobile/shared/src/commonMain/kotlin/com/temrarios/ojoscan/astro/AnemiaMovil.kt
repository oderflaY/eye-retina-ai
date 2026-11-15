package com.temrarios.ojoscan.astro

import com.temrarios.ojoscan.api.AnalisisAnemia
import com.temrarios.ojoscan.api.ValidacionAnemia
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Anemia por la conjuntiva SIN internet: las mismas 12 medidas de color que el servidor
 * (`src/deployment/inference_anemia.medir_color`, sin balance de blancos) + regresión logística.
 * La máscara la pone el segmentador chico del teléfono (`conjuntiva_movil.tflite`).
 * Entrenado y validado con `src/training/train_conjuntiva_movil.py` (Eyes-Defy con Hb de laboratorio).
 */
@Serializable
data class ModeloAnemiaMovil(
    val caracteristicas: List<String>,
    val media: List<Double>,
    val escala: List<Double>,
    val coef: List<Double>,
    val intercepto: Double,
    @kotlinx.serialization.SerialName("zona_gris") val zonaGris: List<Double>,
    @kotlinx.serialization.SerialName("umbral_mascara") val umbralMascara: Double = 0.5,
    val lado: Int = 256,
    val validacion: Map<String, kotlinx.serialization.json.JsonElement> = emptyMap(),
) {
    fun probabilidad(m: Map<String, Double>): Double {
        var z = intercepto
        for ((i, k) in caracteristicas.withIndex()) z += coef[i] * ((m.getValue(k) - media[i]) / escala[i])
        return 1 / (1 + exp(-z))
    }

    private fun num(k: String) = (validacion[k] as? kotlinx.serialization.json.JsonPrimitive)?.content?.toDoubleOrNull()

    /** Mismo formato que la respuesta del servidor, para que la pantalla y el pronóstico no cambien. */
    fun resultado(medidas: Map<String, Double>?, fraccion: Double): AnalisisAnemia {
        val aviso = "Tamizaje de investigación, analizado en el teléfono sin internet: la anemia solo se confirma con un análisis de sangre (hemoglobina)."
        val v = ValidacionAnemia(auc = num("auc_cv"), n = num("n")?.toInt())
        if (medidas == null) return AnalisisAnemia(false, "sin_conjuntiva", "No se ve la parte interna del párpado",
            "Jala tu párpado inferior hacia abajo con un dedo y mira hacia arriba, para que se vea bien la parte rosada de adentro.",
            fraccionConjuntiva = fraccion, aviso = aviso, model = "conjuntiva_movil")
        val p = probabilidad(medidas)
        val (bajo, alto) = zonaGris[0] to zonaGris[1]
        val (res, tit, desc) = when {
            p >= alto -> Triple("palidez", "Conjuntiva pálida: posible anemia",
                "El interior del párpado se ve más pálido de lo normal. Conviene hacer una biometría hemática para medir la hemoglobina.")
            p < bajo -> Triple("sin_palidez", "Color de conjuntiva normal", "El interior del párpado tiene buena coloración; no se ven signos de anemia.")
            else -> Triple("no_concluyente", "Resultado no concluyente",
                "El color quedó en un punto intermedio. Repite la foto con luz blanca de día, jalando bien el párpado. Si vuelve a salir así, hazte una biometría hemática.")
        }
        return AnalisisAnemia(true, res, tit, desc, probabilidad = p, umbral = 0.5, zonaGris = zonaGris,
            fraccionConjuntiva = fraccion, validacion = v, aviso = aviso, model = "conjuntiva_movil")
    }

    companion object {
        private val json = Json { ignoreUnknownKeys = true }
        fun desdeJson(s: String): ModeloAnemiaMovil = json.decodeFromString(serializer(), s)
    }
}

object ColorConjuntiva {
    /** Igual a `np.percentile` (interpolación lineal). [v] se ordena en el lugar. */
    fun percentil(v: DoubleArray, q: Double): Double {
        v.sort()
        val pos = (v.size - 1) * q / 100.0
        val i = floor(pos).toInt()
        val f = pos - i
        return if (i + 1 < v.size) v[i] + f * (v[i + 1] - v[i]) else v[i]
    }

    private fun lineal(c: Double) = if (c <= 0.04045) c / 12.92 else ((c + 0.055) / 1.055).pow(2.4)
    private fun fLab(t: Double) = if (t > 0.008856) t.pow(1.0 / 3) else 7.787 * t + 16.0 / 116

    /** sRGB 0..1 → L*a*b* como `cv2.cvtColor(float32, COLOR_RGB2Lab)` (blanco D65 de OpenCV). */
    fun lab(r: Double, g: Double, b: Double): DoubleArray {
        val R = lineal(r); val G = lineal(g); val B = lineal(b)
        val x = (0.412453 * R + 0.357580 * G + 0.180423 * B) / 0.950456
        val y = 0.212671 * R + 0.715160 * G + 0.072169 * B
        val z = (0.019334 * R + 0.119193 * G + 0.950227 * B) / 1.088754
        val L = if (y > 0.008856) 116 * y.pow(1.0 / 3) - 16 else 903.3 * y
        return doubleArrayOf(L, 500 * (fLab(x) - fLab(y)), 200 * (fLab(y) - fLab(z)))
    }

    /**
     * Las 12 medidas de `medir_color` sin balance de blancos. [px] = r,g,b intercalados (0..1) de los píxeles
     * de la conjuntiva. null si no hay suficientes.
     */
    fun medir(px: FloatArray): Map<String, Double>? {
        val n0 = px.size / 3
        if (n0 < 50) return null
        // sin brillos especulares ni sombras profundas (si quedan < 50, se usan todos)
        var idx = (0 until n0).filter { val v = max(px[3 * it], max(px[3 * it + 1], px[3 * it + 2])); v > 0.08f && v < 0.98f }
        if (idx.size < 50) idx = (0 until n0).toList()
        val n = idx.size
        val eps = 1e-3
        val erit = DoubleArray(n); val la = DoubleArray(n); val ll = DoubleArray(n); val lb = DoubleArray(n)
        var sR = 0.0; var sG = 0.0; var sS = 0.0; var rojos = 0; val rg = DoubleArray(n)
        for ((j, i) in idx.withIndex()) {
            val r0 = px[3 * i].toDouble(); val g0 = px[3 * i + 1].toDouble(); val b0 = px[3 * i + 2].toDouble()
            val r = r0 + eps; val g = g0 + eps; val b = b0 + eps
            erit[j] = ln(r) - ln(g)
            val c = lab(r0, g0, b0); ll[j] = c[0]; la[j] = c[1]; lb[j] = c[2]
            val mx = max(r0, max(g0, b0)); val mn = min(r0, min(g0, b0))
            val s = if (mx > 0) (mx - mn) / mx else 0.0
            var h = if (mx == mn) 0.0 else when (mx) {
                r0 -> 60 * (g0 - b0) / (mx - mn)
                g0 -> 120 + 60 * (b0 - r0) / (mx - mn)
                else -> 240 + 60 * (r0 - g0) / (mx - mn)
            }
            if (h < 0) h += 360
            if ((h < 20 || h > 340) && s > 0.35) rojos++
            sS += s
            val suma = r + g + b
            sR += r / suma; sG += g / suma
            rg[j] = r / g
        }
        return mapOf(
            "eritema" to erit.average(),
            "eritema_p75" to percentil(erit.copyOf(), 75.0),
            "a_media" to la.average(),
            "a_p25" to percentil(la.copyOf(), 25.0),
            "a_p75" to percentil(la.copyOf(), 75.0),
            "l_media" to ll.average(),
            "b_media" to lb.average(),
            "crom_r" to sR / n,
            "crom_g" to sG / n,
            "rojo_alto" to rojos.toDouble() / n,
            "saturacion" to sS / n,
            "r_sobre_g" to percentil(rg, 50.0),
        )
    }

    /** Mayor componente conexo (8 vecinos) de una máscara [ancho]×[alto]. */
    fun mayorComponente(m: BooleanArray, ancho: Int, alto: Int): BooleanArray {
        val etiqueta = IntArray(m.size)
        var mejor = 0; var mejorN = 0; var sig = 0
        val cola = IntArray(m.size)
        for (s in m.indices) {
            if (!m[s] || etiqueta[s] != 0) continue
            sig++; var cab = 0; var fin = 0; cola[fin++] = s; etiqueta[s] = sig
            while (cab < fin) {
                val p = cola[cab++]; val x = p % ancho; val y = p / ancho
                for (dy in -1..1) for (dx in -1..1) {
                    val nx = x + dx; val ny = y + dy
                    if (nx < 0 || ny < 0 || nx >= ancho || ny >= alto) continue
                    val q = ny * ancho + nx
                    if (m[q] && etiqueta[q] == 0) { etiqueta[q] = sig; cola[fin++] = q }
                }
            }
            if (fin > mejorN) { mejorN = fin; mejor = sig }
        }
        return BooleanArray(m.size) { etiqueta[it] == mejor && mejor != 0 }
    }
}
