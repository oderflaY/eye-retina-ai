package com.temrarios.ojoscan.astro

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pulso y calidad de señal PPG (dedo + flash), entrenado y validado con BUT PPG 2.0.0 (PhysioNet, CC BY 4.0).
 * Copia exacta de src/training/train_ppg.py → analizar(); los coeficientes salen de models/ppg/calidad.json.
 *
 * Resultados (validación por persona / externa):
 *  - Pulso en grabaciones buenas del dedo vs ECG: error medio 4.5 lpm (antes 6.3), 90 % a ±10 lpm.
 *  - Calidad: el 30 % mejor de las ventanas del dedo → 6.4 lpm (las rechazadas: 18 lpm).
 *  - Externo MTHS (iPhone, 62 personas): con calidad OK 5.0 lpm, 77 % a ±5 lpm, 70 % de ventanas aceptadas.
 */
object ModeloPpg {
    const val FS = 30.0
    const val N = 300 // 10 s a 30 Hz
    private const val NFFT = 1024
    private const val PADLEN = 15
    private val B = doubleArrayOf(0.06004381823803372, 0.0, -0.12008763647606743, 0.0, 0.06004381823803372)
    private val A = doubleArrayOf(1.0, -3.0220041627237406, 3.5511147110086827, -1.9586859700468588, 0.4374973454010612)
    private val ZI = doubleArrayOf(-0.06004381823803372, -0.06004381823803372, 0.06004381823803372, 0.06004381823803372)

    val RASGOS = listOf("snr", "autocorr", "acuerdo", "cv_intervalos", "perfusion", "asimetria", "curtosis", "deriva",
        "rojo_dc", "n_picos_ok", "plantilla", "entropia", "fuera_banda")
    private val MEDIA = doubleArrayOf(0.546271, 0.264211, 0.125748, 0.235424, 0.070023, 0.008067, 3.276255, -0.724529,
        0.657497, 0.787491, 0.768439, 0.777212, 0.831689)
    private val ESCALA = doubleArrayOf(0.172665, 0.258144, 0.165114, 0.175389, 0.848018, 0.865908, 4.761833, 1.040959,
        0.364996, 0.220287, 0.138286, 0.090978, 0.58561)
    private val COEF = doubleArrayOf(0.459697, 0.262504, -0.042784, 0.11092, -0.313489, -0.122625, 0.107568, 0.122186,
        0.045337, 0.251479, 0.350094, 0.278747, -0.006852)
    private const val INTERCEPTO = -0.121972
    /** Probabilidad de que el pulso esté a ±5 lpm del ECG a partir de la cual se acepta (30 % mejor del dedo). */
    const val UMBRAL = 0.637

    class Analisis(val hr: Double, val hrFft: Double, val rasgos: DoubleArray, val probabilidad: Double) {
        /** 0..1 con el umbral del modelo en 0.5 (así el resto de la app usa «≥ 0.5 = confiable»). */
        val calidad: Double get() = if (probabilidad < UMBRAL) 0.5 * probabilidad / UMBRAL else 0.5 + 0.5 * (probabilidad - UMBRAL) / (1 - UMBRAL)
    }

    // ── filtrado (scipy.signal.filtfilt con relleno impar) ──
    private fun lfilter(x: DoubleArray, zi0: Double): DoubleArray {
        val z = DoubleArray(4) { ZI[it] * zi0 }
        val y = DoubleArray(x.size)
        for (n in x.indices) {
            val yn = B[0] * x[n] + z[0]
            for (i in 0 until 3) z[i] = B[i + 1] * x[n] + z[i + 1] - A[i + 1] * yn
            z[3] = B[4] * x[n] - A[4] * yn
            y[n] = yn
        }
        return y
    }

    fun filtrar(x0: DoubleArray): DoubleArray {
        val m = x0.average()
        val x = DoubleArray(x0.size) { x0[it] - m }
        val n = x.size
        val p = min(PADLEN, n - 1)
        val ext = DoubleArray(n + 2 * p)
        for (i in 0 until p) ext[i] = 2 * x[0] - x[p - i]
        for (i in 0 until n) ext[p + i] = x[i]
        for (i in 0 until p) ext[p + n + i] = 2 * x[n - 1] - x[n - 2 - i]
        val ida = lfilter(ext, ext[0])
        val vuelta = lfilter(ida.reversedArray(), ida.last()).reversedArray()
        return vuelta.copyOfRange(p, p + n)
    }

    /** Potencia |FFT|² con ventana de Hann (np.hanning) y relleno a 1024; devuelve (frecuencias, potencias). */
    private val COS = DoubleArray(NFFT) { cos(2 * PI * it / NFFT) }
    private val SIN = DoubleArray(NFFT) { sin(2 * PI * it / NFFT) }

    private fun espectro(y: DoubleArray, fs: Double): Pair<DoubleArray, DoubleArray> {
        val n = y.size
        val w = DoubleArray(n) { 0.5 - 0.5 * cos(2 * PI * it / (n - 1)) }
        val yw = DoubleArray(n) { y[it] * w[it] }
        val bins = NFFT / 2 + 1
        val f = DoubleArray(bins) { it * fs / NFFT }
        val p = DoubleArray(bins)
        for (k in 0 until bins) {
            var re = 0.0; var im = 0.0; var idx = 0
            for (t in 0 until n) { re += yw[t] * COS[idx]; im -= yw[t] * SIN[idx]; idx = (idx + k) and (NFFT - 1) }
            p[k] = re * re + im * im
        }
        return f to p
    }

    private fun std(a: DoubleArray): Double { val m = a.average(); return sqrt(a.sumOf { (it - m) * (it - m) } / a.size) }

    private fun percentil(a: DoubleArray, q: Double): Double {
        val s = a.sortedArray(); val pos = q / 100 * (s.size - 1); val i = floor(pos).toInt()
        return if (i >= s.size - 1) s.last() else s[i] + (pos - i) * (s[i + 1] - s[i])
    }

    private fun mediana(a: List<Double>): Double { val s = a.sorted(); val n = s.size
        return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2 }

    fun picos(y: DoubleArray, fs: Double, hrGuia: Double): List<Int> {
        val n = y.size
        val umbral = 0.3 * std(y)
        fun prominencia(i: Int): Double {
            var izq = y[i]; var j = i - 1
            while (j >= 0 && y[j] <= y[i]) { izq = min(izq, y[j]); j-- }
            var der = y[i]; j = i + 1
            while (j < n && y[j] <= y[i]) { der = min(der, y[j]); j++ }
            return y[i] - max(izq, der)
        }
        val cand = (1 until n - 1).filter { y[it] > y[it - 1] && y[it] >= y[it + 1] && prominencia(it) >= umbral }
        val dist = max(2, (fs * 60 / hrGuia * 0.6).toInt())
        val elegidos = mutableListOf<Int>()
        for (i in cand.sortedWith(compareByDescending<Int> { y[it] }.thenBy { it })) if (elegidos.all { abs(i - it) >= dist }) elegidos += i
        return elegidos.sorted()
    }

    private fun corr(a: DoubleArray, b: DoubleArray): Double {
        val ma = a.average(); val mb = b.average()
        var sab = 0.0; var saa = 0.0; var sbb = 0.0
        for (i in a.indices) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) * (a[i] - ma); sbb += (b[i] - mb) * (b[i] - mb) }
        return sab / sqrt(saa * sbb)
    }

    private fun limpio(v: Double) = when { v.isNaN() -> 0.0; v == Double.POSITIVE_INFINITY -> 10.0; v == Double.NEGATIVE_INFINITY -> -10.0; else -> v }

    /** Analiza 10 s de rojo crudo (0–255) muestreado a 30 Hz. */
    fun analizar(x: DoubleArray, fs: Double = FS): Analisis {
        val n = x.size
        val yf = filtrar(x)
        val y = DoubleArray(n) { -yf[it] }
        val (f, p) = espectro(y, fs)
        val banda = f.indices.filter { f[it] >= 0.7 && f[it] <= 3.5 }
        val i0 = banda.maxByOrNull { p[it] }!!
        val f0 = f[i0]
        val hrFft = f0 * 60
        val pk = picos(y, fs, hrFft)
        val iv = pk.zipWithNext { a, b -> (b - a) / fs * 1000 }.filter { it in 330.0..1500.0 }
        val hrPicos = if (iv.size >= 2) 60000 / mediana(iv) else Double.NaN
        val total = banda.sumOf { p[it] }.coerceAtLeast(1e-12)
        val snr = banda.filter { abs(f[it] - f0) <= 0.15 || abs(f[it] - 2 * f0) <= 0.15 }.sumOf { p[it] } / total
        val my = y.average(); val sy = std(y)
        val yc = DoubleArray(n) { (y[it] - my) / (sy + 1e-9) }
        val lag = (fs * 60 / hrFft).roundToInt()
        val ac = if (lag in 1 until n - 5) (0 until n - lag).sumOf { yc[it] * yc[it + lag] } / (n - lag) else 0.0
        val dc = x.average()
        val L = (fs * 60 / hrFft).toInt()
        val segs = pk.filter { it - L / 2 >= 0 && it + L / 2 <= n }.map { q -> y.copyOfRange(q - L / 2, q + L / 2) }
        val plantilla = if (segs.size >= 3) {
            val tpl = DoubleArray(segs[0].size) { j -> segs.sumOf { it[j] } / segs.size }
            segs.map { corr(it, tpl) }.average()
        } else 0.0
        val xc = DoubleArray(n) { x[it] - dc }
        val (fx, px) = espectro(xc, fs)
        val bx = fx.indices.filter { fx[it] >= 0.7 && fx[it] <= 3.5 }
        val sbx = bx.sumOf { px[it] }.coerceAtLeast(1e-12)
        val entropia = -bx.sumOf { val q = px[it] / sbx; q * ln(q + 1e-12) } / ln(bx.size.toDouble())
        val fuera = fx.indices.filter { (fx[it] > 0.1 && fx[it] < 0.7) || fx[it] > 3.5 }.sumOf { px[it] } / sbx
        // momentos (scipy skew / kurtosis sesgados)
        val m2 = y.sumOf { (it - my) * (it - my) } / n
        val m3 = y.sumOf { val d = it - my; d * d * d } / n
        val m4 = y.sumOf { val d = it - my; d * d * d * d } / n
        // pendiente de mínimos cuadrados de x contra t (s)
        val tm = (n - 1) / 2.0 / fs
        var sxy = 0.0; var sxx = 0.0
        for (i in 0 until n) { val t = i / fs - tm; sxy += t * (x[i] - dc); sxx += t * t }
        val pendiente = sxy / sxx
        val ivMedia = iv.average()
        val r = doubleArrayOf(
            snr, ac,
            if (!hrPicos.isNaN()) abs(hrFft - hrPicos) / hrFft else 1.0,
            if (iv.size >= 3) sqrt(iv.sumOf { (it - ivMedia) * (it - ivMedia) } / iv.size) / ivMedia else 1.0,
            log10(max((percentil(y, 95.0) - percentil(y, 5.0)) / max(abs(dc), 1e-6) * 100, 1e-4)),
            m3 / m2.let { it * sqrt(it) },
            m4 / (m2 * m2) - 3,
            log10(max(abs(pendiente) / max(abs(dc), 1e-6) * 100, 1e-5)),
            dc / 255.0,
            iv.size / max(n / fs * hrFft / 60, 1.0),
            plantilla, entropia, log10(fuera + 1e-6),
        ).map(::limpio).toDoubleArray()
        var z = INTERCEPTO
        for (i in r.indices) z += COEF[i] * (r[i] - MEDIA[i]) / ESCALA[i]
        return Analisis(if (!hrPicos.isNaN()) hrPicos else hrFft, hrFft, r, 1 / (1 + exp(-z)))
    }

    /** Re-muestrea (lineal) a 30 Hz los últimos 10 s. La cámara del teléfono no siempre da 30 fps exactos. */
    fun remuestrear(t: List<Long>, v: List<Double>): DoubleArray? {
        if (t.size < 2 || t.last() - t.first() < 9_500) return null
        val fin = t.last().toDouble(); val ini = fin - (N - 1) * 1000 / FS
        val out = DoubleArray(N)
        var j = 0
        for (i in 0 until N) {
            val ti = ini + i * 1000 / FS
            while (j < t.size - 2 && t[j + 1] < ti) j++
            val t0 = t[j].toDouble(); val t1 = t[j + 1].toDouble()
            val a = if (t1 > t0) ((ti - t0) / (t1 - t0)).coerceIn(0.0, 1.0) else 0.0
            out[i] = v[j] + a * (v[j + 1] - v[j])
        }
        return out
    }
}
