package com.temrarios.ojoscan

import com.temrarios.ojoscan.astro.EstimadorPpg
import com.temrarios.ojoscan.astro.ModeloPpg
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/** Kotlin debe calcular lo mismo que src/training/train_ppg.py en 8 señales reales de BUT PPG (CC BY 4.0). */
class PpgModeloTest {
    private val vectores = Json.parseToJsonElement(File("src/androidHostTest/resources/ppg_vectores_butppg.json").readText()).jsonArray

    @Test
    fun igualQuePython() {
        for (e in vectores) {
            val o = e.jsonObject
            val id = o["id"]!!.jsonPrimitive.int
            val x = o["rojo"]!!.jsonArray.map { it.jsonPrimitive.double }.toDoubleArray()
            val a = ModeloPpg.analizar(x)
            val hr = o["hr"]!!.jsonPrimitive.double
            assertTrue(abs(a.hr - hr) < 0.05, "$id hr ${a.hr} vs $hr")
            val r = o["rasgos"]!!.jsonObject
            ModeloPpg.RASGOS.forEachIndexed { i, k ->
                val py = r[k]!!.jsonPrimitive.double
                assertTrue(abs(a.rasgos[i] - py) <= 1e-4 + 1e-4 * abs(py), "$id $k kotlin ${a.rasgos[i]} python $py")
            }
            val p = o["p"]!!.jsonPrimitive.double
            assertTrue(abs(a.probabilidad - p) < 1e-3, "$id p ${a.probabilidad} vs $p")
        }
    }

    @Test
    fun estimadorConFpsIrregularDaElPulsoDelEcg() {
        // la misma señal real, entregada a ~29 fps con jitter, como la manda la cámara
        var ok = 0
        for (e in vectores) {
            val o = e.jsonObject
            val x = o["rojo"]!!.jsonArray.map { it.jsonPrimitive.double }
            val ecg = o["hr_ecg"]!!.jsonPrimitive.int
            val est = EstimadorPpg()
            var t = 0L
            val r = java.util.Random(7)
            // interpolar la señal de 30 Hz a tiempos irregulares (~34 ms ± 6)
            while (t <= 9_966) {
                val pos = t / (1000.0 / 30); val i = pos.toInt().coerceAtMost(x.size - 2); val f = pos - i
                est.agregar(t, x[i] + f * (x[i + 1] - x[i]))
                t += 28 + r.nextInt(13)
            }
            est.agregar(9_967, x.last())
            val l = est.estimar(t) ?: continue
            val pyHr = o["hr"]!!.jsonPrimitive.double
            if (abs(l.lpm - pyHr) <= 3) ok++
            println("id ${o["id"]} ecg $ecg python ${pyHr.toInt()} kotlin(jitter) ${l.lpm} calidad ${l.calidad}")
        }
        assertTrue(ok >= vectores.size - 1, "con jitter coincide en $ok/${vectores.size}")
    }
}
