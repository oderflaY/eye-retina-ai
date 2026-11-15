package com.temrarios.ojoscan

import com.temrarios.ojoscan.astro.ColorConjuntiva
import com.temrarios.ojoscan.astro.ModeloAnemiaMovil
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Kotlin = Python (`medir_color` sin balance + logística) en píxeles reales de conjuntiva de Eyes-Defy. */
class AnemiaMovilTest {
    private val modelo = ModeloAnemiaMovil.desdeJson(File("../androidApp/src/main/assets/anemia_movil.json").readText())
    private val casos = Json.parseToJsonElement(File("src/androidHostTest/resources/anemia_movil_vectores.json").readText()).jsonArray

    @Test
    fun medidasYProbabilidadIgualQuePython() {
        for ((n, c) in casos.withIndex()) {
            val o = c.jsonObject
            val px = o["pixeles"]!!.jsonArray.map { it.jsonPrimitive.double.toFloat() }.toFloatArray()
            val m = ColorConjuntiva.medir(px)!!
            val esperado = o["medidas"]!!.jsonObject
            for ((k, v) in esperado) {
                val e = v.jsonPrimitive.double
                val tol = 1e-3 * maxOf(1.0, abs(e))
                assertTrue(abs(m.getValue(k) - e) < tol, "caso $n $k: ${m[k]} vs $e")
            }
            val p = modelo.probabilidad(m)
            assertTrue(abs(p - o["probabilidad"]!!.jsonPrimitive.double) < 2e-3, "caso $n p $p")
        }
    }

    @Test
    fun mayorComponenteYResultado() {
        // dos manchas: 3 píxeles y 5 píxeles → queda la de 5
        val a = 6
        val m = BooleanArray(36)
        listOf(0, 1, 6).forEach { m[it] = true }
        listOf(22, 23, 28, 29, 35).forEach { m[it] = true }
        assertEquals(5, ColorConjuntiva.mayorComponente(m, a, a).count { it })
        assertEquals("sin_conjuntiva", modelo.resultado(null, 0.0).resultado)
    }
}
