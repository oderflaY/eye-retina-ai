package com.temrarios.ojoscan

import com.temrarios.ojoscan.api.AnalisisAnemia
import com.temrarios.ojoscan.api.AnalisisOjo
import com.temrarios.ojoscan.api.AnalisisPerfil
import com.temrarios.ojoscan.api.AnalisisRostro
import com.temrarios.ojoscan.api.IndicadorPerfil
import com.temrarios.ojoscan.api.HallazgoRostro
import com.temrarios.ojoscan.api.OjoEnRostro
import com.temrarios.ojoscan.guia.Nivel
import com.temrarios.ojoscan.guia.ResultadosChequeo
import com.temrarios.ojoscan.guia.pronosticar
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PronosticoTest {
    private fun ojo(principal: String, conf: Double = 0.95) =
        AnalisisOjo(principal, if (principal == "normal") "Sin hallazgos visibles" else principal, confianza = conf, hallazgos = emptyList())
    private fun anemia(resultado: String, p: Double) = AnalisisAnemia(valida = true, resultado = resultado, titulo = resultado, probabilidad = p)
    private fun rostro(ictericia: String = "sin_signos", paralisis: String = "sin_signos", ojos: String = "normal") = AnalisisRostro(
        cara = true,
        hallazgos = listOf(
            HallazgoRostro("ictericia", "Ictericia", 0.1, ictericia),
            HallazgoRostro("paralisis_facial", "Parálisis", 0.1, paralisis),
        ),
        ojos = mapOf("izquierdo" to OjoEnRostro(ojos, ojos, 0.9), "derecho" to OjoEnRostro("normal", "normal", 0.9)),
    )

    @Test
    fun todoNormal() {
        val p = pronosticar(ResultadosChequeo(ojo("normal"), ojo("normal"), anemia("sin_palidez", 0.2), anemia("sin_palidez", 0.3), rostro = rostro()))
        assertEquals(Nivel.OK, p.nivel)
        assertEquals(5, p.usadas.size)
        assertTrue(p.resumen.contains("5 mediciones"))
    }

    @Test
    fun anemiaQueCoincideEsAlta() {
        val p = pronosticar(ResultadosChequeo(conjuntiva = anemia("palidez", 0.8), palma = anemia("palidez", 0.7)))
        assertEquals(Nivel.ALTA, p.conclusiones.first { it.area == "Anemia" }.nivel)
    }

    @Test
    fun anemiaQueDiscrepaEsMedia() {
        val p = pronosticar(ResultadosChequeo(conjuntiva = anemia("palidez", 0.8), palma = anemia("sin_palidez", 0.2)))
        assertEquals(Nivel.MEDIA, p.nivel)
    }

    @Test
    fun palidezMasIctericiaDaCruce() {
        val p = pronosticar(ResultadosChequeo(conjuntiva = anemia("palidez", 0.8), rostro = rostro(ictericia = "posible")))
        assertTrue(p.conclusiones.any { it.area == "Cruce" && it.nivel == Nivel.ALTA })
        assertEquals(Nivel.ALTA, p.conclusiones.first().nivel)  // lo urgente va primero
    }

    @Test
    fun ojoDeCercaMandaSobreElDeLejos() {
        val p = pronosticar(ResultadosChequeo(ojo("normal"), ojo("normal"), rostro = rostro(ojos = "enrojecimiento")))
        assertEquals(Nivel.OK, p.conclusiones.first { it.area == "Ojos" }.nivel)
    }

    @Test
    fun ojoRojoConfirmadoPorLaCara() {
        val p = pronosticar(ResultadosChequeo(ojo("enrojecimiento"), ojo("normal"), rostro = rostro(ojos = "enrojecimiento")))
        val o = p.conclusiones.first { it.area == "Ojos" }
        assertEquals(Nivel.MEDIA, o.nivel)
        assertTrue(o.detalle.contains("foto de la cara"))
    }

    @Test
    fun medicionesFallidasSeReportan() {
        val p = pronosticar(ResultadosChequeo(rostro = AnalisisRostro(cara = false), fallidas = listOf("Palma (sin conexión)")))
        assertTrue(p.usadas.isEmpty())
        assertEquals(2, p.fallidas.size)
    }

    @Test
    fun perfilEntraAlPronostico() {
        fun ind(k: String, n: String) = IndicadorPerfil(k, k, "x", n, k)
        val pf = AnalisisPerfil(listOf(ind("presion", "alta"), ind("corazon", "media"), ind("imc", "ok")))
        val p = pronosticar(ResultadosChequeo(ojo("normal"), ojo("normal"), perfil = pf))
        assertEquals(Nivel.ALTA, p.nivel)
        assertTrue(p.usadas.contains("Tus datos"))
        assertTrue(p.conclusiones.any { it.area == "Cruce" && it.titulo.startsWith("Presión alta") })
        assertTrue(p.conclusiones.any { it.area == "Tus datos" && it.nivel == Nivel.OK })
    }

    @Test
    fun enciaSolaNoAlarmaDeMas() {
        val encia = anemia("palidez", 0.0).copy(experimental = true, probabilidad = null)
        val sola = pronosticar(ResultadosChequeo(encia = encia))
        assertEquals(Nivel.MEDIA, sola.conclusiones.first { it.area == "Anemia" }.nivel)
        // párpado sano + encía pálida: sigue en orden, con la nota
        val conConj = pronosticar(ResultadosChequeo(conjuntiva = anemia("sin_palidez", 0.2), encia = encia))
        val a = conConj.conclusiones.first { it.area == "Anemia" }
        assertEquals(Nivel.OK, a.nivel)
        assertTrue(a.detalle.contains("encía"))
    }

    @Test
    fun pulseraApoyaLaAnemiaYLaApnea() {
        fun ind(k: String, n: String, t: String = k) = IndicadorPerfil(k, k, "x", n, t)
        val pf = AnalisisPerfil(listOf(ind("pulso_tendencia", "media", "Subió 9 lpm"), ind("apnea", "media"), ind("presion", "alta")))
        val p = pronosticar(ResultadosChequeo(conjuntiva = anemia("palidez", 0.8), perfil = pf))
        assertTrue(p.conclusiones.any { it.area == "Cruce" && it.titulo.startsWith("Palidez + pulso") })
        assertTrue(p.conclusiones.any { it.area == "Cruce" && it.titulo.startsWith("Bajadas de oxígeno") && it.nivel == Nivel.ALTA })
    }
}
