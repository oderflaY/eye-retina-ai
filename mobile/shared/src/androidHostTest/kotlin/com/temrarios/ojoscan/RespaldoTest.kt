package com.temrarios.ojoscan

import com.temrarios.ojoscan.astro.EntradaDiario
import com.temrarios.ojoscan.astro.Registro
import com.temrarios.ojoscan.astro.ReporteSalud
import com.temrarios.ojoscan.astro.Respaldo
import com.temrarios.ojoscan.astro.Seguimiento
import com.temrarios.ojoscan.astro.Triage
import com.temrarios.ojoscan.platform.CifradoRespaldo
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RespaldoTest {
    private val D = Seguimiento.DIA
    private val ahora = 20_000 * D

    @Test fun cifradoIdaYVueltaYClaveIncorrecta() {
        val r = Respaldo(creadoMs = ahora, historial = listOf(Registro(ahora, "pulso", "Pulso", 70.0, "lpm")),
            diario = listOf(EntradaDiario(ahora, "me siento bien", 4, Triage.NOMINAL, emptyList(), emptyList(), null, null)), perfil = "{}")
        val json = ReporteSalud.respaldoJson(r)
        val b = CifradoRespaldo.cifrar(json, "secreta123")
        assertTrue(!String(b).contains("me siento bien"))
        assertEquals(r, ReporteSalud.respaldoDe(CifradoRespaldo.descifrar(b, "secreta123")!!))
        assertNull(CifradoRespaldo.descifrar(b, "otra"))
    }

    @Test fun presionYReporte() {
        assertEquals(Triage.NOMINAL, Seguimiento.clasificarPresion(118.0, 76.0))
        assertEquals(Triage.ADVERTENCIA, Seguimiento.clasificarPresion(142.0, 88.0))
        assertEquals(Triage.CRITICO, Seguimiento.clasificarPresion(185.0, 110.0))
        val h = listOf(
            Registro(ahora - 3 * D, "presion", "Presión arterial", 145.0, "mmHg", "145/92", Triage.ADVERTENCIA),
            Registro(ahora - D, "presion", "Presión arterial", 150.0, "mmHg", "150/95", Triage.ADVERTENCIA),
        )
        assertTrue(Seguimiento.alertas(h, ahora).any { it.id.startsWith("presion_alta") })
        val t = ReporteSalud.texto(ReporteSalud.lineas(h, ahora, { "01/01 10:00" }))
        assertTrue("Presión arterial: 2 registro(s)" in t && "Presión alta en varias tomas" in t, t)
    }
}
