package com.temrarios.ojoscan

import com.temrarios.ojoscan.astro.ConsultaHistorial
import com.temrarios.ojoscan.astro.EstadoRevision
import com.temrarios.ojoscan.astro.Registro
import com.temrarios.ojoscan.astro.Seguimiento
import com.temrarios.ojoscan.astro.Triage
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SeguimientoTest {
    private val D = Seguimiento.DIA
    private val ahora = 20_000 * D + 12 * 3_600_000L

    private fun pulso(diasAtras: Double, v: Double) = Registro(ahora - (diasAtras * D).toLong(), "pulso", "Pulso", v, "lpm", "cámara",
        Seguimiento.clasificarPulso(v), "cámara")

    @Test fun recordatorioSeAcortaSiSalioMal() {
        val bien = listOf(pulso(3.0, 70.0))
        assertEquals(EstadoRevision.AL_DIA, Seguimiento.revisiones(bien, ahora).first { it.tipo.clave == "pulso" }.estado)
        val mal = listOf(pulso(3.0, 125.0)) // amarillo → cada 2 días
        assertEquals(EstadoRevision.VENCIDA, Seguimiento.revisiones(mal, ahora).first { it.tipo.clave == "pulso" }.estado)
        assertEquals(EstadoRevision.NUNCA, Seguimiento.revisiones(bien, ahora).first { it.tipo.clave == "pvt" }.estado)
    }

    @Test fun alertaPorTendenciaContraTuLineaBase() {
        val base = (10..30 step 4).map { pulso(it.toDouble(), 64.0) }
        val semana = listOf(pulso(1.0, 80.0), pulso(2.0, 79.0), pulso(3.0, 82.0))
        val a = Seguimiento.alertas(base + semana, ahora)
        assertTrue(a.any { it.id.startsWith("pulso_sube") }, a.toString())
        assertFalse(Seguimiento.alertas(base + listOf(pulso(1.0, 66.0), pulso(2.0, 65.0), pulso(3.0, 63.0)), ahora).any { it.id.startsWith("pulso_sube") })
    }

    @Test fun alertaRojaInmediata() {
        val a = Seguimiento.alertas(listOf(pulso(0.01, 165.0)), ahora)
        assertEquals(Triage.CRITICO, a.first().triage)
    }

    @Test fun anemiaEnModoPruebasNoAlerta() {
        val h = listOf(Registro(ahora - 20 * D, "anemia", "Anemia (párpado)", 0.30, "%prob", "", Triage.NOMINAL, "párpado"),
            Registro(ahora - 2 * D, "anemia", "Anemia (párpado)", 0.55, "%prob", "", Triage.ADVERTENCIA, "párpado"))
        val a = Seguimiento.alertas(h, ahora)
        if (Seguimiento.ANEMIA_EN_PRUEBAS) {
            assertTrue(a.none { "anemia" in it.id }, a.toString())
            assertTrue(Seguimiento.revisiones(h, ahora).none { it.tipo.clave == "anemia" })
        } else assertTrue(a.any { it.id.startsWith("anemia_sube") })
    }

    @Test fun ragPersonalContestaConSusDatos() {
        val h = (1..8).map { pulso(it * 3.0, 60.0 + it) } +
            Registro(ahora - 2 * D, "diario", "Diario", null, "", "tengo fiebre de 38.5 y me duele la cabeza", Triage.ADVERTENCIA, "diario")
        assertTrue(ConsultaHistorial.esPersonal("¿Cómo va mi pulso este mes?"))
        assertFalse(ConsultaHistorial.esPersonal("Tengo mareo"))
        val r = ConsultaHistorial.responder("¿Cómo va mi pulso este mes?", h, ahora)
        assertTrue("Pulso" in r.texto && "Promedio" in r.texto && "Tendencia" in r.texto, r.texto)
        val f = ConsultaHistorial.responder("¿cuándo tuve fiebre?", h, ahora)
        assertTrue("fiebre" in f.texto, f.texto)
        val t = ConsultaHistorial.responder("¿Cuándo me toca revisión?", h, ahora)
        assertTrue("Test de alerta (PVT): Nunca" in t.texto, t.texto)
    }

    @Test fun jsonIdaYVuelta() {
        val h = listOf(pulso(1.0, 70.0))
        assertEquals(h, Seguimiento.deJson(Seguimiento.aJson(h)))
    }
}
