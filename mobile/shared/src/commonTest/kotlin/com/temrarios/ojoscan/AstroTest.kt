package com.temrarios.ojoscan

import com.temrarios.ojoscan.astro.Cdss
import com.temrarios.ojoscan.astro.DetectorImpactos
import com.temrarios.ojoscan.astro.Entorno
import com.temrarios.ojoscan.astro.EstimadorPpg
import com.temrarios.ojoscan.astro.Farmacia
import com.temrarios.ojoscan.astro.Latencia
import com.temrarios.ojoscan.astro.MissionTelemetry
import com.temrarios.ojoscan.astro.MotorPvt
import com.temrarios.ojoscan.astro.Triage
import kotlin.math.PI
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class AstroTest {
    @Test
    fun latenciaMarteEnRangoReal() {
        // Con órbitas circulares Marte queda entre 0.52 y 2.52 UA (real: 0.37-2.68): 4.4 a 21 min de luz
        var min = Double.MAX_VALUE; var max = 0.0
        for (d in 0 until 800) {
            val s = Latencia.segundos(Entorno.MARTE, 1_700_000_000_000L + d * 86_400_000L)
            min = minOf(min, s); max = maxOf(max, s)
        }
        assertTrue(min in 250.0..275.0, "mínimo $min s")
        assertTrue(max in 1240.0..1270.0, "máximo $max s")
        assertEquals("+14m 22s", Latencia.texto(862.0))
        assertTrue(!Latencia.autonomo(Latencia.segundos(Entorno.ISS, 0)))
    }

    @Test
    fun impactoSeveroSoloConGolpeFuerte() {
        val d = DetectorImpactos()
        var t = 0L
        repeat(100) { d.muestra(0.0, 0.0, 9.81, t, Entorno.ISS); t += 20 } // en reposo
        val (_, nada) = d.muestra(0.0, 1.0, 9.81, t, Entorno.ISS)
        assertNull(nada)
        t += 2000
        val (g, golpe) = d.muestra(40.0, 0.0, 9.81, t, Entorno.ISS) // ~4 g de golpe
        assertTrue(g > 3.0)
        assertNotNull(golpe); assertTrue(golpe.severo)
        // en Marte el umbral es mayor
        assertTrue(d.umbral(Entorno.MARTE) > d.umbral(Entorno.ISS))
    }

    @Test
    fun pvtCuentaLapsosYAdelantos() {
        val p = MotorPvt { 0.0 }
        p.ensayos = 4
        var t = 0L
        for (ms in listOf(250, 300, 620, 280)) {
            p.siguiente(); t += p.esperaMs
            if (ms == 300) p.tocar(t - 10) // adelanto antes del estímulo
            p.mostrar(t); t += ms; p.tocar(t)
        }
        val r = p.resultado(t)!!
        assertEquals(1, r.lapsos)
        assertEquals(1, r.adelantos)
        assertEquals(Triage.ADVERTENCIA, r.triage)
        assertEquals(MotorPvt.Fase.TERMINADO, p.fase)
    }

    @Test
    fun ppgEncuentraElPulso() {
        val e = EstimadorPpg()
        var t = 0L
        // 72 lpm = 1.2 Hz, 30 fps, el rojo BAJA con cada latido
        repeat(30 * 12) { i ->
            e.agregar(t, 200 - 3 * sin(2 * PI * 1.2 * i / 30.0)); t += 33
        }
        val l = e.estimar(t)!!
        assertTrue(l.lpm in 68..76, "lpm ${l.lpm}")
        assertTrue(l.calidad > 0.7)
    }

    @Test
    fun radiacionAdelantaLaCaducidad() {
        val prometazina = Farmacia.botiquin.first { it.nombre.startsWith("Prometazina") }
        val iss = Farmacia.estado(prometazina, 300, Entorno.ISS)
        val luna = Farmacia.estado(prometazina, 300, Entorno.LUNA)
        assertTrue(luna.diasRestantes < iss.diasRestantes)
        assertEquals(Triage.CRITICO, Farmacia.estado(prometazina, 600, Entorno.LUNA).triage)
        val paracetamol = Farmacia.botiquin.first { it.nombre.startsWith("Paracetamol") }
        assertEquals(Triage.NOMINAL, Farmacia.estado(paracetamol, 30, Entorno.ISS).triage)
    }

    @Test
    fun asistenteDaPrioridadALaAlarma() {
        val tm = MissionTelemetry(pulso = 125, spo2 = 91.0)
        val r = Cdss.responder("me duele, tengo dolor de pecho", tm)
        assertEquals(Triage.CRITICO, r.triage)
        assertTrue("Baliza SOS" in r.texto)
        assertTrue("125" in r.texto)
        assertEquals("mareo", Cdss.responder("tengo mucho mareo y nauseas", tm).manualId)
        assertTrue("Prometazina" in Cdss.responder("¿sirve la prometazina?", tm).texto)
        assertEquals("sueno", Cdss.responder("no me puedo concentrar", tm).manualId)
    }
}

class FusionAnemiaTest {
    private fun l(m: String, p: Double) = com.temrarios.ojoscan.astro.LecturaAnemia(m, "", "", p, null, 1000L)

    @Test
    fun juntaFuentesQueCoinciden() {
        val sola = com.temrarios.ojoscan.astro.FusionAnemia.combinar(listOf(l("párpado", 0.7)), null, 2000L)!!
        val dos = com.temrarios.ojoscan.astro.FusionAnemia.combinar(listOf(l("párpado", 0.7), l("palma", 0.7)), null, 2000L)!!
        assertTrue(dos.probabilidad > sola.probabilidad, "coincidir sube la certeza")
        val pulso = com.temrarios.ojoscan.astro.FusionAnemia.combinar(listOf(l("párpado", 0.7)), 102.0, 2000L)!!
        assertTrue(pulso.probabilidad > sola.probabilidad, "pulso alto suma evidencia")
    }

    @Test
    fun enciaCasiNoPesa() {
        val r = com.temrarios.ojoscan.astro.FusionAnemia.combinar(listOf(l("párpado", 0.2), l("encía", 0.95)), null, 2000L)!!
        assertTrue(r.probabilidad < 0.4, "la encía experimental no voltea al párpado: ${r.probabilidad}")
        assertEquals(Triage.NOMINAL, r.triage)
    }

    @Test
    fun ignoraMedicionesViejas() {
        assertNull(com.temrarios.ojoscan.astro.FusionAnemia.combinar(listOf(l("palma", 0.9)), null, 1000L + 31L * 86_400_000L))
    }
}

class VeredictoAnemiaTest {
    private val f = com.temrarios.ojoscan.astro.FusionAnemia

    @Test
    fun tresCoincidenEnPalidez() {
        val v = f.veredicto(0.8, mapOf("párpado" to 0.8, "palma" to 0.75, "encía" to 0.7))
        assertEquals("Probable anemia", v.titulo)
        assertTrue(v.faltan.isEmpty())
    }

    @Test
    fun contradiccionPideRepetir() {
        assertEquals("Resultados contradictorios", f.veredicto(0.5, mapOf("párpado" to 0.85, "palma" to 0.2)).titulo)
    }

    @Test
    fun soloUnaEsPreliminar() {
        val v = f.veredicto(0.2, mapOf("párpado" to 0.2))
        assertEquals("Sin signos de anemia (preliminar)", v.titulo)
        assertEquals(listOf("palma", "encía"), v.faltan)
    }
}
