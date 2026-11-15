package com.temrarios.ojoscan

import com.temrarios.ojoscan.astro.BaseAuxilios
import com.temrarios.ojoscan.astro.DocAuxilio
import com.temrarios.ojoscan.astro.EntradaDiario
import com.temrarios.ojoscan.astro.MissionTelemetry
import com.temrarios.ojoscan.astro.Semaforo
import com.temrarios.ojoscan.astro.Triage
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DiarioTest {
    private fun color(t: String, animo: Int? = null) = Semaforo.evaluar(t, animo).triage

    @Test
    fun rojasSeReconocen() {
        val rojas = listOf(
            "Mi papá tiene dolor en el pecho que se le va al brazo izquierdo",
            "no puedo respirar bien, siento que me ahogo",
            "mi hijo se desmayó en la escuela",
            "Mi abuela tiene la cara chueca y habla raro",
            "me corté y no para de sangrar",
            "se me hinchó la lengua después de comer cacahuate",
            "Ya no quiero vivir",
            "mi bebé de 2 meses tiene fiebre",
            "Tuvo una convulsión hace 5 minutos",
            "se cayó, se pegó un golpe en la cabeza y ahora vomita",
            "tengo 41 grados de temperatura",
            "mi hermano se tomó todas las pastillas de mi mamá",
            "mi amigo tomó cloro", "me salieron ronchas y se me hinchan los labios", "le dio un toque eléctrico", "me cayó cloro en el ojo", "me pegué en la cabeza y vomité",
        )
        for (t in rojas) assertEquals(Triage.CRITICO, color(t), t)
    }

    @Test
    fun amarillasSeReconocen() {
        val amarillas = listOf(
            "Me duele la cabeza desde la mañana", "tengo fiebre de 38.5", "Estoy con diarrea y algo de náusea",
            "me corté el dedo cocinando", "me quemé la mano con el sartén", "Me torcí el tobillo jugando fútbol",
            "Me picó un alacrán", "tengo tos y mocos", "me siento mareado", "dolor de 9/10 en la espalda",
            "mi hijo se atragantó con un dulce pero ya respira", "me raspé la rodilla", "mi tío es diabético y está temblando y sudando",
            "me sangra mucho la nariz",
        )
        for (t in amarillas) assertEquals(Triage.ADVERTENCIA, color(t), t)
    }

    @Test
    fun verdesYNegaciones() {
        val verdes = listOf(
            "Hoy me siento muy bien, dormí 8 horas", "Salí a correr y comí bien", "No tengo fiebre ni dolor",
            "ya no me duele el pecho, me siento mejor", "Sin mareo hoy", "no he tenido convulsiones este mes",
        )
        for (t in verdes) assertEquals(Triage.NOMINAL, color(t), t)
    }

    @Test
    fun animoMuyBajoEsAmarillo() {
        assertEquals(Triage.ADVERTENCIA, color("", animo = 1))
        assertEquals(Triage.NOMINAL, color("", animo = 4))
    }

    @Test
    fun crisisMuestraLineaDeLaVida() {
        val e = Semaforo.evaluar("me quiero morir")
        assertEquals(Triage.CRITICO, e.triage)
        assertTrue("800 911 2000" in e.accion)
    }

    @Test
    fun sintomaRepetidoTresDiasSubeYExplica() {
        val dia = 86_400_000L
        val ahora = 20_000 * dia
        val previas = listOf(1L, 2L).map { d ->
            EntradaDiario(ahora - d * dia, "me duele la cabeza", null, Triage.ADVERTENCIA, sintomas = listOf("dolor_cabeza"))
        }
        val e = Semaforo.evaluar("otra vez dolor de cabeza", previas = previas, ahora = ahora)
        assertEquals(Triage.ADVERTENCIA, e.triage)
        assertTrue(e.motivos.any { it.startsWith("Llevas 3 días") }, e.motivos.toString())
        assertTrue("médico" in e.accion)
    }

    @Test
    fun signosCriticosDeLaPulseraPintanRojo() {
        val e = Semaforo.evaluar("me siento bien", tm = MissionTelemetry(spo2 = 86.0))
        assertEquals(Triage.CRITICO, e.triage)
    }

    @Test
    fun ragEncuentraElProcedimientoCorrecto() {
        val base = BaseAuxilios(listOf(
            DocAuxilio("1", "b", "quemadura", "¿Qué hacer con una quemadura?", "Enfríe la quemadura con agua corriente fresca durante 20 minutos. No use hielo ni mantequilla."),
            DocAuxilio("2", "f", "mordedura", "¿Cómo tratar la picadura de un escorpión?", "Lave la zona, aplique frío y busque atención médica de inmediato."),
            DocAuxilio("3", "f", "hemorragia", "¿Cómo detener una hemorragia de un corte?", "Presione la herida con un paño limpio de forma firme y constante."),
            DocAuxilio("4", "b", "atragantamiento", "¿Qué hacer si alguien se atraganta?", "Dé cinco golpes en la espalda y cinco compresiones abdominales."),
            DocAuxilio("5", "f", "nariz", "¿Cómo detener el sangrado nasal?", "Incline la cabeza hacia adelante y presione la parte blanda de la nariz."),
        ), emptyMap(), emptySet(), "")
        assertEquals("1", base.buscar("me quemé con aceite", 1, minimo = 0.5).first().doc.id)
        assertEquals("2", base.buscar("me picó un alacrán", 1, minimo = 0.5).first().doc.id)
        assertEquals("3", base.buscar("me corté y sangra", 1, minimo = 0.5).first().doc.id)
        assertEquals("4", base.buscar("mi hijo se atragantó con un dulce", 1, minimo = 0.5).first().doc.id)
        assertEquals("5", base.buscar("me sangra la nariz", 1, minimo = 0.5).first().doc.id)
        assertTrue(base.buscar("cuál es la capital de Francia", 1, minimo = 0.5).isEmpty())
    }
}
