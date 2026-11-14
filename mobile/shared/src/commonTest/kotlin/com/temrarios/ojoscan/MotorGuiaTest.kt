package com.temrarios.ojoscan

import com.temrarios.ojoscan.guia.Aviso
import com.temrarios.ojoscan.guia.Lectura
import com.temrarios.ojoscan.guia.Modo
import com.temrarios.ojoscan.guia.MotorGuia
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class MotorGuiaTest {
    /** Una lectura en la que todo está bien: un ojo de cerca, centrado, abierto y nítido. */
    private fun buena(t: Long) = Lectura(
        tiempoMs = t, ojoVisible = true, confianza = 0.98f, anchoOjo = 0.40f, ojoX = 0.5f, ojoY = 0.5f,
        apertura = 0.32f, brillo = 130f, nitidez = 80f,
    )

    @Test
    fun cadaReglaDaSuAviso() {
        val m = MotorGuia(Modo.OJO)
        val b = buena(0)
        assertNull(m.primeraFalla(b))
        assertEquals(Aviso.SIN_OJO, m.primeraFalla(b.copy(ojoVisible = false)))
        assertEquals(Aviso.CARA_LEJOS, m.primeraFalla(b.copy(ojoVisible = false, caraLejos = true)))
        assertEquals(Aviso.POCA_LUZ, m.primeraFalla(b.copy(brillo = 30f)))
        assertEquals(Aviso.MUCHA_LUZ, m.primeraFalla(b.copy(brillo = 250f)))
        assertEquals(Aviso.NO_MUEVAS, m.primeraFalla(b.copy(giro = 1f)))
        assertEquals(Aviso.ACERCATE, m.primeraFalla(b.copy(anchoOjo = 0.2f)))
        assertEquals(Aviso.ALEJATE, m.primeraFalla(b.copy(anchoOjo = 0.6f)))
        assertEquals(Aviso.CENTRA, m.primeraFalla(b.copy(ojoX = 0.8f)))
        assertEquals(Aviso.CENTRA, m.primeraFalla(b.copy(ojoY = 0.2f)))
        assertEquals(Aviso.ABRE_OJO, m.primeraFalla(b.copy(apertura = 0.1f)))
        assertEquals(Aviso.ABRE_OJO, m.primeraFalla(b.copy(apertura = null)))
        assertEquals(Aviso.ENFOCANDO, m.primeraFalla(b.copy(nitidez = 3f)))
    }

    @Test
    fun enAnemiaNoSePideAbrirElOjo() {
        // Con el párpado jalado y la mirada arriba la apertura medida no sirve.
        val b = buena(0).copy(apertura = 0.05f)
        assertNull(MotorGuia(Modo.ANEMIA).primeraFalla(b))
        assertEquals(Aviso.ABRE_OJO, MotorGuia(Modo.OJO).primeraFalla(b))
        // El resto de las reglas sí aplica.
        assertEquals(Aviso.ACERCATE, MotorGuia(Modo.ANEMIA).primeraFalla(b.copy(anchoOjo = 0.2f)))
    }

    @Test
    fun laPrioridadEsLaDelEnum() {
        // Poca luz y ojo cerrado a la vez: primero la luz.
        val b = buena(0).copy(brillo = 20f, apertura = 0.05f)
        assertEquals(Aviso.POCA_LUZ, MotorGuia().primeraFalla(b))
    }

    @Test
    fun capturaUnaSolaVezTrasSostenerse() {
        val m = MotorGuia()
        var capturas = 0
        var t = 0L
        while (t <= 2000) {
            if (m.evaluar(buena(t)).capturar) capturas++
            t += 33
        }
        assertEquals(1, capturas)
        assertEquals(Aviso.CAPTURADO, m.evaluar(buena(t)).aviso)
    }

    @Test
    fun unaFallaReiniciaLaCuenta() {
        val m = MotorGuia()
        var t = 0L
        repeat(20) { m.evaluar(buena(t)); t += 33 }          // ~660 ms bien
        val g = m.evaluar(buena(t).copy(apertura = 0.08f))   // parpadeo
        assertEquals(0f, g.progreso)
        t += 33
        val g2 = m.evaluar(buena(t))
        assertTrue(g2.progreso < 0.1f, "debe empezar de cero, no seguir donde iba")
        assertFalse(g2.capturar)
    }

    @Test
    fun moverseRapidoCuentaComoMovimiento() {
        val m = MotorGuia()
        m.evaluar(buena(0))
        // El ojo se movió 0.05 del cuadro en 33 ms ≈ 1.5 cuadros/s
        assertEquals(Aviso.NO_MUEVAS, m.primeraFalla(buena(33).copy(ojoX = 0.55f)))
    }

    @Test
    fun elMensajeNoParpadea() {
        val m = MotorGuia()
        var t = 0L
        repeat(15) { m.evaluar(buena(t).copy(brillo = 20f)); t += 33 }
        assertEquals(Aviso.POCA_LUZ, m.evaluar(buena(t).copy(brillo = 20f)).aviso)
        t += 33
        // Un solo cuadro con otro problema no cambia el mensaje
        assertEquals(Aviso.POCA_LUZ, m.evaluar(buena(t).copy(anchoOjo = 0.2f)).aviso)
    }

    @Test
    fun hablaAlCambiarPeroNoRepite() {
        val m = MotorGuia()
        var t = 0L
        val frases = mutableListOf<String>()
        repeat(200) {
            m.evaluar(buena(t).copy(brillo = 20f)).hablar?.let { frases += it }
            t += 33
        }
        assertEquals(listOf(Aviso.POCA_LUZ.texto), frases)
    }

    @Test
    fun reiniciarPermiteOtraFoto() {
        val m = MotorGuia()
        var t = 0L
        while (!m.evaluar(buena(t)).capturar) t += 33
        m.reiniciar()
        var otra = false
        repeat(60) { t += 33; if (m.evaluar(buena(t)).capturar) otra = true }
        assertTrue(otra)
    }
}
