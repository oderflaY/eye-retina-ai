package com.temrarios.ojoscan

import com.temrarios.ojoscan.pulso.Muestra
import com.temrarios.ojoscan.pulso.parsearHrm
import com.temrarios.ojoscan.pulso.resumir
import com.temrarios.ojoscan.historial.paraServidor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class PulsoTest {
    @Test
    fun parseaFormatoEstandar() {
        // 8 bits, sin contacto ni RR
        assertEquals(72, parsearHrm(byteArrayOf(0x00, 72))!!.first)
        // 16 bits + contacto detectado + 2 RR (1024 = 1000 ms, 512 = 500 ms)
        val (lpm, contacto, rr) = parsearHrm(byteArrayOf(0x17, 0x2C, 0x01, 0x00, 0x04, 0x00, 0x02))!!
        assertEquals(300, lpm); assertEquals(true, contacto); assertEquals(listOf(1000, 500), rr)
        assertNull(parsearHrm(byteArrayOf(0x01)))
    }

    @Test
    fun reposoEsElTramoMasTranquilo() {
        // 60 s a 70 lpm y luego 40 s a 110 lpm (se movió para la foto)
        val m = (0 until 60).map { Muestra(it * 1000L, 70, listOf(857)) } + (60 until 100).map { Muestra(it * 1000L, 110, listOf(545)) }
        val r = assertNotNull(resumir(m))
        assertEquals(70, r.reposo); assertEquals(110, r.maximo); assertEquals(99, r.segundos)
        assertNull(resumir(m.take(5)))
    }
}

class HistorialTest {
    @Test
    fun resumenParaServidor() {
        val dias = (1..30).map { "2026-10-" + it.toString().padStart(2, '0') }
        val h = com.temrarios.ojoscan.historial.ResumenHistorial(
            dias = 30,
            pulsoReposo = dias.mapIndexed { i, d -> com.temrarios.ojoscan.historial.Diario(d, if (i >= 23) 80.0 else 68.0) },
            spo2 = List(20) { if (it < 4) 88.0 else 97.0 },
            sueno = dias.map { com.temrarios.ojoscan.historial.Diario(it, 6.5) } + com.temrarios.ojoscan.historial.Diario("siesta", 0.5),
            pasos = dias.map { com.temrarios.ojoscan.historial.Diario(it, 4000.0) } + com.temrarios.ojoscan.historial.Diario("sin pulsera", 10.0),
        )
        val s = h.paraServidor()
        assertEquals(80.0, s.pulsoReposo7d); assertEquals(68.0, s.pulsoReposoBase)
        assertEquals(0.2, s.spo2Bajas); assertEquals(30, s.suenoNoches); assertEquals(6.5, s.suenoHoras)
        assertEquals(4000.0, s.pasos); assertEquals(30, s.pasosDias)
    }
}
