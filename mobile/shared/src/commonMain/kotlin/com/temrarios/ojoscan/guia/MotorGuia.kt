package com.temrarios.ojoscan.guia

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Decide, cuadro por cuadro, qué indicarle a la persona y cuándo tomar la foto.
 *
 * Reglas (en orden de prioridad): hay un ojo → luz → teléfono quieto → distancia → centrado →
 * ojo bien abierto (solo en [Modo.OJO]) → nitidez. Si todo está bien durante
 * [Umbrales.sostenerMs] seguidos, pide la foto ([Guia.capturar] = true una sola vez). Cualquier
 * falla reinicia la cuenta, así la foto nunca sale movida, borrosa o con el ojo entrecerrado.
 *
 * En [Modo.ANEMIA] la persona jala su párpado de abajo y mira hacia arriba: no se exige apertura
 * (los párpados cambian de forma) y el servidor verifica que se vea la conjuntiva.
 *
 * No depende de la plataforma: se prueba con lecturas sintéticas (MotorGuiaTest).
 */
class MotorGuia(
    private val modo: Modo = Modo.OJO,
    private val u: Umbrales = Umbrales(),
) {
    private var mostrado: Aviso = Aviso.SIN_OJO
    private var candidato: Aviso = Aviso.SIN_OJO
    private var candidatoDesde = 0L
    private var listoDesde: Long? = null
    private var capturado = false
    private var anterior: Lectura? = null
    private var ultimaVozMs = Long.MIN_VALUE / 2
    // Arranca como si ya hubiera dicho el primer aviso: no se habla antes de analizar algo.
    private var ultimaVoz: Aviso? = Aviso.SIN_OJO

    /** Vuelve a empezar (p. ej. si la foto salió mal o sigue el otro ojo). */
    fun reiniciar() {
        mostrado = Aviso.SIN_OJO
        candidato = Aviso.SIN_OJO
        candidatoDesde = 0L
        listoDesde = null
        capturado = false
        anterior = null
        ultimaVoz = Aviso.SIN_OJO
    }

    fun evaluar(l: Lectura): Guia {
        if (capturado) return Guia(Aviso.CAPTURADO, 1f)
        val falla = primeraFalla(l)
        anterior = l

        // Cuenta regresiva: solo avanza mientras todo esté bien, cualquier falla la reinicia.
        val progreso: Float
        var capturar = false
        if (falla == null) {
            val desde = listoDesde ?: l.tiempoMs.also { listoDesde = it }
            progreso = ((l.tiempoMs - desde).toFloat() / u.sostenerMs).coerceIn(0f, 1f)
            if (progreso >= 1f) {
                capturar = true
                capturado = true
            }
        } else {
            listoDesde = null
            progreso = 0f
        }

        // El mensaje cambia solo si el nuevo se sostiene un momento (sin parpadeo), salvo LISTO,
        // que aparece de inmediato para que la persona sepa que ya va bien.
        val nuevo = falla ?: Aviso.LISTO
        if (nuevo != candidato) {
            candidato = nuevo
            candidatoDesde = l.tiempoMs
        }
        if (nuevo == Aviso.LISTO || l.tiempoMs - candidatoDesde >= u.cambioMs) mostrado = nuevo

        val hablar = if (mostrado != ultimaVoz && l.tiempoMs - ultimaVozMs >= u.pausaVozMs) {
            ultimaVoz = mostrado
            ultimaVozMs = l.tiempoMs
            mostrado.texto
        } else null

        return Guia(if (capturar) Aviso.CAPTURADO else mostrado, progreso, capturar, hablar)
    }

    /** Primer problema de la lectura, o null si está lista para la foto. */
    fun primeraFalla(l: Lectura): Aviso? {
        if (!l.ojoVisible) return if (l.caraLejos) Aviso.CARA_LEJOS else Aviso.SIN_OJO
        if (l.brillo < u.brilloMin) return Aviso.POCA_LUZ
        if (l.brillo > u.brilloMax) return Aviso.MUCHA_LUZ
        if (l.giro > u.giroMax || velocidad(l) > u.velocidadMax) return Aviso.NO_MUEVAS
        if (l.anchoOjo < u.anchoMin) return Aviso.ACERCATE
        if (l.anchoOjo > u.anchoMax) return Aviso.ALEJATE
        if (abs(l.ojoX - 0.5f) > u.desvioCentro || abs(l.ojoY - 0.5f) > u.desvioCentro) return Aviso.CENTRA
        if (modo == Modo.OJO) {
            // Sin medida no se puede asegurar que el ojo esté abierto: se pide abrirlo.
            val a = l.apertura
            if (a == null || a < u.aperturaMin) return Aviso.ABRE_OJO
        }
        val n = l.nitidez
        if (n != null && n < u.nitidezMin) return Aviso.ENFOCANDO
        return null
    }

    /** Velocidad del centro del ojo entre este cuadro y el anterior (fracción del cuadro / s). */
    private fun velocidad(l: Lectura): Float {
        val a = anterior ?: return 0f
        if (!a.ojoVisible) return 0f
        val dt = (l.tiempoMs - a.tiempoMs).coerceAtLeast(1) / 1000f
        if (dt > 0.5f) return 0f // cuadros muy separados: no es una medida fiable
        val dx = l.ojoX - a.ojoX
        val dy = l.ojoY - a.ojoY
        return sqrt(dx * dx + dy * dy) / dt
    }
}
