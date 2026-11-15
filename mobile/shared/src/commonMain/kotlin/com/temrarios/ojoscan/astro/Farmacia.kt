package com.temrarios.ojoscan.astro

import kotlin.math.roundToInt

/** Qué tan rápido se degrada con la radiación: medido en el botiquín de la ISS (Du et al., AAPS J 2011). */
enum class Sensibilidad(val factor: Double, val texto: String) { BAJA(0.15, "baja"), MEDIA(0.5, "media"), ALTA(1.0, "alta") }

data class Farmaco(
    val nombre: String,
    val uso: String,
    /** Vida útil en la Tierra, en días. */
    val vidaDias: Int,
    val sensibilidad: Sensibilidad,
    /** Días en almacén antes del lanzamiento. */
    val almacenPrevio: Int = 180,
)

data class EstadoFarmaco(
    val farmaco: Farmaco,
    /** Días que le quedan de vida útil efectiva (negativo = caducado). */
    val diasRestantes: Int,
    /** Potencia estimada (%). USP exige ≥ 90 % hasta la caducidad. */
    val potencia: Double,
    val triage: Triage,
)

/**
 * Matriz farmacéutica y radiación.
 * Modelo: cada día de misión "envejece" al fármaco 1 + s·(dosis/0.5 mSv) días, donde s es su sensibilidad.
 * 0.5 mSv/día es la referencia de la ISS, donde Du et al. vieron que 9 de 35 fármacos ya no cumplían
 * potencia a los 880 días. La potencia cae linealmente de 100 % a 90 % a lo largo de la vida útil.
 */
object Farmacia {
    val botiquin = listOf(
        Farmaco("Prometazina 25 mg (inyectable)", "Mareo espacial intenso", 730, Sensibilidad.ALTA),
        Farmaco("Amoxicilina/clavulánico 875 mg", "Infecciones", 730, Sensibilidad.ALTA),
        Farmaco("Sulfametoxazol/trimetoprim", "Infección urinaria", 1095, Sensibilidad.ALTA),
        Farmaco("Epinefrina 0.3 mg (autoinyector)", "Anafilaxia", 548, Sensibilidad.ALTA),
        Farmaco("Escopolamina/dextroanfetamina", "Mareo espacial", 730, Sensibilidad.MEDIA),
        Farmaco("Ondansetrón 8 mg", "Náusea y vómito", 1095, Sensibilidad.MEDIA),
        Farmaco("Ciprofloxacino 500 mg", "Infecciones", 1095, Sensibilidad.MEDIA),
        Farmaco("Zolpidem 10 mg", "Insomnio", 1095, Sensibilidad.MEDIA),
        Farmaco("Melatonina 3 mg", "Ritmo de sueño", 730, Sensibilidad.MEDIA),
        Farmaco("Furosemida 40 mg", "Edema / retención", 1095, Sensibilidad.MEDIA),
        Farmaco("Lidocaína 2 % (tópica)", "Anestesia local", 730, Sensibilidad.MEDIA),
        Farmaco("Paracetamol 500 mg", "Dolor y fiebre", 1095, Sensibilidad.BAJA),
        Farmaco("Ibuprofeno 400 mg", "Dolor e inflamación", 1095, Sensibilidad.BAJA),
        Farmaco("Loratadina 10 mg", "Alergia", 1095, Sensibilidad.BAJA),
        Farmaco("Modafinilo 200 mg", "Alerta en turnos largos", 1095, Sensibilidad.BAJA),
    )

    fun estado(f: Farmaco, diaMision: Int, entorno: Entorno): EstadoFarmaco {
        val acelera = 1 + f.sensibilidad.factor * (entorno.dosisMsvDia / 0.5)
        val edad = f.almacenPrevio + diaMision * acelera
        val restantes = ((f.vidaDias - edad) / acelera).roundToInt()
        val potencia = (100 - 10 * edad / f.vidaDias).coerceAtLeast(0.0)
        val triage = when {
            restantes < 0 -> Triage.CRITICO
            restantes < 90 -> Triage.ADVERTENCIA
            else -> Triage.NOMINAL
        }
        return EstadoFarmaco(f, restantes, (potencia * 10).roundToInt() / 10.0, triage)
    }

    fun matriz(diaMision: Int, entorno: Entorno) = botiquin.map { estado(it, diaMision, entorno) }.sortedBy { it.diasRestantes }

    fun buscar(texto: String): Farmaco? {
        val t = normalizar(texto)
        return botiquin.firstOrNull { f -> normalizar(f.nombre).split(" ", "/").first().let { it.length > 3 && it in t } }
    }

    /** Límite de carrera de NASA (2022): 600 mSv para toda la vida del astronauta. */
    const val LIMITE_CARRERA_MSV = 600.0
}

/** minúsculas y sin acentos, para comparar texto escrito a mano. */
fun normalizar(s: String): String = s.lowercase()
    .replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ü', 'u').replace('ñ', 'n')
