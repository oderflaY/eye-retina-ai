package com.temrarios.ojoscan

import com.temrarios.ojoscan.astro.BaseAuxilios
import com.temrarios.ojoscan.astro.Semaforo
import com.temrarios.ojoscan.astro.normalizar
import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/** Prueba el RAG con la base real (assets/primeros_auxilios.json) y escribe un reporte de lo que recupera. */
class BaseRealTest {
    private val archivo = File("../androidApp/src/main/assets/primeros_auxilios.json")

    // (frase de diario, temas aceptables para el primer resultado)
    private val casos = listOf(
        "me quemé la mano con aceite caliente" to setOf("quemadura"),
        "me picó un alacrán en el pie" to setOf("mordedura"),
        "me corté el dedo con un cuchillo y sangra" to setOf("hemorragia"),
        "mi hijo se atragantó con un dulce" to setOf("atragantamiento", "bebe"),
        "me sangra mucho la nariz" to setOf("nariz", "hemorragia"),
        "me torcí el tobillo" to setOf("fractura"),
        "mi papá tiene dolor en el pecho" to setOf("corazon", "rcp"),
        "mi abuela tiene la cara chueca y no puede hablar" to setOf("acv"),
        "tuvo una convulsión" to setOf("convulsion"),
        "me mordió un perro" to setOf("mordedura"),
        "hace mucho calor y me siento mareado y con dolor de cabeza" to setOf("calor"),
        "mi amigo tomó cloro" to setOf("envenenamiento"),
        "me salieron ronchas y se me hinchan los labios" to setOf("alergia"),
        "le dio un toque eléctrico" to setOf("electrico"),
        "mi tío es diabético y está temblando y sudando" to setOf("diabetes"),
        "me cayó cloro en el ojo" to setOf("ojo", "envenenamiento"),
        "se desmayó y no despierta" to setOf("desmayo", "rcp"),
        "me pegué en la cabeza y vomité" to setOf("cabeza"),
        "tengo una ampolla en el pie" to setOf("quemadura", "hemorragia", "general"),
        "me raspé la rodilla" to setOf("hemorragia"),
    )

    private val sintomas = listOf(
        "tengo fiebre desde ayer" to listOf("fiebre"),
        "estoy con diarrea" to listOf("diarrea"),
        "tengo mucha tos" to listOf("tos"),
        "me duele la cabeza todos los días" to listOf("dolor de cabeza", "cefalea", "migran"),
        "no puedo dormir" to listOf("sueno", "insomnio"),
        "tengo mucha ansiedad" to listOf("ansiedad"),
        "tengo vómito y náusea" to listOf("nausea", "vomito"),
        "me siento mareado" to listOf("mareo", "vertigo"),
        "me salieron ronchas" to listOf("urticaria", "ronchas", "alergia", "erupcion"),
        "tengo la presión alta" to listOf("presion arterial alta", "hipertension"),
        "tengo dolor de espalda" to listOf("espalda"),
        "me duele la garganta" to listOf("garganta", "faringitis"),
    )

    @Test
    fun recuperaElTemaCorrecto() {
        if (!archivo.exists()) { println("sin base real, se omite"); return }
        val t0 = System.currentTimeMillis()
        val base = BaseAuxilios.desdeJson(archivo.readText())
        val indexar = System.currentTimeMillis() - t0
        var aciertos1 = 0; var aciertos3 = 0
        val reporte = StringBuilder("base: ${base.docs.size} docs, indexada en $indexar ms\n\n")
        for ((frase, temas) in casos) {
            val ev = Semaforo.evaluar(frase)
            val r = base.paraDiario(ev.consulta).filter { it.doc.esPrimerosAuxilios }
            val ok1 = r.firstOrNull()?.doc?.tema in temas
            val ok3 = r.any { it.doc.tema in temas }
            if (ok1) aciertos1++; if (ok3) aciertos3++
            reporte.append("${if (ok1) "OK " else if (ok3) "~3 " else "NO "} [${ev.triage}] $frase\n")
            for (x in r) reporte.append("     ${"%.1f".format(x.puntaje)} ${x.doc.tema} | ${x.doc.pregunta}\n")
        }
        reporte.append("\nprimeros auxilios: acierto@1 $aciertos1/${casos.size}  acierto@2 $aciertos3/${casos.size}\n\n")
        // MedlinePlus: el resumen de «saber más» debe ser del tema del síntoma
        var mlp = 0
        for ((frase, titulo) in sintomas) {
            val m = base.paraDiario(Semaforo.evaluar(frase).consulta, frase).firstOrNull { !it.doc.esPrimerosAuxilios }
            val ok = m != null && titulo.any { normalizar(m.doc.pregunta).contains(it) }
            if (ok) mlp++
            reporte.append("${if (ok) "OK " else "NO "} $frase → ${m?.doc?.pregunta}\n")
        }
        reporte.append("\nMedlinePlus correcto $mlp/${sintomas.size}\n")
        File("build/rag_reporte.txt").writeText(reporte.toString())
        println(reporte)
        assertTrue(aciertos3 >= casos.size * 0.8, "acierto@2 $aciertos3/${casos.size}")
        assertTrue(mlp >= sintomas.size * 0.7, "MedlinePlus $mlp/${sintomas.size}")
    }
}
