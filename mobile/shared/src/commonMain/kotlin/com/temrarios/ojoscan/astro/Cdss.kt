package com.temrarios.ojoscan.astro

import kotlin.math.roundToInt

data class Mensaje(val deAstronauta: Boolean, val texto: String, val triage: Triage = Triage.NOMINAL, val manualId: String? = null)

/**
 * Asistente clínico de a bordo (CDSS) sin conexión: reglas clínicas sobre lo que escribe el astronauta
 * y sobre la telemetría actual. Responde al instante; nunca depende de la Tierra ni de la nube.
 */
object Cdss {
    private val ALARMA = listOf("no respira", "inconsciente", "convulsion", "sin pulso", "dolor de pecho", "dolor en el pecho",
        "no puedo respirar", "sangrado abundante", "desmayo", "me desmaye", "garganta cerrada")

    val sugerencias = listOf("¿Cómo estoy?", "Tengo mareo", "¿Tengo anemia?", "Dolor de pecho", "¿Sirve la prometazina?", "Radiación de hoy")

    fun responder(pregunta: String, tm: MissionTelemetry): Mensaje {
        val t = normalizar(pregunta).trim()
        if (t.isEmpty()) return Mensaje(false, "Escribe lo que sientes o lo que necesitas saber.")

        // 1. Señales de alarma: primero la acción, luego el manual
        if (ALARMA.any { it in t }) {
            val m = Manuales.buscar(t)
            return Mensaje(false,
                "ALERTA CRÍTICA. Activa la Baliza SOS ahora y avisa a la tripulación.\n" +
                    (m?.let { "Procedimiento «${it.titulo}»:\n" + it.pasos.take(3).mapIndexed { i, p -> "${i + 1}. $p" }.joinToString("\n") } ?: "") +
                    contextoVital(tm),
                Triage.CRITICO, m?.id)
        }

        // 2. Estado general
        if (listOf("como estoy", "estado", "resumen", "como voy", "mis signos", "telemetria").any { it in t }) return resumen(tm)

        // 3. Fármacos
        Farmacia.buscar(t)?.let { f ->
            val e = Farmacia.estado(f, tm.diaMision, tm.entorno)
            val veredicto = when (e.triage) {
                Triage.CRITICO -> "NO lo uses como primera opción: ya superó su vida útil efectiva por radiación (potencia estimada ${e.potencia} %)."
                Triage.ADVERTENCIA -> "Úsalo, pero le quedan ${e.diasRestantes} días de vida útil efectiva (potencia ${e.potencia} %). Planea reabastecer."
                else -> "En buen estado: potencia estimada ${e.potencia} %, ${e.diasRestantes} días de vida útil efectiva."
            }
            return Mensaje(false, "${f.nombre} — ${f.uso}.\n$veredicto\nSensibilidad a la radiación: ${f.sensibilidad.texto}; " +
                "dosis del entorno ${tm.entorno.dosisMsvDia} mSv/día.", e.triage)
        }

        // 4. Radiación
        if (listOf("radiacion", "dosis", "msv", "rayos cosmicos").any { it in t } && "tormenta" !in t) {
            val pct = tm.dosisAcumuladaMsv / Farmacia.LIMITE_CARRERA_MSV * 100
            val tr = if (pct > 80) Triage.CRITICO else if (pct > 50) Triage.ADVERTENCIA else Triage.NOMINAL
            return Mensaje(false, "Entorno ${tm.entorno.corto}: ${tm.entorno.dosisMsvDia} mSv/día. Dosis acumulada en la misión: " +
                "${(tm.dosisAcumuladaMsv * 10).roundToInt() / 10.0} mSv (${(pct * 10).roundToInt() / 10.0} % del límite de carrera de NASA, 600 mSv).\n" +
                "Fármacos más afectados hoy: " + Farmacia.matriz(tm.diaMision, tm.entorno).take(2).joinToString { "${it.farmaco.nombre} (${it.diasRestantes} d)" },
                tr, "radiacion")
        }

        // 5. Latencia / comunicación
        if (listOf("latencia", "tierra", "control de mision", "houston", "comunicacion").any { it in t }) {
            val l = tm.latenciaSeg
            return Mensaje(false, "Retraso con la Tierra: ${Latencia.texto(l)} solo ida (${Latencia.texto(l * 2)} ida y vuelta). " +
                if (Latencia.autonomo(l)) "Modo autónomo: decide con este asistente y los manuales; reporta después." else "Puedes consultar al control de misión casi en tiempo real.")
        }

        // 6. Fatiga con el PVT
        if (listOf("cansado", "fatiga", "sueno", "agotado", "concentrar").any { it in t }) {
            val p = tm.pvt
            val base = Manuales.todos.first { it.id == "sueno" }
            val pvt = if (p == null) "Haz el test PVT para medir tu alerta." else
                "Tu último PVT: media ${p.mediaMs} ms, ${p.lapsos} lapsos → ${p.triage.etiqueta}." +
                    if (p.triage != Triage.NOMINAL) " No hagas tareas críticas (EVA, acoplamiento) sin relevo." else ""
            return Mensaje(false, "$pvt\n" + base.pasos.drop(1).joinToString("\n") { "• $it" }, p?.triage ?: Triage.ADVERTENCIA, base.id)
        }

        // 7. Anemia: usa el último tamizaje de la cámara
        if (listOf("anemia", "palido", "palidez", "hemoglobina").any { it in t }) {
            val m = Manuales.todos.first { it.id == "anemia" }
            val a = tm.anemia
            val estado = if (a == null) "Aún no hay tamizaje: abre «Tamizaje de anemia» en el tablero." else
                "Probabilidad de anemia juntando todo: ${a.porcentaje} %\n" +
                    a.fuentes.joinToString("\n") { f -> "  · ${f.nombre}: ${f.probabilidad?.let { "${(it * 100).roundToInt()} %" } ?: f.detalle}" }
            val compensa = (a?.probabilidad ?: 0.0) >= 0.5 && (tm.pulso ?: 0) > 100
            return Mensaje(false, estado + (if (compensa) "\nPalidez CON pulso de ${tm.pulso} lpm: tu corazón está compensando. Reposo y vigila." else "") +
                "\n" + m.pasos.mapIndexed { i, p -> "${i + 1}. $p" }.joinToString("\n"),
                if (compensa) Triage.CRITICO else a?.triage?.takeIf { it != Triage.SIN_DATOS } ?: Triage.ADVERTENCIA, m.id)
        }

        // 8. Síntoma con manual
        Manuales.buscar(t)?.let { m ->
            var tr = m.triage
            var extra = ""
            if (m.id == "trauma") tm.impactos.lastOrNull()?.let {
                extra = "\nLa Caja Negra registró un impacto de ${it.g} g" + if (it.severo) " (SEVERO)." else "."
                if (it.severo) tr = Triage.peor(tr, Triage.CRITICO)
            }
            if (m.id == "mareo" && tm.entorno != Entorno.ISS) extra = "\nCambiaste a ${tm.entorno.corto}: la readaptación a la gravedad también da mareo los primeros días."
            val farm = m.farmacos.mapNotNull { n -> Farmacia.botiquin.firstOrNull { it.nombre.startsWith(n.take(10)) } }
                .map { Farmacia.estado(it, tm.diaMision, tm.entorno) }
                .joinToString("\n") { "  · ${it.farmaco.nombre}: ${if (it.triage == Triage.CRITICO) "CADUCADO por radiación" else "${it.diasRestantes} d, ${it.potencia} %"}" }
            return Mensaje(false, "«${m.titulo}»\n" + m.pasos.mapIndexed { i, p -> "${i + 1}. $p" }.joinToString("\n") + extra +
                (if (farm.isNotEmpty()) "\nFármacos del botiquín:\n$farm" else "") + contextoVital(tm), tr, m.id)
        }

        return Mensaje(false, "No reconocí el síntoma. Puedes preguntarme por: dolor de pecho, falta de aire, mareo, golpes, " +
            "heridas, ojos, dolor de cabeza, cólico renal, alergia, fatiga, radiación, un fármaco del botiquín o \"¿cómo estoy?\".")
    }

    fun resumen(tm: MissionTelemetry): Mensaje {
        val l = mutableListOf<String>()
        l += "Estado general: ${tm.triage.etiqueta} · ${tm.entorno.nombre} · día ${tm.diaMision} de misión."
        l += tm.pulso?.let { "Pulso $it lpm (${tm.pulsoFuente})." } ?: "Sin pulso: conecta la pulsera o mide con PPG."
        l += tm.spo2?.let { "SpO2 ${it} % (${tm.spo2Fuente})" + if (it < 92) " — BAJA." else "." } ?: "SpO2 sin dato."
        l += "IMU: pico ${tm.imuPicoG} g, ${tm.impactos.count { it.severo }} impactos severos."
        l += tm.pvt?.let { "PVT: ${it.mediaMs} ms, ${it.lapsos} lapsos (${it.triage.etiqueta})." } ?: "PVT pendiente."
        val caducos = Farmacia.matriz(tm.diaMision, tm.entorno).count { it.triage == Triage.CRITICO }
        tm.anemia?.let { l += "Probabilidad de anemia: ${it.porcentaje} % (${it.fuentes.size} fuentes)." }
        if (caducos > 0) l += "$caducos fármacos caducados por radiación."
        return Mensaje(false, l.joinToString("\n"), tm.triage)
    }

    private fun contextoVital(tm: MissionTelemetry): String {
        val p = listOfNotNull(tm.pulso?.let { "pulso $it lpm" }, tm.spo2?.let { "SpO2 $it %" })
        return if (p.isEmpty()) "" else "\nTus signos ahora: ${p.joinToString(", ")}."
    }
}
