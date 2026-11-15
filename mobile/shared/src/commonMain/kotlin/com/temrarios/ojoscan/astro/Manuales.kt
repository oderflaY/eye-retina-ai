package com.temrarios.ojoscan.astro

/** Procedimiento médico de a bordo (base de datos local, sin conexión). */
data class Manual(
    val id: String,
    val titulo: String,
    val categoria: String,
    val triage: Triage,
    /** Palabras que lo activan desde el asistente (sin acentos). */
    val claves: List<String>,
    val pasos: List<String>,
    val farmacos: List<String> = emptyList(),
)

object Manuales {
    val todos = listOf(
        Manual("rcp", "Paro cardiaco: RCP en microgravedad", "Emergencia", Triage.CRITICO,
            listOf("paro", "no respira", "inconsciente", "rcp", "sin pulso"),
            listOf(
                "Activa la Baliza SOS y avisa a toda la tripulación.",
                "Fija al paciente: en microgravedad, sujétalo a la camilla o una pared con las correas.",
                "Posición Evetts-Russomano: tus piernas sobre sus hombros, compresiones con los brazos rectos. " +
                    "Alternativa: parada de manos contra el techo con los pies.",
                "Comprime el centro del pecho 5-6 cm a 100-120 por minuto. 30 compresiones : 2 ventilaciones.",
                "Conecta el desfibrilador en cuanto esté listo y sigue sus indicaciones; aleja a todos al descargar.",
                "Turna al rescatador cada 2 minutos: la fatiga en microgravedad es más rápida.",
            ), listOf("Epinefrina 0.3 mg (autoinyector)")),
        Manual("dolor_pecho", "Dolor de pecho", "Cardiovascular", Triage.CRITICO,
            listOf("dolor de pecho", "dolor en el pecho", "opresion", "angina", "infarto"),
            listOf(
                "Suspende toda actividad y fíjate en reposo.",
                "Mide pulso y SpO2 con AstroPulse; registra la hora de inicio del dolor.",
                "Si hay ECG de a bordo, haz un registro de 12 derivaciones y transmítelo.",
                "Si no hay alergia ni sangrado: aspirina 300 mg masticada.",
                "Oxígeno si SpO2 < 94 %.",
                "Dolor > 15 min, sudor frío, falta de aire o desmayo: Baliza SOS y protocolo de evacuación.",
            )),
        Manual("hipoxia", "Hipoxia / baja de oxígeno", "Respiratorio", Triage.CRITICO,
            listOf("falta de aire", "no puedo respirar", "ahogo", "oxigeno bajo", "hipoxia", "spo2"),
            listOf(
                "Ponte la mascarilla de oxígeno de emergencia de inmediato.",
                "Revisa la presión parcial de O2 y CO2 de la cabina: puede ser falla de soporte vital, no del cuerpo.",
                "Si el CO2 está alto (> 4 mmHg), muévete a un módulo con ventilación activa.",
                "Mide SpO2 cada 2 min. Por debajo de 90 % que no sube con oxígeno: Baliza SOS.",
            )),
        Manual("descompresion", "Descompresión / fuga de cabina", "Emergencia", Triage.CRITICO,
            listOf("fuga", "descompresion", "presion de cabina", "alarma de presion"),
            listOf(
                "Mascarilla de oxígeno puesta antes de cualquier otra acción.",
                "Calcula el tiempo de reserva con el procedimiento de fugas y aísla el módulo afectado.",
                "Vigila dolor articular, picazón o confusión: síntomas de enfermedad por descompresión.",
                "Si aparecen: oxígeno al 100 % y repressurizar; reporta en la bitácora de voz.",
            )),
        Manual("mareo", "Síndrome de adaptación espacial (mareo)", "Neurológico", Triage.ADVERTENCIA,
            listOf("mareo", "nausea", "vomito", "vomitar", "ganas de vomitar", "mareado"),
            listOf(
                "Es normal en los primeros 2-3 días de microgravedad o al cambiar de gravedad.",
                "Fija la cabeza y mira un punto fijo con una referencia clara de \"arriba\".",
                "Evita girar la cabeza rápido. Hidrátate en sorbos pequeños.",
                "Leve: escopolamina/dextroanfetamina oral. Con vómito: prometazina 25 mg intramuscular.",
                "Vómito que no cede en 24 h o deshidratación: avisa al médico de vuelo.",
            ), listOf("Escopolamina/dextroanfetamina", "Prometazina 25 mg (inyectable)", "Ondansetrón 8 mg")),
        Manual("trauma", "Golpe o impacto contra la estructura", "Trauma", Triage.ADVERTENCIA,
            listOf("golpe", "impacto", "me pegue", "choque", "caida", "me cai"),
            listOf(
                "Revisa la conciencia: nombre, lugar, fecha. Si hay confusión o pérdida de conocimiento: SOS.",
                "Busca dolor en cuello o espalda: si lo hay, inmoviliza antes de moverte.",
                "Explora la zona del golpe: deformidad, hinchazón o sangrado.",
                "Frío local 20 min. Dolor: paracetamol o ibuprofeno.",
                "La Caja Negra IMU guarda la fuerza del impacto: inclúyela en el reporte.",
            ), listOf("Paracetamol 500 mg", "Ibuprofeno 400 mg")),
        Manual("sangrado", "Herida con sangrado", "Trauma", Triage.ADVERTENCIA,
            listOf("sangre", "sangrado", "herida", "cortada", "corte", "laceracion"),
            listOf(
                "En microgravedad la sangre forma gotas que flotan: usa gasas y aspira para no contaminar la cabina.",
                "Presión directa firme 10 minutos sin levantar la gasa.",
                "Si brota a chorros o empapa las gasas: torniquete arriba de la herida y Baliza SOS.",
                "Limpia con solución salina, anestesia local y cierra con bandas o sutura.",
            ), listOf("Lidocaína 2 % (tópica)")),
        Manual("ojo", "Cuerpo extraño en el ojo / visión borrosa", "Oftalmológico", Triage.ADVERTENCIA,
            listOf("ojo", "vision", "veo borroso", "cuerpo extrano", "particula", "borroso"),
            listOf(
                "Las partículas flotan en la cabina: es la lesión ocular más común en la ISS.",
                "No frotes. Lava con solución salina desde el lagrimal hacia afuera.",
                "Visión borrosa de cerca que avanza en semanas: posible síndrome neuro-ocular del vuelo espacial " +
                    "(SANS). Usa los escaneos de ojo de AstroPulse y repórtalo.",
                "Dolor intenso, luz que molesta o pérdida de visión: avisa al médico de vuelo hoy.",
            )),
        Manual("cabeza", "Dolor de cabeza y congestión", "Neurológico", Triage.NOMINAL,
            listOf("dolor de cabeza", "cefalea", "cabeza", "congestion", "nariz tapada"),
            listOf(
                "Es frecuente por el desplazamiento de líquidos hacia la cabeza en microgravedad.",
                "Revisa el CO2 de la cabina: niveles altos dan dolor de cabeza.",
                "Hidrátate y usa el ejercicio resistivo para redistribuir líquidos.",
                "Paracetamol o ibuprofeno. Con vómito, visión doble o confusión: SOS.",
            ), listOf("Paracetamol 500 mg", "Ibuprofeno 400 mg")),
        Manual("rinon", "Cólico renal (piedra en el riñón)", "Urológico", Triage.ADVERTENCIA,
            listOf("rinon", "colico", "dolor de espalda baja", "orina con sangre", "piedra"),
            listOf(
                "Más frecuente en el espacio por la pérdida de calcio del hueso.",
                "Dolor en oleadas de la espalda baja hacia la ingle: hidrátate (2.5 L al día).",
                "Analgesia con ibuprofeno; reporta orina con sangre.",
                "Fiebre con el dolor: posible infección, inicia antibiótico y avisa al médico de vuelo.",
            ), listOf("Ibuprofeno 400 mg", "Ciprofloxacino 500 mg")),
        Manual("anafilaxia", "Reacción alérgica grave", "Emergencia", Triage.CRITICO,
            listOf("alergia", "anafilaxia", "hinchazon de garganta", "ronchas", "urticaria"),
            listOf(
                "Garganta cerrada, falta de aire o desmayo: epinefrina 0.3 mg intramuscular en el muslo YA.",
                "Activa la Baliza SOS. Repite la epinefrina en 5-15 min si no mejora.",
                "Oxígeno y vigila SpO2 y pulso con AstroPulse.",
                "Solo ronchas sin falta de aire: loratadina 10 mg.",
            ), listOf("Epinefrina 0.3 mg (autoinyector)", "Loratadina 10 mg")),
        Manual("anemia", "Anemia espacial (palidez, cansancio, falta de aire al esfuerzo)", "Hematológico", Triage.ADVERTENCIA,
            listOf("anemia", "palido", "palidez", "hemoglobina", "debil"),
            listOf(
                "En el espacio el cuerpo destruye 54 % más glóbulos rojos que en la Tierra y sigue así toda la misión " +
                    "(Trudel et al., Nature Medicine 2022): la anemia es esperable en misiones largas.",
                "Haz el tamizaje con la cámara: párpado inferior (el más validado), palma o encía. Con buena luz blanca.",
                "Palidez + pulso alto en reposo o falta de aire al esfuerzo: el corazón está compensando. Reduce el esfuerzo físico.",
                "Si hay analizador de sangre a bordo, mide hemoglobina: < 12 g/dL (mujer) o < 13 g/dL (hombre) confirma anemia.",
                "Dieta con hierro y vitamina C; evita esfuerzos máximos y caminatas espaciales hasta confirmar.",
                "Mareo al pasar a gravedad (Luna/Marte) con anemia: riesgo de desmayo, siéntate antes de levantarte.",
            )),
        Manual("sueno", "Fatiga y sueño (ciclos de 90 min de día/noche)", "Desempeño", Triage.NOMINAL,
            listOf("cansado", "sueno", "fatiga", "no puedo dormir", "insomnio", "agotado"),
            listOf(
                "En la ISS amanece cada 90 min: usa antifaz y luz azul solo en tu \"mañana\".",
                "Haz el test PVT: más de 355 ms de media o lapsos indica fatiga que aumenta el riesgo de errores.",
                "Antes de una caminata espacial o maniobra crítica con PVT en rojo: pide relevo.",
                "Melatonina 3 mg 30 min antes de dormir; zolpidem solo si hay 8 h libres por delante.",
            ), listOf("Melatonina 3 mg", "Zolpidem 10 mg", "Modafinilo 200 mg")),
        Manual("radiacion", "Evento de partículas solares (tormenta de radiación)", "Radiación", Triage.CRITICO,
            listOf("radiacion", "tormenta solar", "particulas solares", "llamarada", "dosimetro"),
            listOf(
                "Ve al refugio más blindado (módulo con agua/polietileno a tu alrededor).",
                "Quédate ahí hasta que el dosímetro vuelva al nivel de fondo.",
                "Registra la dosis recibida: AstroPulse recalcula la caducidad de los fármacos.",
                "Náusea en las horas siguientes con dosis > 1 Sv: síndrome de radiación aguda, SOS.",
            )),
    )

    fun buscar(texto: String): Manual? {
        val t = normalizar(texto)
        return todos.maxByOrNull { m -> m.claves.count { it in t } * 10 + m.triage.orden }?.takeIf { m -> m.claves.any { it in t } }
    }
}
