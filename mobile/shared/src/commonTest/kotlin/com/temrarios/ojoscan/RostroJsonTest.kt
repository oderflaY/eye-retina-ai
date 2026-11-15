package com.temrarios.ojoscan

import com.temrarios.ojoscan.api.AnalisisRostro
import kotlinx.serialization.json.Json
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** La respuesta real de /predict/rostro (caso con parálisis) se lee sin errores. */
class RostroJsonTest {
    private val json = Json { ignoreUnknownKeys = true; explicitNulls = false }

    @Test
    fun leeRespuestaReal() {
        val a = json.decodeFromString(AnalisisRostro.serializer(), "{\"cara\": true, \"de_frente\": true, \"avisos_foto\": [], \"hallazgos\": [{\"clave\": \"paralisis_facial\", \"nombre\": \"Asimetría / parálisis facial\", \"probabilidad\": 0.9989, \"resultado\": \"posible\", \"detalle\": {\"comisuras\": 0.0866, \"parpados\": 0.4376, \"cejas\": 0.0018}, \"validacion\": {\"auc_por_persona\": 0.9529, \"sensibilidad\": 0.7778, \"especificidad\": 1.0, \"falsas_alarmas_caras_sanas\": 0.0684, \"personas\": 49}}, {\"clave\": \"ictericia\", \"nombre\": \"Ictericia (ojos amarillos)\", \"probabilidad\": 0.0525, \"resultado\": \"sin_signos\", \"detalle\": {\"amarillo_esclerotica_b\": 6.92, \"ojo_vs_piel\": -3.8}, \"validacion\": {\"auc\": 0.9579, \"sensibilidad\": 0.825, \"especificidad\": 0.9618, \"falsas_alarmas_caras_sanas\": 0.0381, \"casos\": 40}}], \"ojos\": {\"derecho\": {\"principal\": \"normal\", \"titulo\": \"Sin hallazgos visibles\", \"confianza\": 0.9992}, \"izquierdo\": {\"principal\": \"normal\", \"titulo\": \"Sin hallazgos visibles\", \"confianza\": 0.9986}}, \"combinado\": [{\"nivel\": \"alta\", \"texto\": \"Un lado de la cara se mueve menos que el otro y un párpado cierra distinto. Si empezó de golpe, ve a urgencias YA (descartar infarto cerebral). En parálisis de Bell el ojo que no cierra se reseca: lubrícalo y protégelo.\"}], \"caja_cara\": [0.2728927433490753, 0.38831984996795654, 0.42336198687553406, 0.47830331325531006], \"aviso\": \"Tamizaje orientativo hecho con pocos datos públicos; no es diagnóstico. Si la cara se te desvió de repente, se te cae un lado de la boca, no puedes cerrar un ojo, o hay debilidad en un brazo o dificultad para hablar, llama a emergencias (911) YA: puede ser un infarto cerebral.\"}")
        assertTrue(a.cara)
        assertEquals("posible", a.hallazgos.first { it.clave == "paralisis_facial" }.resultado)
        assertTrue(a.combinado.any { it.nivel == "alta" })
        assertEquals(2, a.ojos.size)
    }

    @Test
    fun leeSinCara() {
        val a = json.decodeFromString(AnalisisRostro.serializer(), "{\"cara\": false, \"mensaje\": \"No encontré una cara\"}")
        assertTrue(!a.cara)
        assertTrue(a.hallazgos.isEmpty())
    }
}
