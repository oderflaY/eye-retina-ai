package com.temrarios.ojoscan.api

import io.ktor.client.HttpClient
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.request.forms.formData
import io.ktor.client.request.forms.submitFormWithBinaryData
import io.ktor.client.request.get
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.http.Headers
import io.ktor.http.HttpHeaders
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** Respuestas del servidor Eye-Retina AI (src/deployment/api.py). */
@Serializable
data class Punto(val name: String, val x: Double, val y: Double)

@Serializable
data class PrediccionOjo(
    /** x, y, ancho, alto en píxeles de la imagen enviada. */
    val bbox: List<Double>,
    val landmarks: List<Punto> = emptyList(),
    val confidence: Double,
    @SerialName("is_eye") val esOjo: Boolean,
    @SerialName("image_size") val tamano: List<Int> = emptyList(),
    val model: String = "",
)

@Serializable
data class Hallazgo(val clave: String, val nombre: String, val probabilidad: Double, @SerialName("auc_test") val aucTest: Double? = null)

/** Respuesta de POST /predict/eye/analysis. */
@Serializable
data class AnalisisOjo(
    val principal: String,
    val titulo: String,
    val descripcion: String = "",
    val confianza: Double,
    val hallazgos: List<Hallazgo>,
    val aviso: String = "",
    val model: String = "",
)

/** Validación cruzada del clasificador de anemia (para mostrar qué tan fiable es). */
@Serializable
data class ValidacionAnemia(
    @SerialName("auc_cv") val auc: Double? = null,
    val sensibilidad: Double? = null,
    val especificidad: Double? = null,
    val n: Int? = null,
    /** Fracción de fotos que reciben veredicto (fuera de la zona gris). */
    val cobertura: Double? = null,
)

/** Respuesta de POST /predict/anemia. */
@Serializable
data class AnalisisAnemia(
    val valida: Boolean,
    val resultado: String,
    val titulo: String,
    val descripcion: String = "",
    @SerialName("probabilidad_anemia") val probabilidad: Double? = null,
    val umbral: Double? = null,
    /** [bajo, alto]: entre estas probabilidades el resultado es «no concluyente». */
    @SerialName("zona_gris") val zonaGris: List<Double>? = null,
    @SerialName("fraccion_conjuntiva") val fraccionConjuntiva: Double = 0.0,
    @SerialName("advertencia_luz") val advertenciaLuz: String? = null,
    val validacion: ValidacionAnemia? = null,
    val aviso: String = "",
    val model: String = "",
    /** Método sin validar con hemoglobina (encía): solo apoya, no decide. */
    val experimental: Boolean = false,
    /** Encía muy roja: posible gingivitis (oculta la palidez). */
    val inflamacion: Boolean = false,
)

@Serializable
data class HallazgoRostro(
    val clave: String,
    val nombre: String,
    val probabilidad: Double,
    val resultado: String,
    val validacion: Map<String, Double> = emptyMap(),
)

@Serializable
data class OjoEnRostro(val principal: String, val titulo: String, val confianza: Double)

@Serializable
data class NotaCombinada(val nivel: String, val texto: String)

@Serializable
data class AnalisisRostro(
    val cara: Boolean,
    val mensaje: String? = null,
    val error: String? = null,
    @SerialName("de_frente") val deFrente: Boolean = true,
    @SerialName("avisos_foto") val avisosFoto: List<String> = emptyList(),
    val hallazgos: List<HallazgoRostro> = emptyList(),
    val ojos: Map<String, OjoEnRostro> = emptyMap(),
    val combinado: List<NotaCombinada> = emptyList(),
    @SerialName("caja_cara") val cajaCara: List<Double>? = null,
    val aviso: String = "",
)

/** Lo que el paciente escribe. Se guarda en el teléfono; al servidor se manda sin el nombre. */
@Serializable
data class DatosPerfil(
    val nombre: String = "",
    val edad: Int? = null,
    /** "M" o "F". */
    val sexo: String? = null,
    @SerialName("peso_kg") val peso: Double? = null,
    @SerialName("estatura_cm") val estatura: Double? = null,
    @SerialName("cuello_cm") val cuello: Double? = null,
    /** Cintura a la altura del ombligo. */
    @SerialName("abdomen_cm") val abdomen: Double? = null,
    @SerialName("cadera_cm") val cadera: Double? = null,
    @SerialName("presion_sistolica") val sistolica: Double? = null,
    @SerialName("presion_diastolica") val diastolica: Double? = null,
    /** ninguno · tipico · atipico · no_anginoso */
    @SerialName("dolor_pecho") val dolorPecho: String = "ninguno",
    @SerialName("dolor_con_ejercicio") val dolorEjercicio: Boolean = false,
    val colesterol: Double? = null,
    @SerialName("glucosa_ayunas") val glucosa: Double? = null,
    @SerialName("fc_maxima") val fcMaxima: Double? = null,
    /** Pulso medido EN VIVO por la pulsera durante el chequeo (no se guarda en el teléfono). */
    @SerialName("pulso_reposo") val pulsoReposo: Int? = null,
    @SerialName("pulso_min") val pulsoMin: Int? = null,
    @SerialName("pulso_max") val pulsoMax: Int? = null,
    @SerialName("pulso_segundos") val pulsoSegundos: Int? = null,
    @SerialName("pulso_rmssd") val pulsoRmssd: Double? = null,
    @SerialName("pulso_irregular") val pulsoIrregular: Double? = null,
    /** Últimos 30 días de la pulsera (Mi Fitness → Health Connect), ya resumidos en el teléfono. */
    val historial: HistorialPerfil? = null,
) {
    /** Hay algo que analizar (sin edad y sexo casi nada se puede calcular). */
    val util: Boolean get() = edad != null && sexo != null
}

@Serializable
data class HistorialPerfil(
    val dias: Int,
    @SerialName("pulso_reposo") val pulsoReposo: Double? = null,
    @SerialName("pulso_reposo_7d") val pulsoReposo7d: Double? = null,
    @SerialName("pulso_reposo_base") val pulsoReposoBase: Double? = null,
    @SerialName("pulso_max") val pulsoMax: Int? = null,
    @SerialName("spo2_media") val spo2Media: Double? = null,
    @SerialName("spo2_min") val spo2Min: Double? = null,
    @SerialName("spo2_bajas") val spo2Bajas: Double? = null,
    @SerialName("spo2_n") val spo2N: Int = 0,
    @SerialName("sueno_horas") val suenoHoras: Double? = null,
    @SerialName("sueno_noches") val suenoNoches: Int = 0,
    @SerialName("sueno_cortas") val suenoCortas: Double? = null,
    @SerialName("sueno_profundo") val suenoProfundo: Double? = null,
    val pasos: Double? = null,
    @SerialName("pasos_dias") val pasosDias: Int = 0,
    val vfc: Double? = null,
)

@Serializable
data class IndicadorPerfil(
    val clave: String,
    val nombre: String,
    val valor: String,
    /** ok · media · alta */
    val nivel: String,
    val texto: String,
    val fuente: String = "",
    val probabilidad: Double? = null,
)

/** Respuesta de POST /predict/perfil. */
@Serializable
data class AnalisisPerfil(
    val indicadores: List<IndicadorPerfil> = emptyList(),
    val faltan: List<String> = emptyList(),
    val aviso: String = "",
)

/** Lo que se manda a POST /predict/corazon: pulso a 1 Hz de la pulsera + historial + síntomas. */
@Serializable
data class PeticionCorazon(
    val serie: List<Int>,
    val edad: Int? = null,
    val sexo: String? = null,
    /** reposo · movimiento */
    val actividad: String = "reposo",
    val sintomas: List<String> = emptyList(),
    val historial: HistorialCorazon? = null,
)

/** Mismo historial de 30 días, con los nombres que espera el analizador cardiaco. */
@Serializable
data class HistorialCorazon(
    val pulsoReposo: Double? = null, val pulsoReposo7d: Double? = null, val pulsoReposoBase: Double? = null,
    val spo2Media: Double? = null, val spo2Min: Double? = null, val spo2Bajas: Double? = null, val suenoHoras: Double? = null,
)

@Serializable
data class ValidacionIa(val auc: Double = 0.0, val sensibilidad: Double = 0.0, val especificidad: Double = 0.0, val personas: Int = 0)

@Serializable
data class SalidaIa(
    val clave: String = "", val nombre: String = "", val probabilidad: Double, val nivel: String,
    val ventanas: Int = 0, val validacion: ValidacionIa = ValidacionIa(),
)

@Serializable
data class IaCorazon(val ventanas: Int = 0, val modelos: List<SalidaIa> = emptyList())

@Serializable
data class AhoraCorazon(val lpm: Int, val promedio: Int, val minimo: Int, val maximo: Int, val segundos: Int)

@Serializable
data class AlertaCorazon(val nivel: String, val titulo: String, val texto: String, val accion: String)

@Serializable
data class ComparacionCorazon(val clave: String, val nombre: String, val tuyo: String, val referencia: String, val nivel: String, val texto: String)

/** Respuesta de POST /predict/corazon. riesgo: ok · vigilar · alto · urgente · sin_datos */
@Serializable
data class AnalisisCorazon(
    val riesgo: String,
    val titulo: String,
    val ahora: AhoraCorazon? = null,
    val ia: IaCorazon? = null,
    val comparacion: List<ComparacionCorazon> = emptyList(),
    val alertas: List<AlertaCorazon> = emptyList(),
    val aviso: String = "",
)

fun DatosPerfil.aJson(): String = json.encodeToString(DatosPerfil.serializer(), this)
fun perfilDeJson(texto: String?): DatosPerfil? =
    texto?.let { runCatching { json.decodeFromString(DatosPerfil.serializer(), it) }.getOrNull() }

@Serializable
data class EstadoModelo(val path: String = "", val available: Boolean = false)

@Serializable
data class Salud(
    val status: String,
    val version: String = "",
    val models: Map<String, EstadoModelo> = emptyMap(),
    /** Solo a través del intermediario del VPS: si la PC con la IA está conectada (null = servidor directo). */
    @SerialName("pc_conectada") val pcConectada: Boolean? = null,
    @SerialName("capturas_pendientes") val capturasPendientes: Int = 0,
)

@Serializable
data class CapturaGuardada(val id: String, val image: String)

@Serializable
private data class CuerpoError(val error: DetalleError)

@Serializable
private data class DetalleError(val code: Int, val message: String)

/** Error del servidor ya traducido a algo que se le puede mostrar a la persona. */
class ErrorApi(val codigo: Int, mensaje: String) : Exception(mensaje)

/** Lo que la app sabe de un ojo recortado, para guardarlo como dato de entrenamiento. */
@Serializable
data class MetaCaptura(
    val ojo: String,
    /** Caja del ojo dentro del recorte: x, y, ancho, alto (px). */
    val caja: List<Float>?,
    /** Puntos del ojo dentro del recorte (px), con los nombres de docs/FORMATO_ETIQUETAS.md. */
    val puntos: Map<String, List<Float>>,
    val apertura: Float?,
    val consentimiento: Boolean,
    val origen: String = "ojoscan",
)

private val json = Json { ignoreUnknownKeys = true; explicitNulls = false }

class OjoApi(baseUrl: String) {
    private val base = baseUrl.trimEnd('/')
    private val http = HttpClient {
        install(HttpTimeout) {
            connectTimeoutMillis = 5_000
            requestTimeoutMillis = 75_000
        }
    }

    suspend fun salud(): Salud = leer(http.get("$base/"))

    /** Envía el recorte del ojo al detector (POST /predict/eye). */
    suspend fun predecirOjo(jpeg: ByteArray, umbral: Double = 0.5): PrediccionOjo =
        leer(http.submitFormWithBinaryData("$base/predict/eye?threshold=$umbral", formData { archivo(jpeg) }))

    /** Analiza la foto del ojo (POST /predict/eye/analysis). */
    suspend fun analizarOjo(jpeg: ByteArray): AnalisisOjo =
        leer(http.submitFormWithBinaryData("$base/predict/eye/analysis", formData { archivo(jpeg) }))

    /** Tamizaje de anemia por palidez de la conjuntiva (POST /predict/anemia). */
    suspend fun analizarAnemia(jpeg: ByteArray): AnalisisAnemia =
        leer(http.submitFormWithBinaryData("$base/predict/anemia", formData { archivo(jpeg) }))

    /** Anemia por la palidez de la palma (POST /predict/palma); misma respuesta que la del párpado. */
    suspend fun analizarPalma(jpeg: ByteArray): AnalisisAnemia =
        leer(http.submitFormWithBinaryData("$base/predict/palma", formData { archivo(jpeg) }))

    /** Anemia por la encía, EXPERIMENTAL (POST /predict/encia); misma respuesta que la del párpado. */
    suspend fun analizarEncia(jpeg: ByteArray): AnalisisAnemia =
        leer(http.submitFormWithBinaryData("$base/predict/encia", formData { archivo(jpeg) }))

    /** Rostro completo: asimetría (parálisis), ictericia y los ojos de la misma foto (POST /predict/rostro). */
    suspend fun analizarRostro(jpeg: ByteArray): AnalisisRostro =
        leer(http.submitFormWithBinaryData("$base/predict/rostro", formData { archivo(jpeg) }))

    /** Perfil médico con los datos escritos (POST /predict/perfil). Va como archivo para pasar por el intermediario. */
    suspend fun analizarPerfil(d: DatosPerfil): AnalisisPerfil {
        val cuerpo = d.copy(nombre = "").aJson().encodeToByteArray()
        return leer(
            http.submitFormWithBinaryData(
                "$base/predict/perfil",
                formData {
                    append("file", cuerpo, Headers.build {
                        append(HttpHeaders.ContentType, "application/json")
                        append(HttpHeaders.ContentDisposition, "filename=\"datos.json\"")
                    })
                },
            )
        )
    }

    /** Vigilancia cardiaca con la pulsera (POST /predict/corazon), como archivo para pasar por el intermediario. */
    suspend fun analizarCorazon(p: PeticionCorazon): AnalisisCorazon {
        val cuerpo = json.encodeToString(PeticionCorazon.serializer(), p).encodeToByteArray()
        return leer(
            http.submitFormWithBinaryData(
                "$base/predict/corazon",
                formData {
                    append("file", cuerpo, Headers.build {
                        append(HttpHeaders.ContentType, "application/json")
                        append(HttpHeaders.ContentDisposition, "filename=\"corazon.json\"")
                    })
                },
            )
        )
    }

    /** Guarda el recorte y sus puntos como dato para entrenar (POST /captures/eye). */
    suspend fun guardarCaptura(jpeg: ByteArray, meta: MetaCaptura): CapturaGuardada =
        leer(
            http.submitFormWithBinaryData(
                "$base/captures/eye",
                formData {
                    archivo(jpeg)
                    append("meta", json.encodeToString(MetaCaptura.serializer(), meta))
                },
            )
        )

    private fun io.ktor.client.request.forms.FormBuilder.archivo(jpeg: ByteArray) =
        append(
            "file", jpeg,
            Headers.build {
                append(HttpHeaders.ContentType, "image/jpeg")
                append(HttpHeaders.ContentDisposition, "filename=\"ojo.jpg\"")
            },
        )

    private suspend inline fun <reified T> leer(r: HttpResponse): T {
        val texto = r.bodyAsText()
        if (r.status.value !in 200..299) {
            val msg = runCatching { json.decodeFromString(CuerpoError.serializer(), texto).error.message }
                .getOrDefault("Error ${r.status.value}")
            throw ErrorApi(r.status.value, msg)
        }
        return json.decodeFromString(texto)
    }

    fun cerrar() = http.close()
}

/** Mensaje claro para la persona según el tipo de error. */
fun explicar(e: Throwable): String = when {
    e is ErrorApi && e.codigo == 503 && e.message?.contains("Modelo no disponible") == true ->
        "El sistema recibió la foto, pero el modelo de análisis no está cargado en la computadora."
    e is ErrorApi && e.codigo == 413 -> "La foto es demasiado grande para el servidor."
    e is ErrorApi -> e.message ?: "El servidor respondió con un error (${e.codigo})."
    else -> "No se pudo conectar con el servidor. Revisa tu conexión a internet y la dirección del servidor."
}
