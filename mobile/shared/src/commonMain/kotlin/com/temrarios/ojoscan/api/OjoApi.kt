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
    @SerialName("fraccion_conjuntiva") val fraccionConjuntiva: Double = 0.0,
    @SerialName("advertencia_luz") val advertenciaLuz: String? = null,
    val validacion: ValidacionAnemia? = null,
    val aviso: String = "",
    val model: String = "",
)

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
