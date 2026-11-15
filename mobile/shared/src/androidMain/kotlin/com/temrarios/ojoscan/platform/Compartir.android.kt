package com.temrarios.ojoscan.platform

import android.content.Context
import android.content.Intent
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import android.text.StaticLayout
import android.text.TextPaint
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.FileProvider
import com.temrarios.ojoscan.astro.LineaReporte
import com.temrarios.ojoscan.astro.Triage
import java.io.File
import java.nio.ByteBuffer
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.PBEKeySpec
import javax.crypto.spec.SecretKeySpec

/** Respaldo: "ASTROPULSE1" + sal(16) + iv(12) + AES-256-GCM(json). Clave = PBKDF2-HMAC-SHA256, 150 000 vueltas. */
internal object CifradoRespaldo {
    private val MAGIA = "ASTROPULSE1".toByteArray()
    private const val VUELTAS = 150_000

    private fun clave(pass: String, sal: ByteArray) = SecretKeySpec(
        SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256").generateSecret(PBEKeySpec(pass.toCharArray(), sal, VUELTAS, 256)).encoded, "AES")

    fun cifrar(texto: String, pass: String): ByteArray {
        val r = SecureRandom(); val sal = ByteArray(16).also(r::nextBytes); val iv = ByteArray(12).also(r::nextBytes)
        val c = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE, clave(pass, sal), GCMParameterSpec(128, iv)) }
        val datos = c.doFinal(texto.toByteArray())
        return ByteBuffer.allocate(MAGIA.size + 16 + 12 + datos.size).put(MAGIA).put(sal).put(iv).put(datos).array()
    }

    /** null si la contraseña es incorrecta o el archivo no es un respaldo. */
    fun descifrar(b: ByteArray, pass: String): String? = runCatching {
        require(b.size > MAGIA.size + 28 && b.copyOfRange(0, MAGIA.size).contentEquals(MAGIA))
        val sal = b.copyOfRange(MAGIA.size, MAGIA.size + 16); val iv = b.copyOfRange(MAGIA.size + 16, MAGIA.size + 28)
        val c = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.DECRYPT_MODE, clave(pass, sal), GCMParameterSpec(128, iv)) }
        String(c.doFinal(b.copyOfRange(MAGIA.size + 28, b.size)))
    }.getOrNull()
}

/** PDF tamaño carta con texto ajustado y saltos de página. */
internal fun pdfDe(ctx: Context, nombre: String, lineas: List<LineaReporte>): File {
    val doc = PdfDocument()
    val W = 612; val H = 792; val M = 48
    var n = 1
    var pag = doc.startPage(PdfDocument.PageInfo.Builder(W, H, n).create())
    var y = M.toFloat()
    fun color(t: Triage?) = when (t) { Triage.CRITICO -> 0xFFC62828.toInt(); Triage.ADVERTENCIA -> 0xFFB26A00.toInt(); else -> 0xFF222222.toInt() }
    for (l in lineas) {
        val p = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = when (l.tipo) { LineaReporte.Tipo.TITULO -> 20f; LineaReporte.Tipo.SECCION -> 13.5f; LineaReporte.Tipo.NOTA, LineaReporte.Tipo.SUBTITULO -> 9f; else -> 10.5f }
            typeface = if (l.tipo == LineaReporte.Tipo.TITULO || l.tipo == LineaReporte.Tipo.SECCION) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
            color = if (l.tipo == LineaReporte.Tipo.NOTA || l.tipo == LineaReporte.Tipo.SUBTITULO) 0xFF666666.toInt() else color(l.triage)
        }
        val sangria = if (l.tipo == LineaReporte.Tipo.VIÑETA) 14 else 0
        val txt = if (l.tipo == LineaReporte.Tipo.VIÑETA) "• ${l.texto}" else l.texto
        val lay = StaticLayout.Builder.obtain(txt, 0, txt.length, p, W - 2 * M - sangria).setLineSpacing(1.5f, 1f).build()
        val arriba = if (l.tipo == LineaReporte.Tipo.SECCION) 14f else 4f
        if (y + arriba + lay.height > H - M) {
            doc.finishPage(pag); n++
            pag = doc.startPage(PdfDocument.PageInfo.Builder(W, H, n).create()); y = M.toFloat()
        }
        y += arriba
        pag.canvas.save(); pag.canvas.translate((M + sangria).toFloat(), y); lay.draw(pag.canvas); pag.canvas.restore()
        y += lay.height
    }
    doc.finishPage(pag)
    val dir = File(ctx.cacheDir, "compartir").apply { mkdirs() }
    val f = File(dir, nombre)
    f.outputStream().use { doc.writeTo(it) }
    doc.close()
    return f
}

@Composable
actual fun rememberCompartidor(): Compartidor {
    val ctx = LocalContext.current
    // estado de la operación en curso (un solo diálogo a la vez)
    val pendiente = remember { arrayOfNulls<Any>(3) } // [bytes a guardar | clave, callback]
    val guardar = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/octet-stream")) { uri ->
        @Suppress("UNCHECKED_CAST") val listo = pendiente[1] as? (String?) -> Unit
        val bytes = pendiente[0] as? ByteArray
        if (uri == null || bytes == null) listo?.invoke("Cancelado")
        else listo?.invoke(runCatching { ctx.contentResolver.openOutputStream(uri)!!.use { it.write(bytes) } }.exceptionOrNull()?.let { "No se pudo guardar: ${it.message}" })
        pendiente.fill(null)
    }
    val abrir = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        @Suppress("UNCHECKED_CAST") val listo = pendiente[1] as? (String?, String?) -> Unit
        val pass = pendiente[2] as? String ?: ""
        if (uri == null) listo?.invoke(null, "Cancelado")
        else {
            val b = runCatching { ctx.contentResolver.openInputStream(uri)!!.use { it.readBytes() } }.getOrNull()
            val json = b?.let { CifradoRespaldo.descifrar(it, pass) }
            listo?.invoke(json, if (json == null) "Contraseña incorrecta o el archivo no es un respaldo de AstroPulse" else null)
        }
        pendiente.fill(null)
    }
    return remember {
        object : Compartidor {
            override val disponible = true
            override fun compartirPdf(nombre: String, lineas: List<LineaReporte>) {
                val f = pdfDe(ctx, nombre, lineas)
                val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".archivos", f)
                val i = Intent(Intent.ACTION_SEND).setType("application/pdf").putExtra(Intent.EXTRA_STREAM, uri)
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                ctx.startActivity(Intent.createChooser(i, "Compartir resumen de salud").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            }

            override fun guardarRespaldo(nombre: String, json: String, clave: String, listo: (String?) -> Unit) {
                pendiente[0] = CifradoRespaldo.cifrar(json, clave); pendiente[1] = listo
                guardar.launch(nombre)
            }

            override fun abrirRespaldo(clave: String, listo: (String?, String?) -> Unit) {
                pendiente[1] = listo; pendiente[2] = clave
                abrir.launch(arrayOf("*/*"))
            }
        }
    }
}
