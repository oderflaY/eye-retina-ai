package com.temrarios.ojoscan.platform

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.toComposeImageBitmap
import androidx.compose.ui.unit.dp
import com.temrarios.ojoscan.guia.Lectura
import com.temrarios.ojoscan.guia.Modo
import com.temrarios.ojoscan.guia.Ojo
import platform.AVFAudio.AVSpeechBoundary
import platform.AVFAudio.AVSpeechSynthesisVoice
import platform.AVFAudio.AVSpeechSynthesizer
import platform.AVFAudio.AVSpeechUtterance

/**
 * Pendiente en iOS: la guía en vivo necesita AVFoundation (cámara) + Vision
 * (VNDetectFaceLandmarksRequest) para medir rostro, ángulos y apertura del ojo, y llenar la misma
 * [Lectura] que usa el motor común. Por ahora la pantalla lo avisa en lugar de fallar.
 */
@Composable
actual fun CamaraOjo(
    modifier: Modifier,
    ojo: Ojo,
    modo: Modo,
    trasera: Boolean,
    pedirFoto: Int,
    alLeer: (Lectura) -> Unit,
    alCapturar: (Captura) -> Unit,
    alError: (String) -> Unit,
) {
    LaunchedEffect(Unit) { alError("El escáner con guía en vivo aún no está disponible en iPhone") }
    Box(modifier.background(Color.Black), contentAlignment = Alignment.Center) {
        Text(
            "La guía en vivo (abre el ojo, no te muevas…) por ahora solo funciona en Android.",
            color = Color.White,
            modifier = Modifier.padding(24.dp),
        )
    }
}

@Composable
actual fun rememberVoz(): Voz = remember { VozIos() }

private class VozIos : Voz {
    private val sintetizador = AVSpeechSynthesizer()

    override fun decir(texto: String) {
        sintetizador.stopSpeakingAtBoundary(AVSpeechBoundary.AVSpeechBoundaryImmediate)
        val frase = AVSpeechUtterance.speechUtteranceWithString(texto)
        frase.voice = AVSpeechSynthesisVoice.voiceWithLanguage("es-MX")
        sintetizador.speakUtterance(frase)
    }

    override fun callar() {
        sintetizador.stopSpeakingAtBoundary(AVSpeechBoundary.AVSpeechBoundaryImmediate)
    }
}

actual fun decodificarImagen(bytes: ByteArray): ImageBitmap? =
    runCatching { org.jetbrains.skia.Image.makeFromEncoded(bytes).toComposeImageBitmap() }.getOrNull()
