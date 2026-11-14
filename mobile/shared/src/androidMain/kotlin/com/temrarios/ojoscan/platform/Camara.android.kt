package com.temrarios.ojoscan.platform

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.speech.tts.TextToSpeech
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.temrarios.ojoscan.guia.Modo
import com.temrarios.ojoscan.guia.Ojo
import java.util.Locale
import java.util.concurrent.Executors

@Composable
actual fun CamaraOjo(
    modifier: Modifier,
    ojo: Ojo,
    modo: Modo,
    trasera: Boolean,
    pedirFoto: Int,
    alLeer: (com.temrarios.ojoscan.guia.Lectura) -> Unit,
    alCapturar: (Captura) -> Unit,
    alError: (String) -> Unit,
) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val leer by rememberUpdatedState(alLeer)
    val capturar by rememberUpdatedState(alCapturar)
    val error by rememberUpdatedState(alError)
    var permitido by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        permitido = ok
        if (!ok) error("Sin permiso de cámara. Actívalo en Ajustes → Apps → OjoScan.")
    }
    LaunchedEffect(Unit) { if (!permitido) launcher.launch(Manifest.permission.CAMERA) }
    if (!permitido) return

    val vista = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    var tomaFoto by remember { mutableStateOf<ImageCapture?>(null) }
    val hiloFoto = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(owner, trasera) {
        val principal = ContextCompat.getMainExecutor(context)
        val hilo = Executors.newSingleThreadExecutor()
        val giroscopio = Giroscopio(context).also { it.iniciar() }
        val analizador = AnalizadorOjo(context, giroscopio::magnitud) { l -> principal.execute { leer(l) } }
        val futuro = ProcessCameraProvider.getInstance(context)
        var proveedor: ProcessCameraProvider? = null
        futuro.addListener({
            try {
                val p = futuro.get()
                proveedor = p
                val preview = Preview.Builder().build().also { it.surfaceProvider = vista.surfaceProvider }
                val analisis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                            ).build()
                    )
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(hilo, analizador) }
                val foto = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                            .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                            .build()
                    )
                    .build()
                val selector = if (trasera) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA
                p.unbindAll()
                p.bindToLifecycle(owner, selector, preview, analisis, foto)
                tomaFoto = foto
            } catch (e: Exception) {
                error("No se pudo abrir la cámara ${if (trasera) "trasera" else "frontal"}.")
            }
        }, principal)
        onDispose {
            proveedor?.unbindAll()
            tomaFoto = null
            giroscopio.detener()
            analizador.cerrar()
            hilo.shutdown()
        }
    }

    LaunchedEffect(pedirFoto) {
        if (pedirFoto == 0) return@LaunchedEffect
        val principal = ContextCompat.getMainExecutor(context)
        val foto = tomaFoto ?: run {
            capturar(Captura.Repetir("La cámara aún no está lista, intentemos otra vez"))
            return@LaunchedEffect
        }
        foto.takePicture(hiloFoto, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(img: ImageProxy) {
                val resultado = try {
                    val rot = img.imageInfo.rotationDegrees
                    val bmp = img.toBitmap()
                    img.close()
                    Recortador.procesar(context, bmp, rot, ojo, modo).also { bmp.recycle() }
                } catch (e: Throwable) {
                    img.close()
                    Captura.Repetir("No se pudo procesar la foto, intentemos otra vez")
                }
                principal.execute { capturar(resultado) }
            }

            override fun onError(e: ImageCaptureException) {
                principal.execute { capturar(Captura.Repetir("No se pudo tomar la foto, intentemos otra vez")) }
            }
        })
    }

    DisposableEffect(Unit) { onDispose { hiloFoto.shutdown() } }
    AndroidView(factory = { vista }, modifier = modifier)
}

@Composable
actual fun rememberVoz(): Voz {
    val context = LocalContext.current
    val voz = remember { VozAndroid(context) }
    DisposableEffect(voz) { onDispose { voz.apagar() } }
    return voz
}

private class VozAndroid(context: android.content.Context) : Voz {
    private var lista = false
    private val tts: TextToSpeech = TextToSpeech(context.applicationContext) { estado ->
        if (estado == TextToSpeech.SUCCESS) {
            lista = true
            val mx = Locale.Builder().setLanguage("es").setRegion("MX").build()
            if (tts.isLanguageAvailable(mx) >= TextToSpeech.LANG_AVAILABLE) tts.language = mx
            else tts.language = Locale.Builder().setLanguage("es").build()
        }
    }

    override fun decir(texto: String) {
        if (lista) tts.speak(texto, TextToSpeech.QUEUE_FLUSH, null, "ojoscan")
    }

    override fun callar() {
        tts.stop()
    }

    fun apagar() = tts.shutdown()
}

actual fun decodificarImagen(bytes: ByteArray): ImageBitmap? =
    BitmapFactory.decodeByteArray(bytes, 0, bytes.size)?.asImageBitmap()
