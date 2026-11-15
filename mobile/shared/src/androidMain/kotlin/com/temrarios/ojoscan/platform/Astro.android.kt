package com.temrarios.ojoscan.platform

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.BatteryManager
import android.os.Build
import android.os.StatFs
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.io.File
import java.security.KeyStore
import java.security.MessageDigest
import java.util.concurrent.Executors
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

// ── Bóveda cifrada ───────────────────────────────────────────────────────────

private class BovedaAndroid(ctx: Context) : Boveda {
    private val dir = File(ctx.filesDir, "boveda").apply { mkdirs() }
    private val alias = "astropulse_boveda"
    override val disponible = true

    private fun llave(): SecretKey {
        val ks = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (ks.getEntry(alias, null) as? KeyStore.SecretKeyEntry)?.let { return it.secretKey }
        val gen = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        gen.init(KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256).build())
        return gen.generateKey()
    }

    private fun huella(b: ByteArray) = MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }.take(16)

    override fun guardar(nombre: String, datos: ByteArray): ArchivoBoveda {
        val c = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE, llave()) }
        val cifrado = c.iv + c.doFinal(datos) // [12 bytes IV][texto cifrado + etiqueta GCM]
        val f = File(dir, nombre)
        f.writeBytes(cifrado)
        return ArchivoBoveda(nombre, f.length(), f.lastModified(), huella(cifrado))
    }

    override fun leer(nombre: String): ByteArray? = runCatching {
        val b = File(dir, nombre).readBytes()
        Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.DECRYPT_MODE, llave(), GCMParameterSpec(128, b, 0, 12)) }
            .doFinal(b, 12, b.size - 12)
    }.getOrNull()

    override fun listar(prefijo: String) = (dir.listFiles() ?: emptyArray()).filter { it.name.startsWith(prefijo) }
        .sortedByDescending { it.lastModified() }
        .map { ArchivoBoveda(it.name, it.length(), it.lastModified(), huella(it.readBytes())) }

    override fun borrar(nombre: String) { File(dir, nombre).delete() }
}

@Composable
actual fun rememberBoveda(): Boveda {
    val ctx = LocalContext.current.applicationContext
    return remember { BovedaAndroid(ctx) }
}

// ── Grabadora de voz ─────────────────────────────────────────────────────────

private class GrabadoraAndroid(private val ctx: Context) : Grabadora {
    private var rec: MediaRecorder? = null
    private var tmp: File? = null
    private var player: MediaPlayer? = null
    override val disponible = ctx.packageManager.hasSystemFeature(PackageManager.FEATURE_MICROPHONE)

    override fun iniciar(): Boolean {
        if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return false
        return runCatching {
            val f = File.createTempFile("voz", ".m4a", ctx.cacheDir)
            val r = if (Build.VERSION.SDK_INT >= 31) MediaRecorder(ctx) else @Suppress("DEPRECATION") MediaRecorder()
            r.setAudioSource(MediaRecorder.AudioSource.MIC)
            r.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            r.setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            r.setAudioEncodingBitRate(64_000); r.setAudioSamplingRate(22_050)
            r.setOutputFile(f.absolutePath)
            r.prepare(); r.start()
            rec = r; tmp = f
            true
        }.getOrDefault(false)
    }

    override fun nivel(): Float = runCatching { (rec?.maxAmplitude ?: 0) / 20_000f }.getOrDefault(0f).coerceIn(0f, 1f)

    override fun detener(): ByteArray? {
        val r = rec ?: return null
        rec = null
        runCatching { r.stop() }
        r.release()
        val f = tmp ?: return null
        tmp = null
        return runCatching { f.readBytes() }.getOrNull().also { f.delete() } // el audio en claro no se queda en disco
    }

    override fun reproducir(audio: ByteArray, alTerminar: () -> Unit) {
        pararReproduccion()
        val f = File.createTempFile("rep", ".m4a", ctx.cacheDir).apply { writeBytes(audio) }
        player = MediaPlayer().apply {
            setDataSource(f.absolutePath)
            setOnCompletionListener { f.delete(); alTerminar() }
            prepare(); start()
        }
    }

    override fun pararReproduccion() { player?.runCatching { stop(); release() }; player = null }
}

@Composable
actual fun rememberGrabadora(): Grabadora {
    val ctx = LocalContext.current.applicationContext
    val g = remember { GrabadoraAndroid(ctx) }
    DisposableEffect(Unit) { onDispose { g.detener(); g.pararReproduccion() } }
    return g
}

// ── Acelerómetro ─────────────────────────────────────────────────────────────

@Composable
actual fun EscuchaAcelerometro(activo: Boolean, alMuestra: (x: Double, y: Double, z: Double) -> Unit) {
    val ctx = LocalContext.current
    val cb by rememberUpdatedState(alMuestra)
    DisposableEffect(activo) {
        val sm = ctx.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val s = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val l = object : SensorEventListener {
            override fun onSensorChanged(e: SensorEvent) = cb(e.values[0].toDouble(), e.values[1].toDouble(), e.values[2].toDouble())
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }
        if (activo && s != null) sm.registerListener(l, s, SensorManager.SENSOR_DELAY_GAME)
        onDispose { sm.unregisterListener(l) }
    }
}


// ── Cámara PPG ───────────────────────────────────────────────────────────────

@Composable
actual fun CamaraPpg(modifier: Modifier, alCuadro: (horaMs: Long, rojo: Double, verde: Double) -> Unit) {
    val ctx = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val cb by rememberUpdatedState(alCuadro)
    val vista = remember { PreviewView(ctx) }
    AndroidView({ vista }, modifier)
    DisposableEffect(Unit) {
        val hilo = Executors.newSingleThreadExecutor()
        val principal = ContextCompat.getMainExecutor(ctx)
        var proveedor: ProcessCameraProvider? = null
        if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            val futuro = ProcessCameraProvider.getInstance(ctx)
            futuro.addListener({
                runCatching {
                    val p = futuro.get(); proveedor = p
                    val preview = Preview.Builder().build().also { it.surfaceProvider = vista.surfaceProvider }
                    val analisis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                    analisis.setAnalyzer(hilo) { img ->
                        val plano = img.planes[0]
                        val buf = plano.buffer
                        val w = img.width; val h = img.height; val paso = plano.rowStride
                        var r = 0L; var g = 0L; var n = 0
                        // centro de la imagen, una de cada 8 filas/columnas
                        var y = h / 4
                        while (y < 3 * h / 4) {
                            var x = w / 4
                            while (x < 3 * w / 4) {
                                val i = y * paso + x * 4
                                r += buf.get(i).toInt() and 0xFF; g += buf.get(i + 1).toInt() and 0xFF; n++
                                x += 8
                            }
                            y += 8
                        }
                        val t = System.currentTimeMillis()
                        img.close()
                        if (n > 0) principal.execute { cb(t, r.toDouble() / n, g.toDouble() / n) }
                    }
                    p.unbindAll()
                    val cam = p.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analisis)
                    cam.cameraControl.enableTorch(true)
                }
            }, principal)
        }
        onDispose { proveedor?.unbindAll(); hilo.shutdown() }
    }
}

// ── Dispositivo ──────────────────────────────────────────────────────────────

private class DispositivoAndroid(private val ctx: Context) : Dispositivo {
    private fun estado() = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    override fun bateria(): Int? = (ctx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager)
        .getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY).takeIf { it in 0..100 }
    override fun cargando(): Boolean = estado()?.getIntExtra(BatteryManager.EXTRA_STATUS, -1).let {
        it == BatteryManager.BATTERY_STATUS_CHARGING || it == BatteryManager.BATTERY_STATUS_FULL }
    override fun libreMb(): Long? = runCatching { StatFs(ctx.filesDir.path).availableBytes / (1024 * 1024) }.getOrNull()
    override fun modelo() = "${Build.MANUFACTURER} ${Build.MODEL} · Android ${Build.VERSION.RELEASE}"
    override fun sensores(): List<Pair<String, Boolean>> {
        val sm = ctx.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val pm = ctx.packageManager
        return listOf(
            "Acelerómetro (IMU)" to (sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null),
            "Giroscopio" to (sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE) != null),
            "Cámara + flash (PPG)" to pm.hasSystemFeature(PackageManager.FEATURE_CAMERA_FLASH),
            "Micrófono (bitácora)" to pm.hasSystemFeature(PackageManager.FEATURE_MICROPHONE),
            "Bluetooth LE (pulsera)" to pm.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE),
            "Llavero seguro (cifrado)" to true,
        )
    }
}

@Composable
actual fun rememberDispositivo(): Dispositivo {
    val ctx = LocalContext.current.applicationContext
    return remember { DispositivoAndroid(ctx) }
}

@Composable
actual fun rememberLectorAssets(): LectorAssets {
    val ctx = LocalContext.current.applicationContext
    return remember {
        object : LectorAssets {
            override fun leer(nombre: String): String? =
                runCatching { ctx.assets.open(nombre).bufferedReader(Charsets.UTF_8).use { it.readText() } }.getOrNull()
        }
    }
}

actual fun desfaseHorarioMs(horaMs: Long): Long = java.util.TimeZone.getDefault().getOffset(horaMs).toLong()
