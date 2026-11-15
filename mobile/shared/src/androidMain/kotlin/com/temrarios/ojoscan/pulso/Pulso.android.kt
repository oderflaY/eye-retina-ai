package com.temrarios.ojoscan.pulso

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.ParcelUuid
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.UUID

private val SERVICIO_HR: UUID = UUID.fromString("0000180d-0000-1000-8000-00805f9b34fb")
private val MEDIDA_HR: UUID = UUID.fromString("00002a37-0000-1000-8000-00805f9b34fb")
private val CCCD: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")

private fun permisosBle(): Array<String> =
    if (Build.VERSION.SDK_INT >= 31) arrayOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT)
    else arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)

/** Nombres de pulseras/relojes emparejados que vale la pena probar aunque no se anuncien. */
private fun pareceWearable(n: String?) = n != null && Regex("band|watch|reloj|amazfit|mi smart|redmi|huawei|garmin|polar|fitbit", RegexOption.IGNORE_CASE).containsMatchIn(n)

@SuppressLint("MissingPermission")
private class MonitorBle(private val ctx: Context, private val pedirPermisos: () -> Unit) : MonitorPulso {
    override val estado = MutableStateFlow<EstadoPulso>(EstadoPulso.Inactivo)
    override val muestras = MutableStateFlow<List<Muestra>>(emptyList())
    private val main = Handler(Looper.getMainLooper())
    private val manager = ctx.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager?
    private var gatt: BluetoothGatt? = null
    private var escaneando: ScanCallback? = null
    private var activo = false
    private var candidatos = ArrayDeque<BluetoothDevice>()
    private var nombre = ""

    fun tienePermisos() = permisosBle().all { ContextCompat.checkSelfPermission(ctx, it) == PackageManager.PERMISSION_GRANTED }

    override fun iniciar() {
        if (activo && gatt != null) return
        if (!tienePermisos()) { estado.value = EstadoPulso.SinPermiso; pedirPermisos(); return }
        val adapter = manager?.adapter
        if (adapter == null || !adapter.isEnabled) { estado.value = EstadoPulso.BluetoothApagado; return }
        activo = true
        estado.value = EstadoPulso.Buscando("Buscando tu pulsera…")
        // 1) Pulseras que ya están emparejadas con el teléfono (la Smart Band de Mi Fitness).
        candidatos = ArrayDeque(adapter.bondedDevices.filter { pareceWearable(it.name) })
        // 2) Cualquier aparato que anuncie el servicio de pulso (modo "transmitir frecuencia cardíaca").
        val scanner = adapter.bluetoothLeScanner
        val cb = object : ScanCallback() {
            override fun onScanResult(callbackType: Int, r: ScanResult) {
                if (gatt != null) return
                pararEscaneo()
                conectar(r.device)
            }
            override fun onScanFailed(errorCode: Int) { probarSiguiente() }
        }
        escaneando = cb
        runCatching {
            scanner?.startScan(
                listOf(ScanFilter.Builder().setServiceUuid(ParcelUuid(SERVICIO_HR)).build()),
                ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build(), cb,
            )
        }
        // Si en 6 s nadie anuncia el pulso, se prueba directo con las emparejadas.
        main.postDelayed({ if (activo && gatt == null) { pararEscaneo(); probarSiguiente() } }, 6000)
    }

    private fun pararEscaneo() {
        escaneando?.let { cb -> runCatching { manager?.adapter?.bluetoothLeScanner?.stopScan(cb) } }
        escaneando = null
    }

    private fun probarSiguiente() {
        if (!activo || gatt != null) return
        val d = candidatos.removeFirstOrNull()
        if (d == null) {
            estado.value = if (nombre.isNotEmpty()) EstadoPulso.SinTransmision(nombre)
            else EstadoPulso.Error("No encontré ninguna pulsera. Activa en tu pulsera \"Transmitir frecuencia cardíaca\" y vuelve a intentar.")
            activo = false
            return
        }
        conectar(d)
    }

    private fun conectar(d: BluetoothDevice) {
        nombre = d.name ?: "Pulsera"
        estado.value = EstadoPulso.Conectando(nombre)
        gatt = d.connectGatt(ctx, false, callback, BluetoothDevice.TRANSPORT_LE)
        // Si en 15 s no responde, se suelta y se prueba otra.
        val g = gatt
        main.postDelayed({ if (gatt === g && estado.value is EstadoPulso.Conectando) { soltar(); probarSiguiente() } }, 15000)
    }

    private fun soltar() {
        runCatching { gatt?.disconnect(); gatt?.close() }
        gatt = null
    }

    private val callback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(g: BluetoothGatt, status: Int, newState: Int) {
            main.post {
                if (newState == BluetoothProfile.STATE_CONNECTED) {
                    g.discoverServices()
                } else if (newState == BluetoothProfile.STATE_DISCONNECTED && gatt === g) {
                    soltar()
                    if (activo) {
                        // Se cayó la conexión a media medición: se reintenta con la misma pulsera.
                        if (estado.value is EstadoPulso.Midiendo) { estado.value = EstadoPulso.Conectando(nombre); conectar(g.device) }
                        else probarSiguiente()
                    }
                }
            }
        }

        override fun onServicesDiscovered(g: BluetoothGatt, status: Int) {
            main.post {
                val c = g.getService(SERVICIO_HR)?.getCharacteristic(MEDIDA_HR)
                if (c == null) { soltar(); probarSiguiente(); return@post }
                g.setCharacteristicNotification(c, true)
                val d = c.getDescriptor(CCCD) ?: return@post
                if (Build.VERSION.SDK_INT >= 33) {
                    g.writeDescriptor(d, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)
                } else {
                    @Suppress("DEPRECATION")
                    d.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                    @Suppress("DEPRECATION")
                    g.writeDescriptor(d)
                }
                estado.value = EstadoPulso.Midiendo(nombre, 0, null)
            }
        }

        override fun onCharacteristicChanged(g: BluetoothGatt, c: BluetoothGattCharacteristic, value: ByteArray) = recibir(value)

        @Deprecated("API < 33")
        @Suppress("DEPRECATION")
        override fun onCharacteristicChanged(g: BluetoothGatt, c: BluetoothGattCharacteristic) {
            if (Build.VERSION.SDK_INT < 33) recibir(c.value ?: return)
        }
    }

    private fun recibir(b: ByteArray) {
        val (lpm, contacto, rr) = parsearHrm(b) ?: return
        main.post {
            if (lpm > 0) muestras.value = muestras.value + Muestra(System.currentTimeMillis(), lpm, rr)
            estado.value = EstadoPulso.Midiendo(nombre, lpm, contacto)
        }
    }

    override fun detener() {
        activo = false
        pararEscaneo()
        soltar()
        estado.value = EstadoPulso.Inactivo
    }

    override fun reiniciarSesion() { muestras.value = emptyList() }
}

@Composable
actual fun rememberMonitorPulso(): MonitorPulso {
    val ctx = LocalContext.current.applicationContext
    var monitor: MonitorBle? = null
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { r ->
        if (r.values.all { it }) monitor?.iniciar()
        else monitor?.estado?.value = EstadoPulso.SinPermiso
    }
    val m = remember { MonitorBle(ctx) { launcher.launch(permisosBle()) } }
    monitor = m
    DisposableEffect(Unit) { onDispose { m.detener() } }
    return m
}
