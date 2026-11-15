package com.temrarios.ojoscan.platform

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.PermissionController
import androidx.health.connect.client.permission.HealthPermission
import androidx.health.connect.client.records.HeartRateRecord
import androidx.health.connect.client.records.HeartRateVariabilityRmssdRecord
import androidx.health.connect.client.records.OxygenSaturationRecord
import androidx.health.connect.client.records.RestingHeartRateRecord
import androidx.health.connect.client.records.SleepSessionRecord
import androidx.health.connect.client.records.StepsRecord
import kotlinx.coroutines.launch

private val SALUD = listOf(
    HeartRateRecord::class, RestingHeartRateRecord::class, OxygenSaturationRecord::class,
    SleepSessionRecord::class, StepsRecord::class, HeartRateVariabilityRmssdRecord::class,
).map { HealthPermission.getReadPermission(it) }.toSet()

private fun normales(): Array<String> = buildList {
    add(Manifest.permission.CAMERA)
    add(Manifest.permission.RECORD_AUDIO)
    if (Build.VERSION.SDK_INT >= 31) { add(Manifest.permission.BLUETOOTH_SCAN); add(Manifest.permission.BLUETOOTH_CONNECT) }
    else add(Manifest.permission.ACCESS_FINE_LOCATION)
}.toTypedArray()

@Composable
actual fun PermisosAlIniciar(alTerminar: () -> Unit) {
    val ctx = LocalContext.current.applicationContext
    val scope = rememberCoroutineScope()
    val salud = rememberLauncherForActivityResult(PermissionController.createRequestPermissionResultContract()) { alTerminar() }

    fun pedirSalud() {
        if (HealthConnectClient.getSdkStatus(ctx) != HealthConnectClient.SDK_AVAILABLE) { alTerminar(); return }
        scope.launch {
            val dados = runCatching { HealthConnectClient.getOrCreate(ctx).permissionController.getGrantedPermissions() }.getOrDefault(emptySet())
            if (dados.containsAll(SALUD)) alTerminar()
            else runCatching { salud.launch(SALUD) }.onFailure { alTerminar() }
        }
    }

    val sistema = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { pedirSalud() }

    LaunchedEffect(Unit) {
        val faltan = normales().filter { ContextCompat.checkSelfPermission(ctx, it) != PackageManager.PERMISSION_GRANTED }
        if (faltan.isEmpty()) pedirSalud() else sistema.launch(faltan.toTypedArray())
    }
}
