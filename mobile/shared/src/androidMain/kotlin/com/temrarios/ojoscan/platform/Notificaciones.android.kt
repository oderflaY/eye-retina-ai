package com.temrarios.ojoscan.platform

import android.app.AlarmManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.temrarios.ojoscan.astro.EntradaDiario
import com.temrarios.ojoscan.astro.EstadoRevision
import com.temrarios.ojoscan.astro.Registro
import com.temrarios.ojoscan.astro.Seguimiento
import com.temrarios.ojoscan.astro.Triage
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import java.util.Calendar

private const val CANAL_ALERTAS = "alertas_salud"
private const val CANAL_RECORDATORIOS = "recordatorios"

private fun canales(ctx: Context) {
    if (Build.VERSION.SDK_INT < 26) return
    val nm = ctx.getSystemService(NotificationManager::class.java)
    nm.createNotificationChannel(NotificationChannel(CANAL_ALERTAS, "Alertas de salud", NotificationManager.IMPORTANCE_HIGH)
        .apply { description = "Mediciones en rojo o tendencias que empeoran" })
    nm.createNotificationChannel(NotificationChannel(CANAL_RECORDATORIOS, "Recordatorios de revisión", NotificationManager.IMPORTANCE_DEFAULT)
        .apply { description = "Qué revisiones te tocan hoy" })
}

internal fun notificar(ctx: Context, id: Int, titulo: String, texto: String, urgente: Boolean) {
    if (Build.VERSION.SDK_INT >= 33 &&
        ContextCompat.checkSelfPermission(ctx, android.Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) return
    canales(ctx)
    val abrir = ctx.packageManager.getLaunchIntentForPackage(ctx.packageName)?.let {
        PendingIntent.getActivity(ctx, id, it, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
    }
    val n = NotificationCompat.Builder(ctx, if (urgente) CANAL_ALERTAS else CANAL_RECORDATORIOS)
        .setSmallIcon(android.R.drawable.ic_dialog_info)
        .setContentTitle(titulo).setContentText(texto)
        .setStyle(NotificationCompat.BigTextStyle().bigText(texto))
        .setPriority(if (urgente) NotificationCompat.PRIORITY_HIGH else NotificationCompat.PRIORITY_DEFAULT)
        .setAutoCancel(true).apply { abrir?.let { setContentIntent(it) } }
        .build()
    ctx.getSystemService(NotificationManager::class.java).notify(id, n)
}

/** Alarma diaria inexacta (no gasta batería): ≈10:00 de cada día. */
internal fun programar(ctx: Context) {
    val am = ctx.getSystemService(AlarmManager::class.java) ?: return
    val pi = PendingIntent.getBroadcast(ctx, 7001, Intent(ctx, RecordatorioReceiver::class.java),
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
    val c = Calendar.getInstance().apply {
        set(Calendar.HOUR_OF_DAY, 10); set(Calendar.MINUTE, 0); set(Calendar.SECOND, 0)
        if (timeInMillis <= System.currentTimeMillis()) add(Calendar.DAY_OF_YEAR, 1)
    }
    am.setInexactRepeating(AlarmManager.RTC_WAKEUP, c.timeInMillis, AlarmManager.INTERVAL_DAY, pi)
}

/** Lee el historial guardado (mismas claves que el motor) y avisa qué toca y si hay alertas. */
class RecordatorioReceiver : BroadcastReceiver() {
    override fun onReceive(ctx: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED || intent.action == "android.intent.action.MY_PACKAGE_REPLACED") { programar(ctx); return }
        val prefs = ctx.getSharedPreferences("ojoscan", Context.MODE_PRIVATE)
        val json = Json { ignoreUnknownKeys = true }
        val diario = prefs.getString("diario_v1", null)?.let {
            runCatching { json.decodeFromString(ListSerializer(EntradaDiario.serializer()), it) }.getOrNull()
        }.orEmpty().map { Registro(it.horaMs, "diario", "Diario", it.animo?.toDouble(), "", it.texto, it.triage, "diario") }
        val h = (Seguimiento.deJson(prefs.getString("historial_v1", null)) + diario).sortedBy { it.horaMs }
        val ahora = System.currentTimeMillis()
        val tocan = Seguimiento.revisiones(h, ahora).filter { it.estado == EstadoRevision.VENCIDA || it.estado == EstadoRevision.PRONTO }
        val alertas = Seguimiento.alertas(h, ahora).filter { it.triage != Triage.NOMINAL }
        if (alertas.isNotEmpty()) notificar(ctx, 7100, "AstroPulse · ${alertas.size} alerta(s) de salud",
            alertas.take(3).joinToString("\n") { "• ${it.titulo}" }, true)
        when {
            tocan.isNotEmpty() -> notificar(ctx, 7101, "Te toca revisión: " + tocan.take(3).joinToString(", ") { it.tipo.nombre.lowercase() },
                tocan.joinToString("\n") { "• ${it.tipo.nombre}: ${it.texto.lowercase()}. ${it.tipo.como}" }, false)
            h.isEmpty() -> notificar(ctx, 7101, "¿Cómo te sientes hoy?", "Escribe en tu diario de salud y mide tu pulso: así AstroPulse aprende tu línea base.", false)
        }
    }
}

@Composable
actual fun rememberNotificador(): Notificador {
    val ctx = LocalContext.current.applicationContext
    return remember {
        object : Notificador {
            override fun avisar(id: Int, titulo: String, texto: String, urgente: Boolean) = notificar(ctx, id, titulo, texto, urgente)
            override fun programarRecordatorios() = programar(ctx)
        }
    }
}
