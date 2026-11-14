package com.temrarios.ojoscan

import androidx.compose.ui.window.ComposeUIViewController
import platform.Foundation.NSBundle

/** Pantalla de Compose para iosApp (iOSApp.swift). El servidor sale de Info.plist (API_BASE_URL). */
fun MainViewController() = ComposeUIViewController {
    val servidor = NSBundle.mainBundle.objectForInfoDictionaryKey("API_BASE_URL") as? String
    App(servidor?.takeIf { it.isNotBlank() } ?: "https://2.25.68.28:13443")
}
