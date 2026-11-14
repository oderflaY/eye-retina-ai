package com.temrarios.ojoscan

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import com.temrarios.ojoscan.guia.Modo
import com.temrarios.ojoscan.guia.Ojo
import com.temrarios.ojoscan.platform.Recorte
import com.temrarios.ojoscan.ui.PantallaEscaner
import com.temrarios.ojoscan.ui.PantallaInicio
import com.temrarios.ojoscan.ui.PantallaResultado
import com.temrarios.ojoscan.ui.TemaOjo

/** Opciones que la persona elige en el inicio. */
data class Ajustes(
    val servidor: String,
    val modo: Modo = Modo.OJO,
    val ojo: Ojo = Ojo.IZQUIERDO,
    val trasera: Boolean = false,
    val voz: Boolean = true,
    /** Guardar las fotos en el servidor para entrenar el modelo (requiere consentimiento). */
    val guardar: Boolean = false,
)

private sealed interface Pantalla {
    data object Inicio : Pantalla
    data object Escaner : Pantalla
    class Resultado(val recortes: List<Recorte>) : Pantalla
}

@Composable
fun App(servidorPorDefecto: String) {
    var ajustes by remember { mutableStateOf(Ajustes(servidorPorDefecto)) }
    var pantalla by remember { mutableStateOf<Pantalla>(Pantalla.Inicio) }

    TemaOjo {
        when (val p = pantalla) {
            Pantalla.Inicio -> PantallaInicio(
                ajustes = ajustes,
                alCambiar = { ajustes = it },
                alEscanear = { pantalla = Pantalla.Escaner },
            )
            Pantalla.Escaner -> PantallaEscaner(
                ajustes = ajustes,
                alVolver = { pantalla = Pantalla.Inicio },
                alTerminar = { pantalla = Pantalla.Resultado(it) },
            )
            is Pantalla.Resultado -> PantallaResultado(
                ajustes = ajustes,
                recortes = p.recortes,
                alRepetir = { pantalla = Pantalla.Escaner },
                alInicio = { pantalla = Pantalla.Inicio },
            )
        }
    }
}
