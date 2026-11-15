package com.temrarios.ojoscan

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.togetherWith
import androidx.compose.runtime.Composable
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.collectAsState
import com.temrarios.ojoscan.platform.PermisosAlIniciar
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import com.temrarios.ojoscan.api.DatosPerfil
import com.temrarios.ojoscan.api.aJson
import com.temrarios.ojoscan.api.perfilDeJson
import com.temrarios.ojoscan.pulso.rememberMonitorPulso
import com.temrarios.ojoscan.pulso.resumir
import com.temrarios.ojoscan.historial.EstadoHistorial
import com.temrarios.ojoscan.historial.paraServidor
import com.temrarios.ojoscan.historial.rememberFuenteHistorial
import com.temrarios.ojoscan.api.HistorialPerfil
import com.temrarios.ojoscan.guia.Modo
import com.temrarios.ojoscan.guia.Ojo
import com.temrarios.ojoscan.platform.Recorte
import com.temrarios.ojoscan.platform.rememberAlmacen
import androidx.lifecycle.viewmodel.compose.viewModel
import com.temrarios.ojoscan.astro.AstroPulseEngine
import com.temrarios.ojoscan.ui.astro.AstroPulseMainScreen
import com.temrarios.ojoscan.ui.PantallaCorazon
import com.temrarios.ojoscan.ui.PantallaEscaner
import com.temrarios.ojoscan.ui.PantallaInicio
import com.temrarios.ojoscan.ui.PantallaPaso
import com.temrarios.ojoscan.ui.PantallaPerfil
import com.temrarios.ojoscan.ui.PantallaResultado
import com.temrarios.ojoscan.ui.TemaOjo

/** Opciones que la persona elige en el inicio. */
data class Ajustes(
    val servidor: String,
    /** Chequeo completo: todas las mediciones seguidas y un pronóstico que las combina. */
    val completo: Boolean = true,
    val modo: Modo = Modo.OJO,
    val ojo: Ojo = Ojo.IZQUIERDO,
    val trasera: Boolean = false,
    val voz: Boolean = true,
    /** Guardar las fotos en el servidor para entrenar el modelo (requiere consentimiento). */
    val guardar: Boolean = false,
)

/** Una medición del protocolo: qué se revisa y qué ojo. */
data class Paso(val modo: Modo, val ojo: Ojo)

/** Lo que se capturó en un paso. */
class Medicion(val modo: Modo, val recortes: List<Recorte>)

/** Orden del chequeo completo: lo que pide acercarse primero, la cara y la mano al final. */
val PASOS_COMPLETOS = listOf(
    Paso(Modo.OJO, Ojo.AMBOS),
    Paso(Modo.ANEMIA, Ojo.IZQUIERDO),
    Paso(Modo.ROSTRO, Ojo.AMBOS),
    Paso(Modo.ENCIA, Ojo.AMBOS),
    Paso(Modo.PALMA, Ojo.AMBOS),
)

fun Ajustes.pasos(): List<Paso> = if (completo) PASOS_COMPLETOS else listOf(Paso(modo, ojo))

private sealed interface Pantalla {
    /** Tablero AstroPulse (inicio): telemetría, módulos, farmacia, SOS y asistente. */
    data object Astro : Pantalla
    /** Principal: telemetría de la pulsera Xiaomi + IA cardiaca. */
    data object Corazon : Pantalla
    /** Escaneos con la cámara (ojo, anemia, rostro, palma, encía). */
    data object Inicio : Pantalla
    /** "Tus datos" (perfil médico escrito): primer paso del chequeo completo, o solo guardar desde Corazón. */
    class Datos(val soloGuardar: Boolean = false) : Pantalla
    /** Instrucciones antes de un paso del chequeo completo. */
    class Preparar(val indice: Int, val hechas: List<Medicion>) : Pantalla
    class Escaner(val indice: Int, val hechas: List<Medicion>) : Pantalla
    class Resultado(val mediciones: List<Medicion>, val perfil: DatosPerfil?) : Pantalla
}

@Composable
fun App(servidorPorDefecto: String) {
    var ajustes by remember { mutableStateOf(Ajustes(servidorPorDefecto)) }
    var pantalla by remember { mutableStateOf<Pantalla>(Pantalla.Astro) }
    val almacen = rememberAlmacen()
    var perfil by remember { mutableStateOf(perfilDeJson(almacen.leer("perfil")) ?: DatosPerfil()) }
    /** El perfil se usa en este chequeo (false si la persona lo saltó). */
    var usarPerfil by remember { mutableStateOf(false) }
    /** Pulsera por Bluetooth: mide el pulso en vivo durante todo el chequeo. */
    val pulso = rememberMonitorPulso()
    /** Historial de 30 días de Mi Fitness (Health Connect). */
    val historial = rememberFuenteHistorial()
    /** Motor de AstroPulse: vive mientras viva la actividad. */
    val motor = viewModel { AstroPulseEngine(almacen) }
    // Al abrir se piden todos los permisos de una vez y luego se lee el historial: así entra a cualquier diagnóstico.
    PermisosAlIniciar { historial.leerSiHayPermiso(); pulso.iniciar() }
    val estadoHistorial by historial.estado.collectAsState()
    /** Aviso único por apertura cuando Health Connect está vacío (Mi Fitness no comparte). */
    var avisoHistorialVisto by remember { mutableStateOf(false) }

    /** Perfil + resumen del pulso de esta sesión (si la pulsera midió). */
    fun perfilFinal(): DatosPerfil? {
        val r = resumir(pulso.muestras.value)
        val h = (historial.estado.value as? EstadoHistorial.Listo)?.resumen?.paraServidor()?.let {
            HistorialPerfil(it.dias, it.pulsoReposo, it.pulsoReposo7d, it.pulsoReposoBase, it.pulsoMax, it.spo2Media, it.spo2Min,
                it.spo2Bajas, it.spo2N, it.suenoHoras, it.suenoNoches, it.suenoCortas, it.profundo, it.pasos, it.pasosDias, it.vfc)
        }
        val base = (perfil.takeIf { usarPerfil && it.util } ?: if (r != null || h != null) DatosPerfil() else return null).copy(historial = h)
        if (r == null) return base
        return base.copy(pulsoReposo = r.reposo, pulsoMin = r.minimo, pulsoMax = r.maximo, pulsoSegundos = r.segundos,
            pulsoRmssd = r.rmssd, pulsoIrregular = r.irregular)
    }

    fun empezar() {
        usarPerfil = false
        pulso.reiniciarSesion()
        historial.leerSiHayPermiso()
        pantalla = if (ajustes.completo) Pantalla.Datos() else Pantalla.Escaner(0, emptyList())
    }

    /** Pasa al siguiente paso, o al resultado si ya no quedan. */
    fun avanzar(indice: Int, hechas: List<Medicion>) {
        val pasos = ajustes.pasos()
        pantalla = when {
            indice + 1 >= pasos.size -> Pantalla.Resultado(hechas, perfilFinal())
            ajustes.completo -> Pantalla.Preparar(indice + 1, hechas)
            else -> Pantalla.Escaner(indice + 1, hechas)
        }
    }

    TemaOjo {
        if (!avisoHistorialVisto && estadoHistorial == EstadoHistorial.SinDatos) AlertDialog(
            onDismissRequest = { avisoHistorialVisto = true },
            title = { Text("Falta activar Mi Fitness") },
            text = {
                Text("OjoScan ya puede leer Health Connect, pero está vacío: Mi Fitness todavía no le comparte tus datos.\n\n" +
                    "En Health Connect entra a Permisos de apps → Mi Fitness → Permitir todo. Luego abre Mi Fitness para que sincronice.")
            },
            confirmButton = { TextButton({ avisoHistorialVisto = true; historial.abrirAjustes() }) { Text("Abrir Health Connect") } },
            dismissButton = { TextButton({ avisoHistorialVisto = true }) { Text("Después") } },
        )
        AnimatedContent(
            pantalla,
            transitionSpec = {
                (fadeIn(tween(450)) + scaleIn(tween(450), initialScale = 0.94f)) togetherWith
                    (fadeOut(tween(250)) + scaleOut(tween(250), targetScale = 1.04f))
            },
        ) { p ->
        when (p) {
            Pantalla.Astro -> AstroPulseMainScreen(
                motor = motor,
                pulso = pulso,
                historial = historial,
                alIaCardiaca = { pantalla = Pantalla.Corazon },
                alEscaneos = { pantalla = Pantalla.Inicio },
                alAnemia = { m ->
                    ajustes = ajustes.copy(completo = false, modo = m)
                    pulso.reiniciarSesion()
                    pantalla = Pantalla.Escaner(0, emptyList())
                },
                alChequeoCompleto = { ajustes = ajustes.copy(completo = true); empezar() },
            )
            Pantalla.Corazon -> PantallaCorazon(
                alVolver = { pantalla = Pantalla.Astro },
                servidor = ajustes.servidor,
                pulso = pulso,
                historial = historial,
                perfil = perfil,
                alEscaneos = { pantalla = Pantalla.Inicio },
                alTusDatos = { pantalla = Pantalla.Datos(soloGuardar = true) },
            )
            Pantalla.Inicio -> PantallaInicio(
                alVolver = { pantalla = Pantalla.Astro },
                ajustes = ajustes,
                alCambiar = { ajustes = it },
                alEscanear = ::empezar,
            )
            is Pantalla.Datos -> PantallaPerfil(
                inicial = perfil,
                pulso = pulso,
                historial = historial,
                numeroPaso = if (p.soloGuardar) null else "PASO 1 DE ${ajustes.pasos().size + 1}",
                alGuardar = {
                    perfil = it
                    almacen.guardar("perfil", it.aJson())
                    usarPerfil = true
                    pantalla = if (p.soloGuardar) Pantalla.Corazon else Pantalla.Preparar(0, emptyList())
                },
                alSaltar = { usarPerfil = false; pantalla = if (p.soloGuardar) Pantalla.Corazon else Pantalla.Preparar(0, emptyList()) },
                alCancelar = { pantalla = if (p.soloGuardar) Pantalla.Corazon else Pantalla.Inicio },
            )
            is Pantalla.Preparar -> PantallaPaso(
                pasos = ajustes.pasos(),
                indice = p.indice,
                alEmpezar = { pantalla = Pantalla.Escaner(p.indice, p.hechas) },
                alSaltar = { avanzar(p.indice, p.hechas) },
                alCancelar = { pantalla = Pantalla.Inicio },
            )
            is Pantalla.Escaner -> {
                val paso = ajustes.pasos()[p.indice]
                PantallaEscaner(
                    ajustes = ajustes,
                    paso = paso,
                    pulso = pulso,
                    numero = if (ajustes.completo) "PASO ${p.indice + 2}/${ajustes.pasos().size + 1}" else null,
                    alVolver = { pantalla = if (ajustes.completo) Pantalla.Preparar(p.indice, p.hechas) else Pantalla.Inicio },
                    alTerminar = { avanzar(p.indice, p.hechas + Medicion(paso.modo, it)) },
                )
            }
            is Pantalla.Resultado -> PantallaResultado(
                ajustes = ajustes,
                mediciones = p.mediciones,
                perfil = p.perfil,
                alRepetir = ::empezar,
                alInicio = { pantalla = Pantalla.Astro },
                alAnemia = { m, a ->
                    val metodo = when (m) { Modo.PALMA -> "palma"; Modo.ENCIA -> "encía"; else -> "párpado" }
                    motor.anemia(metodo, a.resultado, a.titulo, a.probabilidad, a.validacion?.auc)
                },
                alEstudio = { tipo, nombre, texto, prob, triage -> motor.registrarEstudio(tipo, nombre, texto, prob, triage) },
            )
        }
        }
    }
}
