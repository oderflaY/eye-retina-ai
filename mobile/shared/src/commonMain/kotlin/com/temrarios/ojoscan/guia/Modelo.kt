package com.temrarios.ojoscan.guia

/** Ojo de la persona (su izquierdo / su derecho, no el lado de la pantalla). */
enum class Ojo(val nombre: String) {
    IZQUIERDO("izquierdo"),
    DERECHO("derecho"),
    AMBOS("ambos"),
}

/** Qué se va a revisar. Cambia la foto que se pide y el análisis que se hace. */
enum class Modo(val titulo: String, val detalle: String) {
    OJO("Ojo", "Enrojecimiento, pterigión, catarata"),
    ANEMIA("Anemia", "Palidez del párpado por dentro"),
}

/**
 * Lo que la cámara midió en un cuadro. Lo llena cada plataforma (Android: detector de ojo
 * TFLite + CameraX). Coordenadas normalizadas 0..1 respecto al cuadro derecho (vertical).
 *
 * Todo gira alrededor del OJO (como un detector de caras, pero del ojo): la app no necesita ver
 * la cara completa, así que se puede acercar el teléfono hasta que el ojo llene la pantalla.
 */
data class Lectura(
    val tiempoMs: Long,
    /** El detector encontró un ojo (confianza ≥ su umbral). */
    val ojoVisible: Boolean = false,
    /** Confianza del detector (0..1). */
    val confianza: Float = 0f,
    /** Ancho del ojo (de esquina a esquina) / ancho del cuadro: qué tan cerca está. */
    val anchoOjo: Float = 0f,
    /** Centro del ojo en el cuadro. */
    val ojoX: Float = 0.5f,
    val ojoY: Float = 0.5f,
    /** Distancia entre párpados / ancho del ojo (≈0.3 abierto, < 0.15 cerrado); null si no se midió. */
    val apertura: Float? = null,
    /** No hay ojo cerca pero sí una cara completa a lo lejos: falta acercarse. */
    val caraLejos: Boolean = false,
    /** Brillo medio de la zona del ojo (0..255). */
    val brillo: Float = 128f,
    /** Varianza del laplaciano en el ojo (más alto = más nítido); null si no se midió. */
    val nitidez: Float? = null,
    /** Velocidad de giro del teléfono (rad/s, giroscopio); 0 si no hay giroscopio. */
    val giro: Float = 0f,
)

/** Límites de la guía, ajustados para un ojo de cerca. */
data class Umbrales(
    /**
     * El ojo debe ocupar entre estas fracciones del ancho del CUADRO de la cámara (3:4). La vista
     * previa recorta los costados en pantallas alargadas (en un 20:9 se ve ≈60 % del ancho), así
     * que 0.28–0.50 del cuadro ≈ 50–85 % del ancho visible de la pantalla.
     */
    val anchoMin: Float = 0.28f,
    val anchoMax: Float = 0.50f,
    val desvioCentro: Float = 0.08f,
    val aperturaMin: Float = 0.18f,       // párpado a párpado / esquina a esquina: abierto ≈0.29, cerrado ≈0.10
    val brilloMin: Float = 60f,
    val brilloMax: Float = 220f,
    val nitidezMin: Float = 12f,
    /** rad/s del giroscopio a partir de los cuales el teléfono "se está moviendo". */
    val giroMax: Float = 0.30f,
    /** Velocidad máxima del ojo en el cuadro (fracción del cuadro por segundo). */
    val velocidadMax: Float = 0.35f,
    /** Tiempo que todo debe estar bien seguido antes de tomar la foto. */
    val sostenerMs: Long = 900,
    /** Un aviso nuevo debe repetirse este tiempo antes de mostrarse (evita parpadeo de mensajes). */
    val cambioMs: Long = 250,
    /** Pausa mínima entre dos frases habladas. */
    val pausaVozMs: Long = 2500,
)

/** Avisos en orden de prioridad: se muestra el primero que falle. */
enum class Aviso(val texto: String, val bueno: Boolean = false) {
    SIN_OJO("Acerca la cámara a tu ojo"),
    CARA_LEJOS("Acércate más, solo a tu ojo"),
    POCA_LUZ("Busca más luz"),
    MUCHA_LUZ("Hay demasiada luz, evita el reflejo"),
    NO_MUEVAS("No muevas la cámara"),
    ACERCATE("Acércate un poco"),
    ALEJATE("Aléjate un poco"),
    CENTRA("Pon el ojo dentro del óvalo"),
    ABRE_OJO("Abre más el ojo"),
    ENFOCANDO("Quieto… enfocando"),
    LISTO("¡Perfecto! No te muevas", bueno = true),
    CAPTURADO("Foto tomada", bueno = true),
}

/** Lo que la pantalla debe mostrar tras evaluar un cuadro. */
data class Guia(
    val aviso: Aviso,
    /** 0..1: cuánto falta para la foto automática (solo avanza en LISTO). */
    val progreso: Float,
    /** true exactamente una vez: la plataforma debe tomar la foto en alta resolución. */
    val capturar: Boolean = false,
    /** Frase para decir en voz alta (null = callar). */
    val hablar: String? = null,
)
