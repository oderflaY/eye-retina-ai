# Perfil médico y encía — OjoScan

Actualizado: 2026-11-14.

## El chequeo completo ahora

| Paso | Qué se mide | Modelo | Fiabilidad |
|---|---|---|---|
| 1. Tus datos | Edad, sexo, peso, estatura, cuello, cintura, cadera, presión, dolor de pecho, laboratorio opcional | Ver abajo | — |
| 2. Ojos de cerca | Ojo rojo, pterigión, catarata | EfficientNetV2 | Exactitud balanceada 0.965 |
| 3. Párpado | Anemia por conjuntiva | Color + clasificador | Detección 86 %, aciertos en sanos 79 % |
| 4. Rostro | Parálisis facial, ictericia, ojos | MediaPipe + clasificadores | AUC 0.95 / 0.96 |
| 5. Encía | Anemia por palidez de la encía | **Experimental** | Sin validar con hemoglobina |

Al final la app arma el **Perfil médico**: los datos de la persona, cada indicador con semáforo y
un pronóstico que cruza todas las mediciones. La palma sigue disponible como medición suelta.

## Perfil (datos escritos) — de `~/Deep_Learning`

* **Grasa corporal** (`bodyfat.csv`, 252 hombres): regresión con edad, peso, estatura, cuello,
  cintura y cadera. Error medio con validación cruzada repetida: **±3.65 puntos**. Sin modelo el
  error es ±6.74 y la fórmula de la Marina da ±4.06. En mujeres se usa la fórmula de la Marina,
  porque el dataset solo tiene hombres.
* **Riesgo coronario** (UCI Cleveland, el de `analysis.ipynb`): regresión logística.
    * Solo cuestionario y presión: **AUC 0.852** (rango del 90 % de las particiones: 0.78–0.92),
      detección 73 %, aciertos en sanos 80 %.
    * Con colesterol, glucosa y pulso máximo: AUC 0.861.
    * El modelo de `analysis.ipynb`, con estudios de hospital, da 0.909 (solo como referencia).
    * **Solo se aplica si hay dolor de pecho.** En Cleveland los pacientes "sin dolor" fueron enviados
      a cardiología por otra causa y la mayoría estaban enfermos; aplicarlo a una persona sana daría un
      riesgo falso.
* Reglas publicadas, no modelos: IMC (OMS), cintura/estatura > 0.5 (Ashwell), presión (AHA 2017),
  glucosa (ADA) y colesterol (NCEP).
* Cruces en el pronóstico: presión alta + posible dolor cardiaco → alta; cintura alta + presión o
  glucosa alta → patrón de síndrome metabólico.
* Archivos: `src/training/train_perfil.py`, `src/deployment/inference_perfil.py`,
  `POST /predict/perfil`. Los datos viajan como archivo `datos.json`, sin el nombre.

## Encía

1. MediaPipe encuentra la abertura de la boca (contorno interno de los labios, `servicio_rostro`).
2. Una U-Net encuentra los dientes. Se entrenó con Dentalai: **Dice 0.92** en prueba (`models/dientes`).
3. **Encía = la franja de la boca justo encima de los dientes de arriba**, de color rosado, sin
   sombra ni brillo. La primera versión tomaba también la lengua; se corrigió midiendo solo encima
   de los dientes.
4. Los dientes sirven de blanco para corregir la luz.
5. El enrojecimiento (eritema) se compara con **261 franjas de encía sana** (grado 0) del dataset
   de gingivitis:
    * más pálida que el 90 % de las sanas → palidez;
    * entre el 10 % y el 25 % → en el límite;
    * más roja que el 95 % de las sanas → posible inflamación. La gingivitis enrojece y tapa la
      palidez, así que no cuenta para la anemia.

**Prueba del medidor:** el eritema sube con el grado de gingivitis (0.355 → 0.376 → 0.393 → 0.416 →
0.430) y separa encía sana de inflamada con AUC 0.73. Esto prueba que el medidor **ve el color**,
pero **no prueba que detecte anemia**: no existe un dataset público de encía con hemoglobina.

Por eso, en el pronóstico **la encía solo apoya**. Si el párpado y la palma dicen algo, la encía lo
confirma o lo matiza, pero nunca sube el nivel. Si es la única medición de anemia, como mucho da
"vigilar".

**Limitaciones:**

* Las fotos de referencia son fotos dentales con flash y separadores; las de la app son selfies.
* Falta probarlo con fotos reales de la app sonriendo y enseñando la encía.
* Para validarlo de verdad hacen falta fotos de encía de personas con biometría hemática. Una opción
  es pedir los datos del estudio de encía (Cureus 2024).

Archivos: `src/training/train_dientes.py`, `src/data/referencia_encias.py`,
`src/deployment/inference_encia.py`, `POST /predict/encia`, `models/encia/referencia.json`.

## Pulsera (Xiaomi Smart Band y similares)

**En vivo, por Bluetooth:** la app lee el servicio estándar de frecuencia cardíaca, sin la nube de Xiaomi.
Para eso hay que activar en la pulsera "Transmitir frecuencia cardíaca". Mide durante todo el chequeo:

* **Pulso en reposo:** la mediana de los 30 s más tranquilos.
* **Mínimo y máximo.**
* **Ritmo:** se calcula con los intervalos entre latidos (VFC y % de latidos irregulares).

**Historial de 30 días, por Health Connect:** se activa en Mi Fitness → Health Connect. El teléfono
lo resume y solo manda promedios al servidor:

| Indicador | Regla | Fuente |
|---|---|---|
| Pulso en reposo | < 50 bajo · 50–90 normal · 91–100 límite · > 100 taquicardia | AHA |
| Tendencia del pulso | Última semana ≥ 7 lpm más alto que las anteriores → vigilar (infección, poco sueño, anemia) | — |
| Oxígeno (SpO₂) | < 92 % alta · < 95 % o > 5 % de lecturas < 90 % → vigilar | — |
| Sueño | < 6 h o < 7 h → vigilar · > 10 h → vigilar | AASM |
| Actividad | < 5,000 pasos/día = sedentario | Paluch 2022 |
| Posible apnea | Bajadas de oxígeno + obesidad, presión ≥ 130 o poco sueño | STOP-BANG |

**Cruces en el pronóstico:**

* Pulso alto o en subida, o mucho sueño, + palidez (párpado o encía) → apoya la anemia.
* Posible apnea + presión alta → alta.
* Cintura alta + pocos pasos → patrón metabólico.

Las pulseras miden peor el oxígeno y el ritmo que un oxímetro o un electrocardiograma. Los resultados
piden siempre confirmarse.
