# Modo Rostro: enfermedades visibles en la cara que complementan al ojo

Tamizaje orientativo, **no diagnóstico**. Una foto de frente de la cara da tres lecturas que se
cruzan entre sí:

| Lectura | Qué mide | Cómo se cruza con el ojo |
|---|---|---|
| **Parálisis / asimetría facial** (Bell, secuela de infarto cerebral) | Diferencia izquierda-derecha de 14 movimientos de MediaPipe (parpadeo, sonrisa, ceja…) + geometría (altura de comisuras, cejas, apertura de párpados) | Si un párpado cierra distinto se avisa con prioridad alta: ojo que no cierra = se reseca, y asimetría súbita = urgencias |
| **Ictericia** (ojos amarillos) | Color CIELAB del blanco del ojo (sin iris ni pestañas), corregido por la luz y comparado con la piel de las mejillas | Es el mismo blanco del ojo que analiza el modo Ojo |
| **Los dos ojos** | Se recortan de la misma foto y pasan por el analizador de ojo externo | Ojo rojo + asimetría → revisión con oftalmología |

## Por qué medidas y no una red sobre la foto

En los datasets públicos las fotos con enfermedad y las sanas vienen de fuentes distintas
(videos caseros vs. youtubers; bancos clínicos con marca de agua vs. stock). Una CNN aprendería
la fuente, no la enfermedad. Las medidas (asimetría, color de la esclerótica) son lo que mira el
médico y no dependen de la cámara.

## Resultados (validación honesta)

### Parálisis facial: `models/rostro/paralisis.joblib`

* Datos: PalsyNet (CC BY 4.0), 49 personas (27 con parálisis), 2,940 cuadros; 907 caras sanas de FFHQ.
* Validación cruzada **por persona** (StratifiedGroupKFold 7 × 5): nadie aparece en entrenamiento y prueba a la vez.
* Se compararon 16 variantes (logística, bosque, extra-trees, gradient boosting; con y sin
  magnitud de movimiento; con y sin FFHQ como negativos). Ganó gradient boosting + movimiento + FFHQ.

| Métrica | Valor |
|---|---|
| AUC por persona | 0.953 |
| Sensibilidad por persona | 78 % (21 de 27) |
| Especificidad por persona | 100 % (22 de 22) |
| Falsas alarmas en caras sanas FFHQ | 6.8 % |
| AUC por foto suelta | 0.857 |

Rasgos más importantes: inclinación de la línea de los labios, altura de las comisuras, estiramiento de la boca, parpadeo desigual.

### Ictericia: `models/rostro/ictericia.joblib`

* Datos: Jaundice_Dataset (CC BY-NC); solo 40 fotos con ictericia tienen la cara completa; 1,498 caras sanas de FFHQ + 48 sanas del dataset.
* Regresión logística sobre 7 medidas de color; validación cruzada estratificada 5 × 10.

| Métrica | Valor |
|---|---|
| AUC | 0.958 ± 0.049 |
| Sensibilidad | 82.5 % |
| Especificidad | 96.2 % |
| Falsas alarmas en caras sanas FFHQ | 3.8 % |

### Lo que se probó y se descartó

* **Rosácea / enrojecimiento facial**: el enrojecimiento de mejillas no separa rosácea de caras sanas
  (AUC 0.52 contra FFHQ). Con fotos de dermatología de cerca y sin cara completa no es medible en
  la app. No se incluye.

## Zonas grises

Ambos modelos dan veredicto solo fuera de [0.35, 0.65]; en medio se reporta «no concluyente».
Si la cara está girada más de 25°, la asimetría no se evalúa.

## Límites

* Pocos datos: 49 personas para parálisis, 40 casos para ictericia. Los números tienen márgenes amplios.
* PalsyNet tiene parálisis marcada; una parálisis leve probablemente no se detecta.
* La ictericia depende de la luz; con luz cálida o amarilla aumenta el riesgo de falsa alarma.
* Las fotos de cara nunca se donan al entrenamiento (son biométricas), aunque esté activado «guardar».

## Arquitectura

```
teléfono → relay VPS (:13443) → worker de la PC → API (:8000, .venv, TensorFlow)
                                                    └→ servicio_rostro (:8001, .venv-etiquetas, MediaPipe)
```

MediaPipe choca con las versiones de TensorFlow de la API, así que corre aparte como servicio
de usuario `ojoscan-rostro.service`:

```
ExecStart=.venv-etiquetas/bin/python -m uvicorn src.deployment.servicio_rostro:app --host 127.0.0.1 --port 8001
```

## Reentrenar

```bash
.venv-etiquetas/bin/python src/data/rasgos_rostro.py --palsy ~/TemrariosSpace/datasets/Rostro/palsy/frames \
    --ffhq data/raw/ffhq_listos --max_ffhq 1000
.venv-etiquetas/bin/python src/data/rasgos_color_rostro.py --ictericia ~/TemrariosSpace/datasets/Rostro/ictericia \
    --ffhq data/raw/ffhq_listos --max_ffhq 1500
python -m src.training.train_rostro_paralisis
python -m src.training.train_rostro_ictericia
```

## ¿Anemia por la cara? Probado: NO se puede (2026-11-14)

No hay datos públicos de caras con hemoglobina, así que se usó la piel que sale alrededor del ojo
en Eyes-defy-anemia (218 personas con anemia confirmada por sangre). Se quitó la conjuntiva, el
blanco del ojo y el iris y se midió solo la piel (`experimento_anemia_piel.py`, `control_anemia_piel.py`).

| Prueba | AUC |
|---|---|
| Piel de la cara (sin controlar nada) | 0.835, parecía muy buena |
| Aro de plástico del aparato (no es la persona) | 0.69 |
| Solo condiciones de la foto (aro + blanco del ojo + brillo) | 0.79 |
| **Piel quitando el efecto de las condiciones de la foto** | **0.54 = azar** |
| Conjuntiva quitando el efecto de las condiciones de la foto | 0.74 |
| Conjuntiva + piel vs. solo conjuntiva (con condiciones) | 0.866 vs. 0.882: la piel no suma |

Conclusión: la piel de la cara **no detecta anemia**. Su aparente acierto venía de que las fotos
de personas con y sin anemia se tomaron con distinta luz/sesión. La palidez de la cara solo es
visible en anemias graves y se confunde con el tono de piel y la luz. No se integra al modo Rostro.

Esto también afecta al modelo del ojo: parte de su AUC 0.83 viene de las condiciones de la foto.
La señal real de la conjuntiva está entre 0.74 (quitando todo el efecto de la foto, que también
quita algo real) y 0.83.

Siguiente paso con datos reales: datasets de Ghana (CC BY 4.0, 710 niños con hemoglobina) de
**conjuntiva, palma y uñas**, descargados en `~/TemrariosSpace/datasets/Anemia/ghana`. La palidez de
la palma es un signo que usa la OMS.

### Segunda ronda: quitar las confusiones (2026-11-14)

**Ojo (conjuntiva).** Eyes-defy-anemia mezcla India (Chennai) e Italia (Bari) y la copia que usamos
no dice el país. La piel más oscura tiene 65 % de anemia y la más clara 22 %: el tono de piel solo ya
da AUC 0.68. Comparando **solo personas del mismo tono** (cuartiles), la conjuntiva da **AUC 0.777**
(antes se reportaba 0.83). Reentrenar con pesos por tono no mejora (0.773). La app ahora muestra los
números honestos (`validar_anemia_honesto.py`): detección 86 %, acierto en sanos 79 %, veredicto en
63 % de las fotos. Para eliminar la confusión del todo hace falta la versión original de IEEE
DataPort (trae país y hemoglobina por foto).

**Palma y uñas (Ghana, un solo sitio, una imagen por niño; las 12 copias aumentadas son idénticas
en color)** — `experimento_anemia_ghana.py`:

| Prueba | Palma (475 niños) | Uñas (476) |
|---|---|---|
| Solo tamaño/forma/nitidez del recorte (no es la persona) | 0.64 | 0.55 |
| Color | 0.78 | 0.68 |
| **Color quitando el efecto del recorte** | **0.75** (p < 0.025) | **0.62** |
| Color, solo el lote principal de archivos | 0.74 | 0.62 |
| Palma + uñas del mismo niño | 0.76 | |

La palma tiene señal real (~0.75, igual que la conjuntiva). Las uñas son débiles y no suman.
Límites: son niños de 6–59 meses y los recortes no traen blanco de referencia para corregir la luz.
