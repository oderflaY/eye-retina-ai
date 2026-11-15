# Pulso con la cámara (PPG: dedo + flash)

`src/training/train_ppg.py` (venv `.venv-rag`: wfdb, scipy, scikit-learn) → coeficientes copiados en
`mobile/shared/.../astro/Ppg.kt` (`ModeloPpg`). La app re-muestrea a 30 Hz los últimos 10 s del rojo, analiza
1 vez por segundo y solo actualiza el pulso de la misión cuando la calidad pasa el umbral.

## Datos

| Dataset | Licencia | Uso |
|---|---|---|
| [BUT PPG 2.0.0](https://physionet.org/content/butppg/2.0.0/) (Brno, PhysioNet) — 3 888 × 10 s, 50 personas, Xiaomi Mi9 / Huawei P20 Pro, pulso de referencia por ECG | CC BY 4.0 | Entrenamiento y validación por persona |
| [MTHS](https://github.com/MahdiFarvardin/MEDVSE) (Farvardin et al. 2022) — 62 personas, iPhone 5s, pulso y SpO₂ de oxímetro | CC BY-NC-ND 4.0 | **Solo evaluación** (no se empaqueta nada entrenado con él) |

## Método

- Pasa-banda Butterworth 0.7–3.5 Hz (ida y vuelta) → guía espectral (FFT con Hann) → picos con prominencia ≥ 0.3·σ
  separados ≥ 0.6 periodos → pulso = 60 000 / mediana de intervalos.
- Calidad: regresión logística con 13 rasgos (concentración espectral, autocorrelación, acuerdo FFT/picos, CV de intervalos,
  perfusión, asimetría, curtosis, deriva, nivel de rojo, picos esperados, parecido entre latidos, entropía espectral,
  energía fuera de banda). Objetivo: pulso a ±5 lpm del ECG. Umbral = 30 % mejor de las ventanas del dedo.

## Resultados (reports/ppg/resultados.json)

| | App anterior | Nuevo |
|---|---|---|
| BUT PPG, grabaciones buenas del dedo: error medio | 6.3 lpm (sin lectura en 30 %) | **4.5 lpm** (siempre da lectura) |
| … dentro de ±10 lpm | 83 % | **90 %** |
| BUT PPG, dedo, ventanas aceptadas por calidad | — | 6.4 lpm (rechazadas: 18 lpm) |
| MTHS externo (iPhone), aceptadas por calidad (70 %) | 7.8 lpm (85 % con lectura) | **5.0 lpm, 77 % a ±5, 90 % a ±10** |

- Calidad: AUC 0.70 para «pulso correcto» y 0.82 frente a la etiqueta humana de BUT PPG (validación agrupada por persona).
- **SpO₂ con la cámara: NO funciona** con estos datos. Ratio de ratios rojo/azul en MTHS: error 1.8 puntos frente a 1.5
  prediciendo la media; correlación por persona −0.14 (solo 5 personas con SpO₂ < 94 %). No se incluye en la app.
- `PpgModeloTest` comprueba que Kotlin da lo mismo que Python en 8 señales reales de BUT PPG
  (`androidHostTest/resources/ppg_vectores_butppg.json`, CC BY 4.0, Nemcova et al., PhysioNet).
