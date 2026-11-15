# Anemia por conjuntiva palpebral · reentreno con hemoglobina real (2026-11-15)

Script: `python -m src.training.train_anemia_hb {extraer|evaluar|extraer_app_crudo|evaluar2|guardar}`
Resultados: `reports/anemia_hb/resultados.json`, `resultados2.json`. Modelo: `models/anemia/` (anterior en `models/anemia_respaldo_v2/`).

## Datos
- **Eyes-Defy-Anemia original** (Dimauro et al. 2022; copia de Kaggle `harshwardhanfartale/eyes-defy-anemia`):
  217 personas (India 95, 68 con anemia; Italia 122, 23 con anemia), 1 foto por persona, recorte a mano de la
  conjuntiva palpebral, Hb de laboratorio, sexo y edad. Anemia = OMS (Hb < 12 mujer / < 13 hombre).
  Las fotos se tomaron con un aditamento macro que estandariza la luz.
- Externo: Bangladesh (Kaggle `nahiyan1402/anemiadataset`, 104 personas, solo % R/G/B + Hb).

## Validación
5 particiones × 10–20 repeticiones, **una persona = una foto** (sin fuga). Como India tiene más anemia y piel más
oscura, la AUC global premia adivinar el país: se reporta la **AUC dentro de cada país** (promedio ponderado).

| Rasgos | India | Italia | Dentro del país |
|---|---|---|---|
| Solo sexo + edad | 0.65 | 0.79 | 0.73 |
| App actual (U-Net + balance de blancos) | 0.57 | 0.88 | 0.74 |
| Recorte a mano, sin balance | 0.69 | 0.90 | 0.81 |
| CNN EfficientNetV2-B0 (embedding) | 0.57 | 0.91 | 0.76 |
| **App U-Net, sin balance (nuevo, logística)** | **0.68** | **0.93** | **0.82** |

- Hemoglobina (g/dL): MAE 1.23 con color contra 1.31 solo con país+sexo → el color casi no estima la Hb exacta.
- Externo Bangladesh (otra cámara, sin aditamento): AUC 0.51 → **no transfiere**.

## Validación externa con Ghana (fotos de celular, niños)
Mendeley `nt7r8hv2pz` (CC BY 4.0; la versión con Hb, CP-AnemiC, fue retirada por los autores): 517 niños de 6–59 meses
de 10 hospitales, conjuntiva ya recortada, anémico/no según Hb de laboratorio. Se usa 1 foto original por niño
(las copias aumentadas tienen el mismo color y, si se mezclan, inflan la AUC: por eso los artículos reportan > 90 %).

| Entrenamiento → prueba | AUC por niño |
|---|---|
| Eyes-Defy → Ghana (externo) | 0.62 |
| Ghana → Ghana (5 particiones por niño) | 0.63 |
| Ghana + Eyes-Defy → Ghana | 0.64 |
| CNN EfficientNetV2-B0 + color → Ghana | 0.67 |

En piel oscura y con cámara de celular sin aditamento el color de la conjuntiva da ≈ 0.62–0.67: sirve como apoyo,
no como diagnóstico.

## Modelo en producción (v3)
Regresión logística sobre las 12 medidas de color de la U-Net **sin** balance de blancos (`balance_blancos: false`
en el paquete; `inference_anemia.AnalizadorAnemia` lo respeta). Zona gris 0.35–0.65 (cobertura 81 %):
sensibilidad 0.88, especificidad 0.63 (ponderadas 50/50 por país).

## Limitaciones
- Funciona bien en piel clara (Italia) y poco en piel oscura (India, 0.68).
- Validado solo con la luz estandarizada del aditamento; con luz ambiente del celular **no está validado**
  (Bangladesh lo sugiere). Por eso en la app sigue en **MODO PRUEBAS**.
- El dataset más grande disponible (IEEE DataPort 10.21227/n7km-nr64, India, 1 485 personas con Hb, CC BY 4.0)
  requiere cuenta; es el siguiente paso para validar con celular.
