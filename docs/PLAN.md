# Plan de trabajo — Eye-Retina AI (8 semanas)

## Resumen ejecutivo

El sistema tiene dos fases. La **Fase 1** localiza ojos en rostros (caja + 6
landmarks) y segmenta vasos en fondos de ojo. La **Fase 2** reutiliza esos
backbones y el pipeline de datos para clasificar enfermedades (retinopatía
diabética, glaucoma, DMAE, catarata) en modo multilabel.

Todo el código vive en `src/` con módulos independientes por etapa (datos →
modelo → entrenamiento → evaluación → despliegue) y se controla con un YAML por
experimento en `configs/`. Así, pasar a la Fase 2 implica agregar una
arquitectura y un YAML, sin reescribir el pipeline.

**Estado (2026-11-13):** la segmentación de vasos (Parte B, semanas 3-4) está
implementada y entrenada en FIVES con validación externa en HRF; `/predict/retina`
ya responde. Métricas en `reports/retina/retina_results.json`.

**Estado (semana 1):** la detección de ojos está implementada de punta a punta
y probada con datos sintéticos. **Falta un dataset real de ojos en
`data/raw/eyes/`**: la carpeta está vacía y no se reporta ninguna métrica real.

## Flujo general

```
                 ┌──────────────────────── data/raw/ ────────────────────────┐
                 │  eyes/  (rostros + cajas/landmarks)   retina/ (fundus+máscaras)│
                 └───────────────┬───────────────────────────────┬────────────┘
                                 │ prepare_eyes.py               │ prepare_retina.py (sem. 3)
                                 ▼                               ▼
               data/processed/eyes/{annotations,splits,stats}.json   data/processed/retina/
                                 │
          ┌──────── tf.data: leer → letterbox → cache → aumentar → codificar → batch → prefetch
          ▼
  ┌───────────────┐   ┌──────────────────┐   ┌──────────────────────────┐
  │ A. Detector   │   │ B. U-Net vasos   │   │ C. Clasificador multilabel│
  │ MobileNetV2 / │   │ ResNet50 / EffB4 │   │ EfficientNetB5 / DenseNet │
  │ EfficientNetB0│   │ Dice + BCE       │   │ BCE/Focal + class_weight  │
  └──────┬────────┘   └────────┬─────────┘   └────────────┬─────────────┘
         └──────────── evaluación + Grad-CAM ─────────────┘
                                 │
                 SavedModel / TFLite  →  FastAPI (/predict/eye|retina|disease)  →  Gradio
```

## Organización sugerida de los datos locales

```
data/raw/
├── eyes/                         # Detección de ojos (cualquiera de estos formatos)
│   ├── images/*.jpg|png
│   └── annotations.csv           # o COCO .json, o labels/*.txt (YOLO), o *.xml (VOC)
└── retina/
    ├── FIVES/      → segmentación de vasos (800 imágenes + máscaras)
    ├── HRF/        → segmentación de vasos (45 imágenes alta resolución)
    └── OCT2017_Kermany/ → clasificación OCT (Fase 2)
```

El formato exacto de las etiquetas de ojos está en
[FORMATO_ETIQUETAS.md](FORMATO_ETIQUETAS.md). Si el detector automático
encuentra más de un formato o clases desconocidas, se detiene y pide
confirmación (`data.label_format`).

## Plan por semana

| Semana | Tareas | Entregables |
| --- | --- | --- |
| 1 | Estructura del repo, entorno (TF + CUDA), cargadores genéricos, detección de formato de etiquetas, pipeline `tf.data`, detector de ojos completo, tests | `src/data`, `src/models/eye_detector.py`, `src/training/train_eye.py`, tests en verde ✅ |
| 2 | Conseguir/curar el dataset real de ojos, EDA, primer entrenamiento (MobileNetV2 224), evaluación en test, ajuste de umbrales | `notebooks/01_eda_ojos.ipynb`, `reports/eye_detection/*` con métricas reales |
| 3 | Comparar backbones (MobileNetV2 vs EfficientNetB0, 128 vs 224), K-Fold si hay < 500 imágenes, análisis de errores; preparar loader de retina (máscaras PNG/GIF/TIF, recorte del disco, CLAHE) | Tabla comparativa de detectores, `prepare_retina.py` |
| 4 | U-Net (ResNet50 / EfficientNetB4) para vasos con Dice + BCE; entrenar en FIVES, validar en HRF/DRIVE | `src/models/unet.py`, métricas Dice, IoU, Se/Sp, HD95 |
| 5 | Despliegue Fase 1: exportar SavedModel + TFLite, API `/predict/eye` y `/predict/retina`, Docker, demo Gradio | Contenedor funcionando, documentación de endpoints |
| 6 | Explicabilidad: Grad-CAM/Grad-CAM++ para ambos modelos, SHAP o LIME en una muestra, informe de errores típicos | `reports/*/explain/`, notebook de interpretabilidad |
| 7 | Preparación Fase 2: clasificador multilabel (EfficientNetB5/DenseNet201) sobre ODIR-5K/RFMiD, focal loss, `class_weight`, reentrenar solo la cabeza | `src/models/classifier.py`, `configs/disease_*.yaml` |
| 8 | MLOps: versionado de modelos (MLflow o W&B Artifacts), cola de *active learning* (baja confianza → revisión), borrador del documento de validación clínica (ISO 13485, IEC 62304, SaMD/COFEPRIS) | `docs/VALIDACION_CLINICA.md`, registro de modelos, reporte final |

## Arquitecturas

| Tarea | Arquitectura | Entrada | Salida | Pérdida | Métrica principal |
| --- | --- | --- | --- | --- | --- |
| A. Detección de ojos | MobileNetV2 o EfficientNetB0 + cabeza YOLO-lite (FPN C4+C5) | 224×224 o 128×128 | rejilla S/16 × 17: obj, caja, 6 landmarks | BCE (o focal) + Huber | AP@0.5, F1, NME |
| B. Vasos de retina | U-Net, encoder ResNet50 o EfficientNetB4 | 512×512 o 384×384 | máscara binaria | Dice + BCE | Dice, IoU, HD95 |
| C. Enfermedades | EfficientNetB5 o DenseNet201 | 456×456 (B5) | `Dense(n, sigmoid)` | BCE + `class_weight` / focal | AUC-ROC por clase, F1 |

### Por qué YOLO-lite y no SSD para los ojos

Un rostro tiene casi siempre 2 ojos de tamaño parecido. Una sola rejilla a
stride 16 (14×14 celdas en 224 px) basta. No hace falta la pirámide de anclas
de SSD y el modelo queda en ~2.4 M parámetros, apto para TFLite en móvil. Si
dos ojos caen en la misma celda (rostros muy pequeños) se conserva el mayor y
se registra la colisión en `train_summary.json`. Si son frecuentes, la
solución es subir a 320 px o bajar el stride a 8.

## Riesgos y limitaciones

| Riesgo | Impacto | Mitigación |
| --- | --- | --- |
| No hay dataset real de ojos con landmarks en `data/raw/eyes/` | Bloquea las métricas reales de la Fase 1 | Conseguir o anotar uno (CVAT/Label Studio con 6 keypoints); el pipeline acepta cajas sin landmarks |
| Datasets públicos con licencia solo de investigación | No se pueden usar en un producto | Ver `datasets/README.md`; recolectar datos propios con consentimiento |
| Cambio de dominio (otra cámara u otra población) | Caída fuerte de desempeño | Validación externa por dataset; aumentación de color; calibración por sitio |
| Desbalance (fondo ≫ ojos; enfermedades raras) | Recall bajo | `pos_weight`, focal loss, `class_weight`, oversampling |
| GPU de 6 GB (RTX 3050) | Lotes pequeños en U-Net 512 y EffB5 | Mixed precision, batch 4–8, acumulación de gradientes, 384 px |
| Uso clínico | Regulatorio | Es una herramienta de apoyo; requiere validación clínica y registro (COFEPRIS/FDA) |

## Siguientes pasos inmediatos

1. Colocar un dataset de ojos en `data/raw/eyes/` (ver formato).
2. `python -m src.data.prepare_eyes --preview 16` y revisar `data/augmented/eyes/`.
3. `python -m src.training.train_eye` y `python -m src.evaluation.evaluate_eye`.
