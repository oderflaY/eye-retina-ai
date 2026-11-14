# Eye-Retina AI

Sistema en TensorFlow para analizar ojos y retinas.

| Fase | Componente | Estado |
| --- | --- | --- |
| 1 | Detector de un ojo (bbox + 6 landmarks + ojo/no-ojo), `src/models/eye_detector.py` | Implementado · **pendiente de entrenamiento con datos reales** |
| 1 | Detector multi-ojo de rejilla para rostros completos, `src/models/eye_grid_detector.py` | Implementado · pendiente de entrenamiento |
| 1b | Segmentación de vasos (U-Net ResNet50), `src/models/unet.py` | Implementado · entrenado en FIVES, validación externa en HRF |
| 2 | Clasificación multilabel de enfermedades | Parte B |

> Herramienta de investigación. **No es un dispositivo médico** ni sustituye el diagnóstico de un oftalmólogo.

## Estructura

```
eye-retina-ai/
├── configs/            eye.yaml (detector de un ojo) · eye_grid.yaml (multi-ojo) · retina_unet.yaml (vasos)
├── data/raw/eyes/      imágenes + etiquetas (ver docs/FORMATO_ETIQUETAS.md)
├── data/raw/retina/    FIVES, HRF, OCT2017 (enlaces a ../datasets)
├── notebooks/          01_eda_eyes.ipynb
├── src/
│   ├── config.py       rutas, constantes, set_global_seed, .env
│   ├── data/           loader_eyes (EyeDataLoader), augment, eye_labels (autodetección),
│   │                   io, splits, synthetic, + módulos del detector de rejilla
│   │                   retina_preprocess, prepare_retina, loader_retina (Parte B)
│   ├── models/         eye_detector, eye_grid_detector, losses, unet, seg_losses
│   ├── training/       train_eye, train_eye_grid, train_retina
│   ├── evaluation/     evaluate_eye, evaluate_eye_grid, eye_metrics, explain,
│   │                   evaluate_retina, seg_metrics
│   └── deployment/     inference_eye, inference_retina, api (FastAPI), demo (Gradio),
│                       predictor_grid, export_grid
├── tests/              pytest (con datos sintéticos)
├── models/  reports/  logs/
└── requirements.txt  Dockerfile  .env.example
```

## Instalación

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Entrenar el detector de ojos

```bash
python src/training/train_eye.py --config configs/eye.yaml
python src/training/train_eye.py --config configs/eye.yaml --override train.epochs=20 train.batch_size=16
tensorboard --logdir logs/eye
```

Salidas:

| Archivo | Contenido |
| --- | --- |
| `models/eye_best.h5` | Mejor checkpoint (`val_loss`) |
| `models/eye_savedmodel/` | SavedModel float32 para TF Serving o `EyeDetector` |
| `reports/eye_results.json` | Métricas en test |
| `reports/eye/` | Matriz de confusión, curva PR, predicciones, Grad-CAM, peores predicciones |

Para evaluar de nuevo un modelo ya guardado:

```bash
python src/evaluation/evaluate_eye.py --config configs/eye.yaml --model models/eye_best.h5
```

## Segmentación de vasos de retina (Parte B)

Datos: FIVES (800 fondos 2048 px con vasos anotados; normal, retinopatía
diabética, glaucoma, DMAE) y HRF (45 fondos 3504×2336) como validación externa.
Descomprime los archivos de `../datasets` en `extraido/` dentro de cada carpeta.

```bash
python -m src.data.prepare_retina                 # recorte al FOV, cuadrado, 1024 px, CLAHE
python src/training/train_retina.py --config configs/retina_unet.yaml
python src/evaluation/evaluate_retina.py --config configs/retina_unet.yaml   # solo evaluar
```

| División | Origen | Imágenes |
| --- | --- | --- |
| train / val | FIVES train (85 / 15 %, estratificado por enfermedad) | 510 / 90 |
| test | FIVES test oficial | 200 |
| external | HRF completo (otra cámara y otra población) | 45 |

* Modelo: U-Net con encoder ResNet50 (ImageNet), decoder de 5 niveles y una
  rama a resolución completa para vasos finos (32.6 M parámetros).
* Entrena con recortes de 512 px (lote 4, mixed precision) y predice sobre la
  imagen completa de 1024 px. Pérdida BCE + Dice. Checkpoint por `val_dice`.
* El umbral se elige en validación y se guarda en `models/retina/preprocess.json`.
* Métricas dentro del FOV: Dice, IoU, sensibilidad, especificidad, AUC, HD95,
  por enfermedad. Reporte en `reports/retina/retina_results.json`, por imagen en
  `per_image.csv`, figuras de mejores y peores casos en `reports/retina/figures/`.

## Servir la API y la demo

```bash
uvicorn src.deployment.api:app --reload --port 8000     # Swagger en http://localhost:8000/docs
curl -F file=@ojo.jpg "localhost:8000/predict/eye?threshold=0.5"
python src/deployment/demo.py                            # Gradio en http://127.0.0.1:7860
```

| Método | Ruta | Respuesta |
| --- | --- | --- |
| GET | `/` | Estado y disponibilidad de los modelos |
| POST | `/predict/eye` | `{bbox, landmarks[6], confidence, is_eye, image_size, model}` |
| POST | `/predict/eyes` | Todos los ojos de un rostro (requiere el modelo de rejilla) |
| POST | `/predict/retina?include=none\|mask\|overlay\|both` | `{vessel_density, vessel_pixels, fov_pixels, threshold, image_size, model, mask_png?, overlay_png?}` (PNG en base64) |
| POST | `/predict/disease` | 501 hasta la Fase 2 |

Los errores responden `{"error": {"code", "message"}}` con código 400, 413, 500, 501 o 503.

## Sin datos reales todavía

```bash
python -m src.data.synthetic --out data/synthetic/eyes --n 400
python src/training/train_eye.py --override data.data_dir=data/synthetic/eyes model.weights=null train.epochs=3
```

Los datos sintéticos solo verifican que el pipeline funciona. Sus métricas no dicen nada del desempeño real.

## Calidad y reproducibilidad

* Semilla 42 en `random`, NumPy y TF (`set_global_seed`).
* `pytest` corre todas las pruebas con datos sintéticos, sin descargar nada.
* `black .` y `flake8` para el estilo.
