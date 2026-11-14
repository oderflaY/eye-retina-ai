#!/bin/bash
# Cola de entrenamiento en la GPU (uno tras otro). Log por modelo en logs/cola/.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs/cola
py=.venv/bin/python
paso() { echo "=== $(date +%H:%M) $1"; shift; "$@" > "logs/cola/$N.log" 2>&1; echo "=== $(date +%H:%M) fin $N: $?"; }
N=ojo_v2;     paso "analizador v2" $py -m src.training.train_ojo_externo --red b2 --tam 288 --batch 16 --lr_fino 7e-5 --suavizado 0.1 --epocas 30 --xla --salida models/ojo_externo_v2 --reporte reports/ojo_externo_v2
N=eye_cerca;  paso "detector de ojo de cerca" $py src/training/train_eye.py --config configs/eye_cerca.yaml
N=conjuntiva; paso "segmentador de conjuntiva" $py src/training/train_retina.py --config configs/conjuntiva_unet.yaml
N=anemia;     paso "clasificador de anemia" $py -m src.training.train_anemia
N=tflite;     paso "detector → TFLite para la app" $py -m src.deployment.export_tflite_ojo
