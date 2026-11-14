#!/bin/bash
# Lo que falta de la cola (el analizador v2 ya terminó).
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs/cola
py=.venv/bin/python
paso() { echo "=== $(date +%H:%M) $1"; shift; "$@" > "logs/cola/$N.log" 2>&1; echo "=== $(date +%H:%M) fin $N: $?"; }
N=eye_cerca;  paso "detector de ojo de cerca" $py src/training/train_eye.py --config configs/eye_cerca.yaml
N=tflite;     paso "detector → TFLite para la app" $py -m src.deployment.export_tflite_ojo
N=conjuntiva; paso "segmentador de conjuntiva" $py src/training/train_retina.py --config configs/conjuntiva_unet.yaml
N=anemia;     paso "clasificador de anemia" $py -m src.training.train_anemia
