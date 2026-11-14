#!/bin/bash
# Detector de ojo de cerca v2 (etiquetas corregidas: párpados a la mitad del ojo, esquinas por posición).
cd "$(dirname "$0")/.." || exit 1
py=.venv/bin/python
paso() { echo "=== $(date +%H:%M) $1"; shift; "$@" > "logs/cola/$N.log" 2>&1; echo "=== $(date +%H:%M) fin $N: $?"; }
N=eye_cerca_v2; paso "detector de ojo de cerca v2" $py src/training/train_eye.py --config configs/eye_cerca.yaml
N=tflite_v2;    paso "detector v2 → TFLite para la app" $py -m src.deployment.export_tflite_ojo
