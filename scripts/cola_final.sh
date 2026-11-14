#!/bin/bash
# Últimos pasos de la cola (tras el congelamiento de las 13:23).
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs/cola
py=.venv/bin/python
paso() { echo "=== $(date +%H:%M) $1"; shift; "$@" > "logs/cola/$N.log" 2>&1; echo "=== $(date +%H:%M) fin $N: $?"; }
N=conjuntiva; paso "segmentador de conjuntiva" $py src/training/train_retina.py --config configs/conjuntiva_unet.yaml
N=anemia;     paso "clasificador de anemia" $py -m src.training.train_anemia
