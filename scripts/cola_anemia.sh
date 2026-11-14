#!/bin/bash
# Segmentador de conjuntiva (batch 4) + clasificador de anemia.
cd "$(dirname "$0")/.." || exit 1
py=.venv/bin/python
paso() { echo "=== $(date +%H:%M) $1"; shift; "$@" > "logs/cola/$N.log" 2>&1; local rc=$?; echo "=== $(date +%H:%M) fin $N: $rc"; return $rc; }
N=conjuntiva; paso "segmentador de conjuntiva" $py src/training/train_retina.py --config configs/conjuntiva_unet.yaml || exit 1
N=anemia;     paso "clasificador de anemia" $py -m src.training.train_anemia
