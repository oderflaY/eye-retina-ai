#!/bin/bash
# Segunda vuelta: apertura más larga + conjuntiva a 320 px (candidatos aparte).
cd "$(dirname "$0")/.." || exit 1
py=.venv/bin/python
paso() { echo "=== $(date +%H:%M) $1"; shift; "$@" > "logs/cola/$N.log" 2>&1; local rc=$?; echo "=== $(date +%H:%M) fin $N: $rc"; return $rc; }
N=apertura_v2;   paso "apertura v2" $py -m src.training.train_apertura --epocas 60 --paciencia 12 --alpha 0.75 --salida models/apertura_v2 --reporte reports/apertura_v2
N=conjuntiva_v2; paso "conjuntiva v2 (320 px)" $py src/training/train_retina.py --config configs/conjuntiva_unet_v2.yaml
