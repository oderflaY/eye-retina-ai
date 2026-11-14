#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
py=.venv/bin/python
paso() { echo "=== $(date +%H:%M) $1"; shift; "$@" > "logs/cola/$N.log" 2>&1; local rc=$?; echo "=== $(date +%H:%M) fin $N: $rc"; return $rc; }
mkdir -p reports/conjuntiva; cp models/conjuntiva/config_used.yaml reports/conjuntiva/
N=conjuntiva_eval; paso "evaluar conjuntiva" $py src/evaluation/evaluate_retina.py --config models/conjuntiva/config_used.yaml --model models/conjuntiva/best.keras
N=anemia;          paso "clasificador de anemia" $py -m src.training.train_anemia
