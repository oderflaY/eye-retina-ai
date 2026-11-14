#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
echo "=== $(date +%H:%M) detector v2 → TFLite + evaluación"
.venv/bin/python -m src.deployment.export_tflite_ojo > logs/cola/tflite_v2.log 2>&1; rc=$?
echo "=== $(date +%H:%M) fin tflite_v2: $rc"
