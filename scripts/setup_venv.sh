#!/usr/bin/env bash
# Crea el entorno virtual con TensorFlow + CUDA y corrige la carga de librerías.
#
# TensorFlow 2.21 busca libcusolver.so.11 solo en LD_LIBRARY_PATH y no la
# encuentra dentro de los paquetes nvidia-* del venv, por lo que no registra la
# GPU. Este script instala un .pth que precarga esas librerías al iniciar Python.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
SP=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
cat > "$SP/_nvidia_preload.py" <<'PY'
"""Precarga las librerías CUDA de los paquetes nvidia-* (ver scripts/setup_venv.sh)."""
import ctypes
import glob
import os

_base = os.path.join(os.path.dirname(__file__), "nvidia")
for _name in ("nvjitlink/lib/libnvJitLink.so.12", "cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12",
              "cusparse/lib/libcusparse.so.12", "cusolver/lib/libcusolver.so.11"):
    for _p in glob.glob(os.path.join(_base, _name)):
        try:
            ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
PY
echo "import _nvidia_preload" > "$SP/nvidia_preload.pth"
.venv/bin/python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
