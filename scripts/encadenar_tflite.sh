#!/bin/bash
# Espera a que acabe la cola actual y lanza la del detector v2 con los mismos límites.
while systemctl is-active -q entreno-cola; do sleep 30; done
sleep 20
sudo -n systemctl reset-failed entreno-cola 2>/dev/null
sudo -n systemd-run --unit=entreno-cola --collect -p WorkingDirectory=/home/oderflay/TemrariosSpace/eye-retina-ai -p Nice=-5 -p ManagedOOMPreference=avoid -p MemoryMax=4608M -p MemorySwapMax=1G systemd-inhibit --what=sleep:idle:handle-lid-switch --why="Cola de entrenamiento OjoScan" setpriv --reuid=1000 --regid=1000 --init-groups env HOME=/home/oderflay USER=oderflay PATH=/usr/bin:/bin TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 TF_GPU_ALLOCATOR=cuda_malloc_async /home/oderflay/TemrariosSpace/eye-retina-ai/scripts/cola_tflite.sh
