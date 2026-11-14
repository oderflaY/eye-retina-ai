#!/usr/bin/env bash
# Sube el relay al VPS y lo reconstruye. Uso: relay/deploy/publicar.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VPS=root@2.25.68.28
KEY=~/.ssh/id_vps_claude
ssh -i $KEY $VPS 'mkdir -p /root/ojoscan-relay/datos && chown 10001 /root/ojoscan-relay/datos'
scp -i $KEY app.py requirements.txt Dockerfile docker-compose.yml $VPS:/root/ojoscan-relay/
ssh -i $KEY $VPS 'cd /root/ojoscan-relay && docker compose up -d --build && sleep 3 && curl -s https://2.25.68.28:13443/'
# APK más reciente para descargar desde el teléfono: https://2.25.68.28:13443/app
if [[ -f ../mobile/androidApp/build/outputs/apk/debug/androidApp-debug.apk ]]; then
  ssh -i $KEY $VPS 'mkdir -p /root/ojoscan-relay/datos/descargas'
  scp -i $KEY ../mobile/androidApp/build/outputs/apk/debug/androidApp-debug.apk $VPS:/root/ojoscan-relay/datos/descargas/OjoScan.apk.tmp
  ssh -i $KEY $VPS 'cd /root/ojoscan-relay/datos/descargas && mv OjoScan.apk.tmp OjoScan.apk && chown -R 10001 .'
fi
