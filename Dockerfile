# Imagen de inferencia (CPU) para la API FastAPI.
# Construir:  docker build -t eye-retina-ai .
# Ejecutar:   docker run -p 8000:8000 -v $PWD/models:/app/models \
#               -e EYE_MODEL_PATH=models/eye_detection/eye_det_mobilenetv2_224/best.keras eye-retina-ai
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# En el contenedor basta TensorFlow para CPU
RUN pip install --no-cache-dir "tensorflow-cpu>=2.15" numpy pandas scikit-learn \
    opencv-python-headless albumentations pyyaml fastapi "uvicorn[standard]" python-multipart

COPY src ./src
COPY configs ./configs

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
