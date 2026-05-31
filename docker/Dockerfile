FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    INSWAPPER_MODEL_PATH=/opt/models/inswapper_128.onnx

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-download the buffalo_l detection bundle so cold start needs no network.
RUN python -c "from insightface.app import FaceAnalysis; \
    FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']).prepare(ctx_id=0, det_size=(640,640))"

# inswapper_128.onnx is not on the InsightFace CDN. Place the file at
# ./models/inswapper_128.onnx in the build context before `docker build`.
# Missing file -> COPY fails the build (intentional).
COPY models/inswapper_128.onnx /opt/models/inswapper_128.onnx

COPY app/ /app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
