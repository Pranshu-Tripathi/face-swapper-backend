# Running face-swapper-backend locally

Local dev exercises the **full** API surface — including the legacy multipart
endpoints (`/api/v1/templates`, `/api/v1/faces/extract`,
`/api/v1/process/merge`, `/api/v1/process/replace`) — using the original
`docker/Dockerfile`. The slim Cloud Run image (`docker/Dockerfile.cloudrun`)
omits `inswapper_128.onnx`, so the swap path returns 503 there.

For Cloud Run deploy, see [`deploy.md`](./deploy.md). For the k8s path, see
the README.

## Prerequisites

- Python 3.11+ (for the bare `uvicorn` path) or Docker (for the container path)
- The `inswapper_128.onnx` model file (~530 MB). It is **not** on the
  InsightFace CDN — fetch it from a HuggingFace mirror and place it at
  `./models/inswapper_128.onnx` before building or starting.

## Option A — bare `uvicorn` (fastest iteration)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt

export INSWAPPER_MODEL_PATH="$PWD/models/inswapper_128.onnx"
export STORAGE_ROOT="$PWD/.storage"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The first request after startup waits ~15–25 s while models load.
`GET /healthz` responds immediately; `GET /readyz` flips to 200 once the
engine is ready.

## Option B — full Docker image

Closer to prod. Verifies the image builds and the container runs cleanly.

```bash
# weights must already be at ./models/inswapper_128.onnx
docker build -f docker/Dockerfile -t face-swapper-cpu:dev .

mkdir -p /tmp/face-storage
docker run --rm -p 8000:8000 \
  -v /tmp/face-storage:/storage \
  face-swapper-cpu:dev
```

## Smoke test (both options)

The local flow uses the legacy multipart API:

```bash
curl localhost:8000/healthz
curl localhost:8000/readyz

TPL=$(curl -sF file=@some-template.jpg  localhost:8000/api/v1/templates       | jq -r .template_id)
FACE=$(curl -sF file=@some-selfie.jpg    localhost:8000/api/v1/faces/extract   | jq -r .extracted_face_id)
curl -sH 'content-type: application/json' \
  -d "{\"template_id\":\"$TPL\",\"extracted_face_id\":\"$FACE\"}" \
  localhost:8000/api/v1/process/merge
# the merged image lands in $STORAGE_ROOT/outputs/
```

## Testing the GCS `/merge` endpoint locally

The new `POST /merge` endpoint (the one Cloud Run uses) talks directly to
GCS. To exercise it from your laptop:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project "$PROJECT_ID"
# Grant your user roles/storage.objectUser on gs://${PROJECT_ID}-face-swap.

curl -sH 'content-type: application/json' \
  -d '{
    "bucket": "shraddha-prod-face-swap",
    "template_object": "templates/festival.jpg",
    "selfie_object":   "uploads/test.jpg",
    "output_object":   "merges/test.jpg"
  }' \
  localhost:8000/merge
```

Upload a template + selfie to the bucket first, then check that
`gs://${BUCKET}/merges/test.jpg` exists after the call returns 200.

## Configuration

Environment variables (read by `app/main.py` at startup):

| Var | Default | Purpose |
|---|---|---|
| `STORAGE_ROOT` | `/storage` | Root for `templates/`, `extracted/`, `outputs/` (legacy endpoints only). |
| `INSWAPPER_MODEL_PATH` | `/storage/models/inswapper_128.onnx` | Path to the swap model weights. If missing, the swapper isn't loaded and `/api/v1/process/merge` returns 503. |
| `DET_SIZE` | `640` | InsightFace detector input size. |
| `MAX_DETECT_WIDTH` | `2000` | Detection runs on a downscaled copy above this width. |
| `DET_THRESH` | `0.3` | Detection confidence threshold. |
| `INTRA_OP_THREADS` | `4` | ONNX intra-op threads. |
| `INTER_OP_THREADS` | `2` | ONNX inter-op threads. |
