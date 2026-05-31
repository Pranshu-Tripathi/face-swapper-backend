# face-swapper-backend

CPU-only face swap API. FastAPI + InsightFace + ONNX Runtime. Designed to run
on a single Kubernetes node, but local-dev workflows are first-class — see
the three paths below from fastest to most production-like.

The full design is in [`architecture.md`](./architecture.md).

## Prerequisites

- Python 3.11+ (only for path 1)
- Docker (paths 2 and 3)
- `kind` or `minikube` or docker-desktop with kubernetes enabled (path 3 only)
- The `inswapper_128.onnx` model file (~530 MB). It is **not** on the
  InsightFace CDN — fetch it from a HuggingFace mirror and place it at
  `./models/inswapper_128.onnx` before building the image or starting the
  service.

## Path 1 — Run the FastAPI app directly

Fastest iteration loop. No Docker, no cluster.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt

export INSWAPPER_MODEL_PATH=$PWD/models/inswapper_128.onnx
export STORAGE_ROOT=$PWD/.storage          # local stand-in for the PVC

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The first request after startup waits ~15–25 s while models load.
`GET /healthz` responds immediately; `GET /readyz` flips to 200 once the
engine is ready.

### Smoke test

```bash
curl localhost:8000/healthz
curl localhost:8000/readyz

TPL=$(curl -sF file=@some-template.jpg localhost:8000/api/v1/templates | jq -r .template_id)
FACE=$(curl -sF file=@some-selfie.jpg  localhost:8000/api/v1/faces/extract | jq -r .extracted_face_id)
curl -sH 'content-type: application/json' \
  -d "{\"template_id\":\"$TPL\",\"extracted_face_id\":\"$FACE\"}" \
  localhost:8000/api/v1/process/merge
# the merged image lands in $STORAGE_ROOT/outputs/
```

## Path 2 — Run in Docker

Closer to prod. Verifies the image builds and the container runs cleanly.

```bash
# weights must already be at ./models/inswapper_128.onnx
docker build -f docker/Dockerfile -t face-swapper-cpu:dev .

mkdir -p /tmp/face-storage
docker run --rm -p 8000:8000 \
  -v /tmp/face-storage:/storage \
  face-swapper-cpu:dev
```

The same curl smoke from path 1 works against `localhost:8000`.

## Path 3 — Run in a local Kubernetes cluster

Full stack. Uses the manifests in `k8s/`, which are tuned for local clusters
(small CPU/memory requests, default storage class, local image tag).

```bash
# 0. one-time: spin up a cluster
kind create cluster --name face-swapper

# 1. build the image
docker build -f docker/Dockerfile -t face-swapper-cpu:dev .

# 2. load it into the cluster
#    kind:           kind load docker-image face-swapper-cpu:dev --name face-swapper
#    minikube:       minikube image load face-swapper-cpu:dev
#    docker-desktop: nothing — the local daemon's images are visible

kind load docker-image face-swapper-cpu:dev --name face-swapper

# 3. apply everything
kubectl apply -k k8s/

# 4. wait for rollout (model load is ~20 s; readinessProbe gates traffic)
kubectl -n face-swapper rollout status deploy/face-swapper

# 5. port-forward (ingress is excluded from kustomization on purpose)
kubectl -n face-swapper port-forward svc/face-swapper 8000:80
```

Smoke test as in path 1.

To tear down: `kubectl delete -k k8s/` (or `kind delete cluster --name face-swapper`).

## Endpoints

| Method | Path | Body | Purpose |
|---|---|---|---|
| GET | `/healthz` | — | Liveness. Returns 200 immediately. |
| GET | `/readyz` | — | Readiness. 503 until the engine is loaded, then 200. |
| POST | `/api/v1/templates` | multipart `file` | Register a background template. |
| POST | `/api/v1/faces/extract` | multipart `file` | Detect + crop the largest face from an upload. |
| POST | `/api/v1/process/merge` | `{template_id, extracted_face_id}` | Run swap+blend, return output id. |

## Repo layout

```
app/             FastAPI service + CV/ONNX engine
  core_engine.py   CPUFaceEngine — detect, extract, swap, blend
  main.py          endpoints, lifespan, dependency wiring
  storage.py       id generation + path-traversal guard
docker/          Dockerfile (slim Python 3.11 + libgl + pre-downloaded buffalo_l)
k8s/             namespace/configmap/pvc/deployment/service/ingress/cleanup-cron
                 + kustomization.yaml (excludes ingress for local)
models/          drop inswapper_128.onnx here (gitignored)
```

## Configuration

Environment variables (read by `app/main.py` at startup, defaults shown):

| Var | Default | Purpose |
|---|---|---|
| `STORAGE_ROOT` | `/storage` | Root for `templates/`, `extracted/`, `outputs/`. |
| `INSWAPPER_MODEL_PATH` | `/opt/models/inswapper_128.onnx` | Path to the swap model weights. |
| `DET_SIZE` | `640` | InsightFace detector input size. |
| `MAX_DETECT_WIDTH` | `1080` | Detection runs on a downscaled copy above this width. |
| `INTRA_OP_THREADS` | `4` | ONNX intra-op threads. |
| `INTER_OP_THREADS` | `2` | ONNX inter-op threads. |

In the k8s deployment these come from `k8s/configmap.yaml` via `envFrom`.
