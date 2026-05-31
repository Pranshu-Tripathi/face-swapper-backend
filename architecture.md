# CPU Face Swapper — Kubernetes Node Deployment Plan

This plan targets a single-node (or small node-pool) Kubernetes deployment. The service is CPU-bound, stateful in its on-disk cache between endpoint hops, and must coexist with other workloads without monopolizing the node's cores.

---

## 📂 Project Structure & Repo Layout

```text
face-swapper-cpu/
├── app/
│   ├── main.py                 # FastAPI endpoints and orchestration
│   ├── core_engine.py          # Isolated CV/ONNX pipeline & CPU thread configs
│   └── requirements.txt        # Python dependencies
├── docker/
│   └── Dockerfile              # Container image build
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml          # Tunables (thread counts, det_size, etc.)
│   ├── pvc.yaml                # PersistentVolumeClaim for /storage
│   ├── deployment.yaml         # Pod spec with CPU limits & probes
│   ├── service.yaml            # ClusterIP service
│   └── ingress.yaml            # Optional external exposure
└── README.md
```

At runtime the pod mounts a `PersistentVolumeClaim` at `/storage`, giving:

```text
/storage/
├── templates/   # Source background frames (Endpoint 1)
├── extracted/   # Isolated user facial crops (Endpoint 2)
└── outputs/     # Completed composition results (Endpoint 3)
```

This replaces the bare local-disk approach. Asset hand-off between endpoints still happens via the filesystem (avoiding large binary blobs over HTTP), but the volume survives pod restarts and is decoupled from the container's writable layer.

---

## 🐳 Container Image

The image bundles the FastAPI app, OpenCV, InsightFace, and the ONNX models. Bake the models into the image (or pull via an init container) so cold start doesn't depend on network.

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./
# Pre-download InsightFace models so the pod starts offline-capable
RUN python -c "from insightface.app import FaceAnalysis; \
    FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']).prepare(ctx_id=0, det_size=(640,640))"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

Notes:

- Single Uvicorn worker per pod — model memory is large and threading is already saturated by ONNX intra-op parallelism. Scale by replicas, not workers.
- `OMP_NUM_THREADS` / `MKL_NUM_THREADS` env vars match the ONNX thread cap so BLAS libraries don't fight for cores.

---

## ⚙️ CPU Performance Tuning (K8s-aligned)

Resource requests/limits in the Deployment must agree with the in-process thread caps — otherwise the kernel CFS throttles the pod and tail latencies spike.

| Setting | Value | Where |
|---|---|---|
| `resources.requests.cpu` | `2` | Deployment |
| `resources.limits.cpu` | `4` | Deployment |
| `resources.requests.memory` | `2Gi` | Deployment |
| `resources.limits.memory` | `4Gi` | Deployment |
| `intra_op_num_threads` | `4` | ConfigMap → env |
| `inter_op_num_threads` | `2` | ConfigMap → env |
| `OMP_NUM_THREADS` | `4` | Dockerfile / env |

### ONNX Runtime Session Config

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.inter_op_num_threads = 2
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
```

### Downscaled Inference Input

High-resolution photos (e.g., 4K images) crawl on CPU. The detection pipeline auto-downscales wide images to a max width of **640px** or **1080px** strictly during detection, mapping coordinates back to the original full-size array for cropping/swapping.

### In-Memory Buffer Transfers

Within a single request, decode uploads directly from memory via `cv2.imdecode` on the raw byte stream — only persist to the PVC at endpoint boundaries.

---

## 💾 Storage: PersistentVolumeClaim

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: face-swapper-storage
  namespace: face-swapper
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

- `ReadWriteOnce` is sufficient for a single-replica deployment. If you scale beyond one pod, switch to `ReadWriteMany` (NFS / CephFS / EFS) or move asset hand-off to object storage (S3/MinIO) keyed by `template_id` / `extracted_face_id`.
- Add a CronJob to prune `/storage/outputs/` and `/storage/extracted/` older than N hours so the PVC doesn't fill.

---

## 📋 Endpoint Design Specifications

### 🟩 Endpoint 1 — Template Registry

- **Route:** `POST /api/v1/templates`
- **Content-Type:** `multipart/form-data`
- **Input:** `file: UploadFile`

**Flow:** Receives a base template image, generates a unique ID, writes to `/storage/templates/`.

```json
{
  "status": "success",
  "template_id": "tpl_20260531_141022.jpg",
  "path": "/storage/templates/tpl_20260531_141022.jpg"
}
```

### 🟩 Endpoint 2 — Face Isolation & Extraction

- **Route:** `POST /api/v1/faces/extract`
- **Content-Type:** `multipart/form-data`
- **Input:** `file: UploadFile`

**Flow:**

1. Load source file into memory.
2. Run InsightFace SCRFD detector (`CPUExecutionProvider`).
3. Select largest detected face.
4. Pad bounding box by **15%** on each side (preserves hair/jaw).
5. Save crop to `/storage/extracted/`.

```json
{
  "status": "success",
  "extracted_face_id": "face_user_b9102.jpg",
  "faces_found": 1,
  "bounding_box": {"x1": 142, "y1": 88, "x2": 450, "y2": 412}
}
```

### 🟩 Endpoint 3 — Orchestrated Blend & Swap

- **Route:** `POST /api/v1/process/merge`
- **Content-Type:** `application/json`

```json
{
  "template_id": "tpl_20260531_141022.jpg",
  "extracted_face_id": "face_user_b9102.jpg"
}
```

**Flow:**

1. Read both assets from `/storage`.
2. Detect facial structural markers on the template.
3. Align extracted face to template's head tilt / yaw.
4. Run `inswapper_128.onnx` on CPU.
5. Blend with `cv2.seamlessClone` (Poisson) for color/illumination match.
6. Persist to `/storage/outputs/`.

```json
{
  "status": "success",
  "output_id": "merged_out_99214.jpg",
  "retrieval_url": "/storage/outputs/merged_out_99214.jpg",
  "processing_time_seconds": 1.42
}
```

### 🩺 Health Endpoints (required for k8s probes)

- `GET /healthz` — process up, returns 200 immediately.
- `GET /readyz` — returns 200 only after `CPUFaceEngine` finishes loading models.

---

## ☸️ Kubernetes Manifests

### Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: face-swapper
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: face-swapper-config
  namespace: face-swapper
data:
  INTRA_OP_THREADS: "4"
  INTER_OP_THREADS: "2"
  DET_SIZE: "640"
  MAX_DETECT_WIDTH: "1080"
  STORAGE_ROOT: "/storage"
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-swapper
  namespace: face-swapper
spec:
  replicas: 1
  strategy:
    type: Recreate          # PVC is RWO; avoid two pods racing for the mount
  selector:
    matchLabels: { app: face-swapper }
  template:
    metadata:
      labels: { app: face-swapper }
    spec:
      containers:
        - name: api
          image: registry.example.com/face-swapper-cpu:0.1.0
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef: { name: face-swapper-config }
          resources:
            requests: { cpu: "2", memory: "2Gi" }
            limits:   { cpu: "4", memory: "4Gi" }
          volumeMounts:
            - { name: storage, mountPath: /storage }
          readinessProbe:
            httpGet: { path: /readyz, port: 8000 }
            initialDelaySeconds: 20
            periodSeconds: 5
          livenessProbe:
            httpGet: { path: /healthz, port: 8000 }
            initialDelaySeconds: 30
            periodSeconds: 15
      volumes:
        - name: storage
          persistentVolumeClaim:
            claimName: face-swapper-storage
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: face-swapper
  namespace: face-swapper
spec:
  type: ClusterIP
  selector: { app: face-swapper }
  ports:
    - port: 80
      targetPort: 8000
```

### Ingress (optional)

Expose externally via your cluster's ingress controller (nginx, Traefik, etc.). Set client-body size large enough for image uploads (e.g. `nginx.ingress.kubernetes.io/proxy-body-size: "25m"`).

---

## 📈 Scaling & Operational Notes

- **Replicas:** keep at `1` while PVC is `ReadWriteOnce`. To scale horizontally, either switch the volume to `ReadWriteMany` or move the asset cache to object storage and key requests by ID.
- **HPA:** CPU-based HPA is only meaningful once storage is shared; otherwise scale vertically by raising `limits.cpu`.
- **Node placement:** add `nodeSelector` / affinity to pin pods to CPU-optimized nodes (e.g. `node.kubernetes.io/instance-type: c6i.xlarge`). Add a toleration if those nodes are tainted.
- **Cold start:** model load takes ~15–25 s. `readinessProbe.initialDelaySeconds` and `Recreate` strategy together prevent traffic from hitting a half-warm pod.
- **Garbage collection:** schedule a CronJob (`find /storage/outputs -mmin +120 -delete`) so the PVC doesn't fill silently.

---

## 🛠 Core Implementation Blueprint

Boilerplate for `core_engine.py` with CPU-optimized configuration:

```python
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class CPUFaceEngine:
    def __init__(self, det_size: int = 640):
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def extract_face(self, img_bytes: bytes):
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        faces = self.app.get(img)
        if not faces:
            raise ValueError("No faces found in uploaded frame.")

        target_face = max(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
        )
        bbox = target_face.bbox.astype(int)

        h, w, _ = img.shape
        pad_x = int((bbox[2] - bbox[0]) * 0.15)
        pad_y = int((bbox[3] - bbox[1]) * 0.15)

        x1 = max(0, bbox[0] - pad_x)
        y1 = max(0, bbox[1] - pad_y)
        x2 = min(w, bbox[2] + pad_x)
        y2 = min(h, bbox[3] + pad_y)

        return img[y1:y2, x1:x2], bbox
```
