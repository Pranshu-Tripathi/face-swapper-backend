# Deploying face-swapper-backend to Cloud Run

This service runs the face-swap worker behind `shraddha-backend`. It exposes a
single `POST /merge` endpoint that reads template + selfie objects from a GCS
bucket, composites the selfie into the template's white slot, and writes the
result back to GCS.

It does **not** need a FUSE volume mount, public exposure, or the
`inswapper_128.onnx` weights — the Cloud Run image uses
`docker/Dockerfile.cloudrun` (slim, ~530 MB smaller than the local image).

> For local development (full image + legacy multipart endpoints), see
> [`local.md`](./local.md) or the `README.md`.

> Most project-level setup (Artifact Registry repo, buckets, lifecycle rules,
> `shraddha-backend`'s runtime SA) is owned by **`shraddha-backend/discuss/deploy.md`**.
> This doc only covers what's specific to this service. Run §1, §2, §3, §4 of
> that doc first, then come back here.

## 0. Variables

These must match the values used when deploying `shraddha-backend`.

```bash
export PROJECT_ID="shraddha-prod"
export REGION="asia-south1"
export SERVICE_NAME="face-swapper-backend"
export AR_REPO="apps"

export FACESWAP_BUCKET="${PROJECT_ID}-face-swap"
export FACESWAP_SA="face-swapper@${PROJECT_ID}.iam.gserviceaccount.com"

# Created in shraddha-backend/discuss/deploy.md §1.5 — this service's runtime SA.
# Created in shraddha-backend/discuss/deploy.md §1.3 — shared face-swap bucket.

gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"
```

## 1. IAM bindings for `face-swapper`

`shraddha-backend/discuss/deploy.md §1.7` already grants
`roles/storage.objectUser` on the face-swap bucket. If you skipped that doc,
run it now:

```bash
gcloud storage buckets add-iam-policy-binding "gs://${FACESWAP_BUCKET}" \
  --member="serviceAccount:${FACESWAP_SA}" \
  --role="roles/storage.objectUser"
```

The runtime SA needs *read* on `templates/` and `uploads/`, and *write* on
`merges/`. `storage.objectUser` covers all three.

## 2. Build and push the slim image

From this repo root:

```bash
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}:$(git rev-parse --short HEAD)"

gcloud builds submit \
  --config docker/cloudbuild.cloudrun.yaml \
  --substitutions=_IMAGE="$IMAGE" \
  .
```

Why a `cloudbuild.yaml` instead of `--tag`: `gcloud builds submit --tag`
always uses `./Dockerfile` and rejects `--file`. The config file at
`docker/cloudbuild.cloudrun.yaml` points the Docker step at
`docker/Dockerfile.cloudrun` and pins `machineType: E2_HIGHCPU_8` +
`timeout: 1800s`. The trailing `.` is still required — it sets the build
context root so `COPY app/` resolves.

Notes:
- The slim Dockerfile skips the `COPY models/inswapper_128.onnx` step
  (~530 MB) because slot-replace doesn't need the swap model. Use the
  default `docker/Dockerfile` only for local / k8s where the legacy
  `/api/v1/process/merge` is exercised.
- The buffalo_l detection bundle (~300 MB) is still baked in — it's needed
  to crop the selfie before compositing.

## 3. Deploy to Cloud Run

```bash
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --service-account "$FACESWAP_SA" \
  --no-allow-unauthenticated \
  --execution-environment gen2 \
  --concurrency 4 \
  --cpu 2 --memory 4Gi \
  --cpu-boost \
  --min-instances 0 --max-instances 5 \
  --timeout 300s \
  --port 8080
```

Why each flag:

- **`--no-allow-unauthenticated`** — only callers with `roles/run.invoker`
  can reach it. `shraddha-backend`'s runtime SA gets that binding in
  `shraddha-backend/discuss/deploy.md §1.8`, which **must be re-run after
  this service exists** so the resource reference resolves.
- **`--concurrency 4`** — swap is CPU-bound. With 2 vCPUs, 4 in-flight
  requests is the realistic ceiling before latency degrades. Tune up if
  you bump CPU.
- **`--cpu 2 --memory 4Gi`** — model + ORT arenas + transient image
  buffers can push past 2 GiB on large images. If you see SIGKILL in
  logs, bump memory before bumping concurrency.
- **`--cpu-boost`** — extra CPU during startup so the ~10 s buffalo_l
  load (the swapper isn't loaded in this image) doesn't bleed into the
  first request.
- **`--min-instances 0`** — fine for dev. Set to `1` in prod once you
  want to avoid cold starts. The lazy load in `app/main.py:30` runs in a
  daemon thread, so `/readyz` returns 503 for ~10 s after a cold start;
  callers see 503 until then.
- **`--timeout 300s`** — first-request worst case (cold start + large
  image) can run ~30 s. 300 s leaves headroom.
- **`--port 8080`** — matches the slim Dockerfile's `EXPOSE 8080` and
  the `$PORT` env var Cloud Run injects.

Get the URL:

```bash
gcloud run services describe "$SERVICE_NAME" --format='value(status.url)'
```

## 4. Wire it into `shraddha-backend`

Two follow-ups in the backend project after this service exists:

1. **Re-run `shraddha-backend/discuss/deploy.md §1.8`** to grant the
   backend SA `roles/run.invoker` on this service:

   ```bash
   gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
     --member="serviceAccount:shraddha-backend@${PROJECT_ID}.iam.gserviceaccount.com" \
     --role="roles/run.invoker"
   ```

2. **Update `FACE_SWAP_SERVICE_URL`** on the backend:

   ```bash
   FACESWAP_URL=$(gcloud run services describe "$SERVICE_NAME" --format='value(status.url)')
   gcloud run services update shraddha-backend \
     --update-env-vars "FACE_SWAP_SERVICE_URL=${FACESWAP_URL}"
   ```

## 5. Contract

```
POST {service_url}/merge
Authorization: Bearer <google-id-token, audience = service_url>
Content-Type: application/json

{
  "bucket": "<faceswap-bucket>",
  "template_object": "templates/<id>",
  "selfie_object":   "uploads/yyyy/mm/dd/<uuid>.<ext>",
  "output_object":   "merges/yyyy/mm/dd/<uuid>.jpg"
}
```

Success: 200 with `{"status":"ok","processing_time_seconds":N.N}`. The merge
bytes are written by this service to `gs://<bucket>/<output_object>` —
`shraddha-backend` signs a read URL for the mobile client to fetch.

Error semantics (handled by FastAPI exception handlers in `app/main.py`):
- `422` — no face detected in the selfie or no white slot in the template
- `404` — template/selfie object missing in GCS
- `503` — engine still loading (cold start)

## 6. Smoke test

This service refuses public traffic, so easiest path is to drive it through
`shraddha-backend`. From any machine with `curl`:

```bash
BACKEND_URL=$(gcloud run services describe shraddha-backend --format='value(status.url)')

# 1. Get an upload URL for the selfie
TICKET=$(curl -fsS -X POST "${BACKEND_URL}/v1/face-swap/upload-url" \
  -H 'Content-Type: application/json' -d '{"content_type":"image/jpeg"}')
PUT=$(echo "$TICKET" | jq -r .signed_put_url)
SELFIE=$(echo "$TICKET" | jq -r .selfie_id)

# 2. Upload the selfie directly to GCS
curl -fsS -X PUT -H 'Content-Type: image/jpeg' --data-binary @selfie.jpg "$PUT"

# 3. Trigger the merge — this calls face-swapper-backend internally
curl -fsS -X POST "${BACKEND_URL}/v1/face-swap/merge" \
  -H 'Content-Type: application/json' \
  -d "{\"template_id\":\"festival.jpg\",\"selfie_id\":\"${SELFIE}\"}" \
  | jq .
```

To hit this service directly (debugging only), mint an ID token yourself:

```bash
URL=$(gcloud run services describe "$SERVICE_NAME" --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token --audiences="$URL")

curl -fsS -X POST "${URL}/merge" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"bucket\":\"${FACESWAP_BUCKET}\",\"template_object\":\"templates/festival.jpg\",\"selfie_object\":\"uploads/test.jpg\",\"output_object\":\"merges/test.jpg\"}"
```

Your user account needs `roles/run.invoker` on the service for that to work.

## 7. Common gotchas

- **`401 Unauthorized` from `/merge`**: caller didn't attach the ID token,
  or the audience doesn't match the service URL. `shraddha-backend`'s
  `HTTPFaceSwapper` mints the token with `audience = FACE_SWAP_SERVICE_URL`
  — if that env var is wrong (e.g. a stale URL after redeploy), the token
  is valid but rejected.
- **`403` from `/merge`**: caller's SA lacks `roles/run.invoker` on this
  service. Re-run §4.1.
- **GCS `403` inside `/merge`**: this service's runtime SA lacks
  `roles/storage.objectUser` on the face-swap bucket. See §1.
- **`422 no face detected in template`**: the template doesn't have a
  white circular slot that `detect_white_slot` recognizes. Tune
  `white_threshold` / `min_radius_frac` in `core_engine.py:177`, or fix
  the template.
- **`503 not ready` for 10-20s after deploy**: cold start. The buffalo_l
  model loads in a background thread on app startup. Set
  `--min-instances 1` to avoid this in prod.
- **Image push fails**: usually missing AR writer permission on the
  Cloud Build SA. See `shraddha-backend/discuss/deploy.md` §1 — same
  fix applies here.

## 8. Tear down

```bash
gcloud run services delete "$SERVICE_NAME" --region="$REGION"
# Keep the bucket and SA — they're shared with shraddha-backend.
```
