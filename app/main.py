from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.core_engine import (
    CPUFaceEngine,
    EngineNotReadyError,
    NoFaceDetectedError,
)
from app.storage import new_id, safe_resolve


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage_root = Path(os.environ.get("STORAGE_ROOT", "/storage"))
    for sub in ("templates", "extracted", "outputs"):
        (storage_root / sub).mkdir(parents=True, exist_ok=True)
    app.state.storage_root = storage_root
    app.state.engine = None

    def _load() -> None:
        app.state.engine = CPUFaceEngine(
            det_size=int(os.environ.get("DET_SIZE", "640")),
            max_detect_width=int(os.environ.get("MAX_DETECT_WIDTH", "2000")),
            intra_op_threads=int(os.environ.get("INTRA_OP_THREADS", "4")),
            inter_op_threads=int(os.environ.get("INTER_OP_THREADS", "2")),
            det_thresh=float(os.environ.get("DET_THRESH", "0.3")),
        )

    threading.Thread(target=_load, daemon=True).start()
    yield


app = FastAPI(title="face-swapper-cpu", lifespan=lifespan)


def get_engine(request: Request) -> CPUFaceEngine:
    eng = request.app.state.engine
    if eng is None or not eng.ready:
        raise HTTPException(503, "engine not ready")
    return eng


def get_storage_root(request: Request) -> Path:
    return request.app.state.storage_root


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x1: int
    y1: int
    x2: int
    y2: int


class TemplateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    template_id: str
    path: str


class ExtractResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    extracted_face_id: str
    faces_found: int
    bounding_box: BoundingBox


class MergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    template_id: str = Field(min_length=1)
    extracted_face_id: str = Field(min_length=1)


class MergeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    output_id: str
    retrieval_url: str
    processing_time_seconds: float


@app.exception_handler(NoFaceDetectedError)
async def _no_face_handler(_: Request, exc: NoFaceDetectedError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(EngineNotReadyError)
async def _not_ready_handler(_: Request, exc: EngineNotReadyError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def _missing_handler(_: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"not found: {exc}"})


@app.exception_handler(ValueError)
async def _value_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> dict:
    eng = request.app.state.engine
    if eng is None or not eng.ready:
        raise HTTPException(503, "not ready")
    return {"status": "ready"}


@app.post("/api/v1/templates", response_model=TemplateResponse)
async def upload_template(
    file: UploadFile = File(...),
    storage_root: Path = Depends(get_storage_root),
) -> TemplateResponse:
    template_id = new_id("tpl")
    dest = storage_root / "templates" / template_id
    dest.write_bytes(await file.read())
    return TemplateResponse(
        status="success", template_id=template_id, path=str(dest)
    )


@app.post("/api/v1/faces/extract", response_model=ExtractResponse)
async def extract_face(
    file: UploadFile = File(...),
    engine: CPUFaceEngine = Depends(get_engine),
    storage_root: Path = Depends(get_storage_root),
) -> ExtractResponse:
    crop_bytes, embedding, bbox = engine.extract_face(await file.read())
    face_id = new_id("face_user")
    jpg_path = storage_root / "extracted" / face_id
    jpg_path.write_bytes(crop_bytes)
    # Sidecar .npy holds the recognition embedding; merge reads this so we
    # never have to re-detect on the tight face crop.
    np.save(jpg_path.with_suffix(".npy"), embedding)
    return ExtractResponse(
        status="success",
        extracted_face_id=face_id,
        faces_found=1,
        bounding_box=BoundingBox(**bbox),
    )


@app.post("/api/v1/process/merge", response_model=MergeResponse)
async def merge(
    body: MergeRequest,
    engine: CPUFaceEngine = Depends(get_engine),
    storage_root: Path = Depends(get_storage_root),
) -> MergeResponse:
    template_path = safe_resolve(storage_root, "templates", body.template_id)
    face_path = safe_resolve(storage_root, "extracted", body.extracted_face_id)
    embedding_path = face_path.with_suffix(".npy")
    if not embedding_path.is_file():
        raise FileNotFoundError(f"embedding missing for {body.extracted_face_id}")
    embedding = np.load(embedding_path)

    start = time.perf_counter()
    output_bytes = engine.swap_with_embedding(
        template_path.read_bytes(), embedding
    )
    elapsed = time.perf_counter() - start

    output_id = new_id("merged_out")
    dest = storage_root / "outputs" / output_id
    dest.write_bytes(output_bytes)

    return MergeResponse(
        status="success",
        output_id=output_id,
        retrieval_url=str(dest),
        processing_time_seconds=round(elapsed, 2),
    )
