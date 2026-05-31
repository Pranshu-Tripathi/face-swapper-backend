"""CPU-only face detection + swap pipeline used by the FastAPI layer.

The `inswapper_128.onnx` model is not bundled with InsightFace's `buffalo_l`
and is no longer on the InsightFace CDN. The operator must supply it; the
engine reads its path from the `inswapper_model_path` argument or the
`INSWAPPER_MODEL_PATH` env var, defaulting to `/storage/models/inswapper_128.onnx`.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


class NoFaceDetectedError(ValueError):
    """No face was found in the input image."""


class EngineNotReadyError(RuntimeError):
    """The engine was used before model loading completed."""


class CPUFaceEngine:
    def __init__(
        self,
        det_size: int = 640,
        max_detect_width: int = 1080,
        intra_op_threads: int = 4,
        inter_op_threads: int = 2,
        inswapper_model_path: str | None = None,
    ):
        self.max_detect_width = max_detect_width
        self._ready = False

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = inter_op_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

        path = (
            inswapper_model_path
            or os.environ.get("INSWAPPER_MODEL_PATH")
            or "/storage/models/inswapper_128.onnx"
        )
        self.swapper = get_model(
            path,
            providers=["CPUExecutionProvider"],
            session_options=opts,
        )

        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def extract_face(
        self, img_bytes: bytes, pad_ratio: float = 0.15
    ) -> tuple[bytes, dict]:
        if not self._ready:
            raise EngineNotReadyError("engine not ready")

        img = _decode(img_bytes)
        face = self._detect_largest(img)
        bbox = face.bbox.astype(int)

        h, w = img.shape[:2]
        pad_x = int((bbox[2] - bbox[0]) * pad_ratio)
        pad_y = int((bbox[3] - bbox[1]) * pad_ratio)
        x1 = max(0, bbox[0] - pad_x)
        y1 = max(0, bbox[1] - pad_y)
        x2 = min(w, bbox[2] + pad_x)
        y2 = min(h, bbox[3] + pad_y)

        crop = img[y1:y2, x1:x2]
        return _encode_jpeg(crop), {
            "x1": int(bbox[0]),
            "y1": int(bbox[1]),
            "x2": int(bbox[2]),
            "y2": int(bbox[3]),
        }

    def swap_and_blend(self, template_bytes: bytes, face_bytes: bytes) -> bytes:
        if not self._ready:
            raise EngineNotReadyError("engine not ready")

        template_img = _decode(template_bytes)
        face_img = _decode(face_bytes)

        target_face = self._detect_largest(template_img)
        source_face = self._detect_largest(face_img)

        swapped = self.swapper.get(
            template_img, target_face, source_face, paste_back=True
        )
        blended = _seamless_blend(template_img, swapped, target_face.bbox)
        return _encode_jpeg(blended)

    def _detect_largest(self, img: np.ndarray):
        h, w = img.shape[:2]
        if w > self.max_detect_width:
            scale = self.max_detect_width / w
            small = cv2.resize(img, (self.max_detect_width, int(h * scale)))
            faces = self.app.get(small)
            for f in faces:
                f.bbox = f.bbox / scale
                if getattr(f, "kps", None) is not None:
                    f.kps = f.kps / scale
        else:
            faces = self.app.get(img)

        if not faces:
            raise NoFaceDetectedError("no faces detected")

        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )


def _decode(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image bytes")
    return img


def _encode_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _seamless_blend(dst: np.ndarray, src: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    h, w = dst.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    mask = np.zeros((h, w), dtype=np.uint8)
    # Erode the rect a few pixels so seamlessClone has clean interior gradients.
    inset = 3
    mask[y1 + inset : y2 - inset, x1 + inset : x2 - inset] = 255

    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
