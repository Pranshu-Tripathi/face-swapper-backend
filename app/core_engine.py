"""CPU-only face detection + swap pipeline used by the FastAPI layer.

The `inswapper_128.onnx` model is not bundled with InsightFace's `buffalo_l`
and is no longer on the InsightFace CDN. The operator must supply it; the
engine reads its path from the `inswapper_model_path` argument or the
`INSWAPPER_MODEL_PATH` env var, defaulting to `/storage/models/inswapper_128.onnx`.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

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
        max_detect_width: int = 2000,
        intra_op_threads: int = 4,
        inter_op_threads: int = 2,
        det_thresh: float = 0.3,
        inswapper_model_path: str | None = None,
    ):
        self.max_detect_width = max_detect_width
        self._ready = False

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = inter_op_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        # det_size=640 matches SCRFD's training resolution; larger values
        # (e.g. 1024) cause anchor-scale mismatch and miss faces entirely.
        # det_thresh lowered from default 0.5 to help printed/screenshot faces.
        self.app.prepare(
            ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh
        )

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
        self, img_bytes: bytes, pad_ratio: float = 0.7
    ) -> tuple[bytes, np.ndarray, dict]:
        # pad_ratio 0.7 widens the saved JPEG to head+shoulders so the
        # /replace composite fills the slot like a portrait, not a passport
        # zoom. The recognition embedding is unaffected — /merge keeps using
        # the embedding directly.
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
        return _encode_jpeg(crop), face.normed_embedding, {
            "x1": int(bbox[0]),
            "y1": int(bbox[1]),
            "x2": int(bbox[2]),
            "y2": int(bbox[3]),
        }

    def swap_with_embedding(
        self, template_bytes: bytes, source_embedding: np.ndarray
    ) -> bytes:
        if not self._ready:
            raise EngineNotReadyError("engine not ready")

        template_img = _decode(template_bytes)
        try:
            target_face = self._detect_largest(template_img)
        except NoFaceDetectedError as e:
            raise NoFaceDetectedError("no face detected in template image") from e

        # inswapper only reads .normed_embedding from source_face; bbox/kps
        # come from target_face. Synthesizing a Face-shaped object lets us
        # skip a second detection pass on a tight crop.
        source_face = SimpleNamespace(normed_embedding=source_embedding)
        swapped = self.swapper.get(
            template_img, target_face, source_face, paste_back=True
        )
        blended = _seamless_blend(template_img, swapped, target_face.bbox)
        return _encode_jpeg(blended)

    def replace_in_slot(
        self, template_bytes: bytes, face_crop_bytes: bytes
    ) -> bytes:
        if not self._ready:
            raise EngineNotReadyError("engine not ready")

        template_img = _decode(template_bytes)
        slot = detect_white_slot(template_img)
        if slot is None:
            raise NoFaceDetectedError("no user slot detected in template")

        face_crop = _decode(face_crop_bytes)
        cx, cy, r = slot
        composited = composite_into_slot(template_img, face_crop, cx, cy, r)
        return _encode_jpeg(composited)

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


def detect_white_slot(
    img: np.ndarray,
    min_radius_frac: float = 0.08,
    white_threshold: int = 235,
    circularity_min: float = 0.7,
) -> tuple[int, int, int] | None:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_radius = int(min(h, w) * min_radius_frac)

    best: tuple[int, int, int] | None = None
    best_area = 0.0
    for c in contours:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < min_radius:
            continue
        area = cv2.contourArea(c)
        circle_area = np.pi * r * r
        if circle_area == 0 or area / circle_area < circularity_min:
            continue
        if area > best_area:
            best_area = area
            best = (int(round(cx)), int(round(cy)), int(round(r)))
    return best


def composite_into_slot(
    template: np.ndarray,
    face_crop: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    feather: int = 8,
) -> np.ndarray:
    size = 2 * r
    fh, fw = face_crop.shape[:2]
    scale = size / min(fh, fw)
    resized = cv2.resize(
        face_crop, (int(round(fw * scale)), int(round(fh * scale)))
    )
    rh, rw = resized.shape[:2]
    x0 = (rw - size) // 2
    y0 = (rh - size) // 2
    crop = resized[y0 : y0 + size, x0 : x0 + size]

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (r, r), max(1, r - feather), 255, -1)
    ksize = 2 * feather + 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    alpha = mask.astype(np.float32) / 255.0

    out = template.copy()
    th, tw = out.shape[:2]
    x1, y1 = cx - r, cy - r
    x2, y2 = x1 + size, y1 + size
    sx1, sy1 = max(0, -x1), max(0, -y1)
    dx1, dy1 = max(0, x1), max(0, y1)
    dx2, dy2 = min(tw, x2), min(th, y2)
    w_eff, h_eff = dx2 - dx1, dy2 - dy1
    if w_eff <= 0 or h_eff <= 0:
        return out

    fg = crop[sy1 : sy1 + h_eff, sx1 : sx1 + w_eff].astype(np.float32)
    bg = out[dy1:dy2, dx1:dx2].astype(np.float32)
    a = alpha[sy1 : sy1 + h_eff, sx1 : sx1 + w_eff, None]
    out[dy1:dy2, dx1:dx2] = (fg * a + bg * (1.0 - a)).astype(np.uint8)
    return out


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
