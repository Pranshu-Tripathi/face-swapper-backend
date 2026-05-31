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
        # auto-composite fallback in swap_or_composite fills the slot like
        # a portrait. The recognition embedding is computed before cropping
        # and is unaffected.
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

    def swap_or_composite(
        self,
        template_bytes: bytes,
        source_embedding: np.ndarray,
        face_crop_bytes: bytes,
    ) -> bytes:
        if not self._ready:
            raise EngineNotReadyError("engine not ready")

        template_img = _decode(template_bytes)
        faces = self._detect_all(template_img)
        human_faces = [f for f in faces if _is_human_face(template_img, f)]

        if human_faces:
            target_face = max(
                human_faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            # inswapper only reads .normed_embedding from source_face; bbox/kps
            # come from target_face. Synthesizing a Face-shaped object lets us
            # skip a second detection pass on a tight crop.
            source_face = SimpleNamespace(normed_embedding=source_embedding)
            swapped = self.swapper.get(
                template_img, target_face, source_face, paste_back=True
            )
            blended = _seamless_blend(template_img, swapped, target_face.bbox)
            return _encode_jpeg(blended)

        slot = detect_white_slot(template_img)
        if slot is not None:
            cx, cy, r = slot
            face_crop = _decode(face_crop_bytes)
            composited = composite_into_slot(template_img, face_crop, cx, cy, r)
            return _encode_jpeg(composited)

        raise NoFaceDetectedError(
            "no human face or user slot detected in template"
        )

    def _detect_all(self, img: np.ndarray):
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
        return faces

    def _detect_largest(self, img: np.ndarray):
        faces = self._detect_all(img)
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


def _is_human_face(
    img: np.ndarray,
    face,
    patch_size: int = 7,
    sat_min: int = 30,
    val_min: int = 40,
) -> bool:
    h, w = img.shape[:2]

    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 5:
        # kps order: left eye, right eye, nose, left mouth, right mouth.
        # Cheek midpoint ≈ halfway between same-side eye and mouth corner.
        sample_points = [(kps[0] + kps[3]) / 2.0, (kps[1] + kps[4]) / 2.0]
    else:
        x1, y1, x2, y2 = face.bbox.astype(int)
        sample_points = [np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])]

    patches = []
    half = patch_size // 2
    for px, py in sample_points:
        cx, cy = int(round(px)), int(round(py))
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half + 1)
        y2 = min(h, cy + half + 1)
        if x2 > x1 and y2 > y1:
            patches.append(img[y1:y2, x1:x2])
    if not patches:
        return False

    combined = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    hsv = cv2.cvtColor(combined.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    mean_h, mean_s, mean_v = hsv.mean(axis=0)

    if mean_s < sat_min or mean_v < val_min:
        return False
    # Human skin sits at the red/orange end of the hue wheel; H wraps at 180
    # in OpenCV, so accept the two bands. Blue Krishna (H≈100–120) and green
    # deities (H≈40–80) fall outside both.
    return mean_h <= 25 or mean_h >= 160


def _seamless_blend(dst: np.ndarray, src: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    # Feathered alpha blend, NOT cv2.seamlessClone. Poisson blending rebalances
    # the swapped face's colors to match the template's surrounding gradient,
    # which visibly dilutes the user's identity (skin tone, lips, brow shading
    # all drift toward the template). A straight alpha composite over the face
    # ellipse keeps the swapper's pixels verbatim; the feather hides the seam.
    h, w = dst.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ax = max(1, int((x2 - x1) * 0.5))
    ay = max(1, int((y2 - y1) * 0.55))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    feather = max(3, min(ax, ay) // 6)
    ksize = feather * 2 + 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]

    return (
        src.astype(np.float32) * alpha + dst.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
