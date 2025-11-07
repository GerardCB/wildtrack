import os, shutil, tempfile, zipfile
import cv2, numpy as np
from importlib.metadata import version, PackageNotFoundError
from megadetector.detection import run_detector as md_run

from .base import Detector


# ---------- config knobs (env overrides) ----------
_ENV_BACKEND  = os.getenv("WILDTRACK_MD_BACKEND", "auto").lower()  # auto|5|10
_ENV_MD_MODEL = os.getenv("WILDTRACK_MD_MODEL", "MDV5A")           # name or absolute path


# ---------- utils ----------
def _md_cache_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "megadetector_models")

def _delete_md_cache():
    try:
        shutil.rmtree(_md_cache_dir(), ignore_errors=True)
    except Exception:
        pass

def _md_major_version() -> int:
    try:
        v = version("megadetector")
        return int(v.split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        return 5


# ---------- adapters to normalize outputs ----------
class _MD5Adapter:
    def __init__(self, model_name_or_path=_ENV_MD_MODEL):
        self.det = md_run.load_detector(model_name_or_path)

    def detect_rgb(self, image_rgb: np.ndarray):
        """
        Returns (xyxy np.float32 [N,4], scores np.float32 [N], classes np.int32 [N])
        """
        out = self.det.generate_detections_one_image(image_rgb)
        dets = out.get("detections", []) if isinstance(out, dict) else out

        H, W = image_rgb.shape[:2]
        rows = []
        for d in dets:
            cat = d.get("category", d.get("class", None))
            try:
                cat = int(cat) if cat is not None else -1
            except Exception:
                # sometimes category can be "1"|"2"|"3"
                cat = int(str(cat)) if cat is not None else -1

            conf = float(d.get("conf", d.get("score", 0.0)))
            bbox = d.get("bbox") or d.get("box") or d.get("b", None)
            if bbox is None or len(bbox) != 4:
                continue

            x0_rel, y0_rel, w_rel, h_rel = bbox
            x0, y0 = x0_rel * W, y0_rel * H
            x1, y1 = (x0_rel + w_rel) * W, (y0_rel + h_rel) * H
            rows.append((conf, [x0, y0, x1, y1], cat))

        if not rows:
            return _empty_outputs()

        rows.sort(key=lambda r: r[0], reverse=True)
        xyxy   = np.array([r[1] for r in rows], np.float32)
        scores = np.array([r[0] for r in rows], np.float32)
        classes= np.array([r[2] for r in rows], np.int32)
        return xyxy, scores, classes


class _MD10Adapter:
    def __init__(self, model_name_or_path=_ENV_MD_MODEL):
        self.det = md_run.load_detector(model_name_or_path)

    def detect_rgb(self, image_rgb: np.ndarray):
        out = self.det.generate_detections_one_image(image_rgb)
        dets = out.get("detections", []) if isinstance(out, dict) else out

        H, W = image_rgb.shape[:2]
        rows = []
        for d in dets:
            # try a set of common keys across versions
            cat = d.get("category", d.get("class_id", d.get("class", None)))
            try:
                cat = int(cat) if cat is not None else -1
            except Exception:
                cat = int(str(cat)) if cat is not None else -1

            conf = float(d.get("conf", d.get("score", d.get("confidence", 0.0))))

            # bbox can be normalized xywh or absolute xyxy; detect which one
            bbox = d.get("bbox") or d.get("box") or d.get("b", None)
            if bbox is None or len(bbox) != 4:
                continue

            x0, y0, x1, y1 = _coerce_bbox_to_xyxy(bbox, W, H, assume_normalized_xywh=True)
            rows.append((conf, [x0, y0, x1, y1], cat))

        if not rows:
            return _empty_outputs()

        rows.sort(key=lambda r: r[0], reverse=True)
        xyxy   = np.array([r[1] for r in rows], np.float32)
        scores = np.array([r[0] for r in rows], np.float32)
        classes= np.array([r[2] for r in rows], np.int32)
        return xyxy, scores, classes


def _empty_outputs():
    return (np.empty((0, 4), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.int32))

def _coerce_bbox_to_xyxy(bbox, W, H, assume_normalized_xywh=True):
    """
    bbox: [x,y,w,h] (normalized) -> xyxy in absolute pixels
          or already absolute [x0,y0,x1,y1] (we try to detect scale)
    """
    x0, y0, a, b = bbox
    # Heuristic: if values look <=1, treat as normalized xywh
    if assume_normalized_xywh and max(x0, y0, a, b) <= 1.00001:
        x1 = (x0 + a) * W
        y1 = (y0 + b) * H
        x0 = x0 * W
        y0 = y0 * H
    else:
        # already absolute; assume it's xyxy or xywh; try to infer
        if a > x0 and b > y0:
            # looks like xyxy
            x1, y1 = a, b
        else:
            # fallback: treat as xywh in pixels
            x1 = x0 + a
            y1 = y0 + b
    return float(x0), float(y0), float(x1), float(y1)


# ---------- singleton loader with cache-retry ----------
_MD_BACKEND = None  # instance of adapter

def _load_backend():
    global _MD_BACKEND
    if _MD_BACKEND is not None:
        return _MD_BACKEND

    # Choose backend
    major = _md_major_version()
    backend = _ENV_BACKEND
    if backend == "auto":
        backend = "10" if major >= 10 else "5"

    Adapter = _MD10Adapter if backend == "10" else _MD5Adapter

    # Load with one retry if cached weights are corrupt
    try:
        _MD_BACKEND = Adapter(_ENV_MD_MODEL)
        return _MD_BACKEND
    except zipfile.BadZipFile:
        _delete_md_cache()
        _MD_BACKEND = Adapter(_ENV_MD_MODEL)
        return _MD_BACKEND


# ---------- public detector ----------
class MegaDetectorV5(Detector):
    """
    Megadetector wrapper that works with MD 5.x and 10.x.
    Filters to animals-only when animals_only=True (MD category '1').
    """

    def __init__(self, conf_thresh: float = 0.4, animals_only: bool = True):
        self.conf_thresh = float(conf_thresh)
        self.animals_only = bool(animals_only)

    def detect_bgr(self, frame_bgr: np.ndarray):
        if frame_bgr is None or frame_bgr.size == 0:
            return _empty_outputs()

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        backend = _load_backend()
        xyxy, scores, classes = backend.detect_rgb(image_rgb)

        if xyxy.shape[0] == 0:
            return _empty_outputs()

        # filter by class/category (MD: 1=animal, 2=person, 3=vehicle)
        keep = np.ones((xyxy.shape[0],), dtype=bool)
        if self.animals_only:
            keep &= (classes == 1)

        keep &= (scores >= self.conf_thresh)

        if not np.any(keep):
            return _empty_outputs()

        return xyxy[keep], scores[keep], classes[keep]
