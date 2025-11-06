import cv2, numpy as np
from megadetector.detection import run_detector as md_run
from .base import Detector

_MD_MODEL = None

def _get_md():
    global _MD_MODEL
    if _MD_MODEL is None:
        _MD_MODEL = md_run.load_detector("MDV5A")
    return _MD_MODEL

class MegaDetectorV5(Detector):
    def __init__(self, conf_thresh: float = 0.4, animals_only: bool = True):
        self.conf_thresh = conf_thresh
        self.animals_only = animals_only

    def detect_bgr(self, frame_bgr):
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        model = _get_md()
        result = model.generate_detections_one_image(image_rgb)
        H, W = image_rgb.shape[:2]
        rows = []
        for det in result["detections"]:
            cat = det.get("category", None)
            conf = float(det.get("conf", 0.0))
            if self.animals_only and str(cat) != "1":  # MD animal category
                continue
            if conf < self.conf_thresh:
                continue
            x0_rel, y0_rel, w_rel, h_rel = det["bbox"]
            x0, y0 = x0_rel*W, y0_rel*H
            x1, y1 = (x0_rel+w_rel)*W, (y0_rel+h_rel)*H
            rows.append((conf, [x0, y0, x1, y1], int(cat) if cat is not None else -1))
        if not rows:
            return np.empty((0,4), np.float32), np.array([]), np.array([])
        rows.sort(key=lambda r: r[0], reverse=True)
        xyxy = np.array([r[1] for r in rows], np.float32)
        scores = np.array([r[0] for r in rows], np.float32)
        classes = np.array([r[2] for r in rows], np.int32)
        return xyxy, scores, classes
