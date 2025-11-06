from typing import Tuple
import numpy as np

class Detector:
    """Interface for per-frame object detection."""
    def detect_bgr(self, frame_bgr) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          boxes_xyxy: (N,4) float32
          scores:     (N,)  float32
          classes:    (N,)  int32
        """
        raise NotImplementedError
