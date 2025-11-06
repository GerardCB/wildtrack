from typing import Dict, Any, List
import numpy as np

class VideoSegmenter:
    """Interface for 'seed at frame k, propagate over video' segmentation."""
    def add_boxes_and_propagate(self, init_frame_idx: int, boxes_xyxy: np.ndarray,
                                starting_obj_id: int = 0) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts like:
          { "frame_idx": int, "object_ids": np.ndarray[int32], "masks": List[np.ndarray] }
        """
        raise NotImplementedError
