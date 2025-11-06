from sam2.sam2_video_predictor import SAM2VideoPredictor
import torch
import numpy as np
from typing import Dict, Any, List, Optional

from .base import VideoSegmenter
from ..devices import maybe_autocast, is_mps


class SAM2Segmenter(VideoSegmenter):
    def __init__(
        self,
        jpeg_folder: str,
        device: str = "cpu",
        vos_optimized: bool = False,
        apply_post: bool = True,
        model_id: str = "facebook/sam2.1-hiera-large",
        predictor_mode: str = "eval",
    ):
        self.device = device

        # Fully disable compilation on MPS/macOS
        self.predictor: SAM2VideoPredictor = SAM2VideoPredictor.from_pretrained(
            model_id,
            device=device,
            mode=predictor_mode,
            apply_postprocessing=apply_post,
            vos_optimized=vos_optimized,
            compile_image_encoder=False,
            compile_prompt_encoder=False,
            compile_mask_decoder=False,
            compile_memory_attention=False,
            compile_vos=False,
        )

        with torch.inference_mode():
            with maybe_autocast(device):
                self.state = self.predictor.init_state(jpeg_folder)

    def add_boxes_and_propagate(
        self,
        init_frame_idx: int,
        boxes_xyxy: np.ndarray,
        starting_obj_id: int = 0,
        end_frame_idx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Add new prompts at `init_frame_idx` and only propagate from there.
        If `end_frame_idx` is provided, stop there; otherwise go to last frame.
        """
        boxes_xyxy = np.asarray(boxes_xyxy, np.float32)
        if boxes_xyxy.shape[0] == 0:
            return []

        outputs: List[Dict[str, Any]] = []

        def _collect():
            for f_idx, obj_ids, ms in it:
                ms_np = [
                    (m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m))
                    for m in ms
                ]
                outputs.append({
                    "frame_idx": int(f_idx),
                    "object_ids": np.array(obj_ids, np.int32),
                    "masks": ms_np,
                })

        with torch.inference_mode():
            with maybe_autocast(self.device):
                # Add prompts at the requested starting frame
                for local_id, box in enumerate(boxes_xyxy):
                    self.predictor.add_new_points_or_box(
                        self.state,
                        int(init_frame_idx),
                        int(starting_obj_id + local_id),
                        box=box.reshape(1, 4).astype(np.float32),
                    )

                # Use start_frame_idx to only propagate forward from there
                it = self.predictor.propagate_in_video(
                    self.state,
                    reverse=False,
                    start_frame_idx=int(init_frame_idx),
                )
   
                _collect()

        if is_mps(self.device):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        return outputs