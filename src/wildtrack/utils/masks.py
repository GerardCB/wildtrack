"""
Mask and box utilities used across the WildTrack pipeline.

This module centralizes helpers for:
- robust mask binarization (supports tensors / logits)
- box <-> mask IoU computations
- deriving tight bbox from a mask
- filtering/assigning detection boxes against existing tracked masks

Keeping these here helps keep the incremental pipeline clean.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ----------------------------- Core mask helpers ----------------------------- #

def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically-stable sigmoid for logits coming from segmentation models.
    Clips large magnitudes to avoid overflow in exp/tanh.
    """
    x = np.clip(x, -60.0, 60.0)
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def mask_to_binary(m, target_hw: Optional[Tuple[int, int]] = None, thresh: float = 0.5) -> np.ndarray:
    """
    Convert an arbitrary mask (bool/uint8/float/logits or torch tensor) to a boolean array.
    Optionally resize to (H, W) = target_hw using nearest-neighbor.

    Args:
        m: mask-like array, possibly a torch tensor or logits
        target_hw: (H, W) to resize the binary mask into, if provided
        thresh: threshold in [0,1] after optional sigmoid

    Returns:
        Boolean mask of shape (H, W)
    """
    if _HAS_TORCH and torch.is_tensor(m):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)

    # Squeeze any singleton dims to reach HxW
    while m.ndim > 2:
        if m.shape[0] == 1:
            m = np.squeeze(m, 0)
        elif m.shape[-1] == 1:
            m = np.squeeze(m, -1)
        else:
            m = np.squeeze(m)

    m_min, m_max = float(np.nanmin(m)), float(np.nanmax(m))
    if np.isnan(m_min) or np.isnan(m_max):
        # Return an empty mask if values are NaN
        return np.zeros(target_hw or m.shape[:2], dtype=bool)

    # If values look like logits or are outside [0,1], squash to [0,1]
    if m_min < 0.0 or m_max > 1.0:
        m = stable_sigmoid(m)

    binmask = (m >= thresh)

    if target_hw is not None and binmask.shape[:2] != target_hw:
        binmask = cv2.resize(
            (binmask.astype(np.uint8) * 255),
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_NEAREST
        ) > 128

    return binmask


def mask_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute a tight [x1,y1,x2,y2] bbox from a (boolean or 0/1) mask.
    Returns None if the mask is empty.

    Args:
        mask: HxW boolean or numeric mask

    Returns:
        np.ndarray([x1, y1, x2, y2], dtype=float32) or None
    """
    mb = (mask > 0.5) if mask.dtype != bool else mask
    ys, xs = np.where(mb)
    if ys.size == 0 or xs.size == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ----------------------------- IoU computations ----------------------------- #

def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    IoU between two boxes in [x1,y1,x2,y2] format.
    Returns 0.0 if any box is invalid (non-positive area).
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_box_mask_iou(box_xyxy: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute IoU between a box and a mask.

    IoU = area(mask ∩ box) / area(mask ∪ box)

    Args:
        box_xyxy: [x1,y1,x2,y2]
        mask: HxW (bool or numeric)

    Returns:
        IoU in [0,1]
    """
    x0, y0, x1, y1 = map(int, box_xyxy)
    h, w = mask.shape[:2]
    x0 = max(0, min(x0, w)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h)); y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    mb = (mask > 0.5) if mask.dtype != bool else mask
    mask_in_box = mb[y0:y1, x0:x1]
    box_area = (x1 - x0) * (y1 - y0)
    mask_area = mb.sum()
    intersection = mask_in_box.sum()
    union = box_area + mask_area - intersection
    return 0.0 if union == 0 else float(intersection) / float(union)


# ---------------------- New/Existing assignment utilities -------------------- #

def filter_new_boxes(
    boxes_xyxy: np.ndarray,
    existing_masks: List[np.ndarray],
    iou_threshold: float = 0.3
) -> Tuple[np.ndarray, List[int]]:
    """
    Keep boxes that do NOT sufficiently overlap any existing mask.

    Args:
        boxes_xyxy: (N,4) array of [x1,y1,x2,y2]
        existing_masks: list of HxW masks (bool or 0/1)
        iou_threshold: box-vs-mask IoU to consider an already-tracked object

    Returns:
        new_boxes: (M,4) float32 array of boxes considered new
        idxs: indices of `boxes_xyxy` kept in `new_boxes`
    """
    new_boxes, idxs = [], []
    for i, b in enumerate(boxes_xyxy):
        if not any(compute_box_mask_iou(b, m) >= iou_threshold for m in existing_masks):
            new_boxes.append(b); idxs.append(i)
    if not new_boxes:
        return np.empty((0, 4), np.float32), []
    return np.array(new_boxes, np.float32), idxs


def assign_boxes_to_tracks(
    boxes_xyxy: np.ndarray,
    masks_by_obj: Dict[int, np.ndarray],
    iou_threshold: float = 0.3,
    target_hw: Tuple[int, int] | None = None,  # FIXED: Add resolution parameter
) -> Tuple[List[int], List[int]]:
    """
    For each detection box, either assign it to an EXISTING track (by IoU with that
    track's current mask on this frame), or mark it as NEW (return its index in
    `new_indices`).

    Args:
        boxes_xyxy: (N,4) detections on the current frame IN FULL RESOLUTION
        masks_by_obj: {obj_id: mask} mapping IN SAM2 DECIMATED RESOLUTION
        iou_threshold: min IoU between detection box and the track's mask
        target_hw: (H, W) = target resolution to resize masks to (CRITICAL!)

    Returns:
        assigned_track_ids: list of length N; for each detection i:
            - if matched to an existing track -> that obj_id (int)
            - if NEW (no match) -> -1
        new_indices: indices i in boxes_xyxy that were marked NEW (-1)
    """
    # FIXED: Resize masks to match detection frame resolution
    obj_mask: Dict[int, np.ndarray] = {}
    for oid, m in masks_by_obj.items():
        obj_mask[int(oid)] = mask_to_binary(m, target_hw=target_hw)

    assigned: List[int] = []
    new_idxs: List[int] = []

    for i, det in enumerate(boxes_xyxy):
        det = det.astype(np.float32)
        best_iou, best_oid = 0.0, None
        
        # Compare box against masks (now in same coordinate space!)
        for oid, mask in obj_mask.items():
            iou = compute_box_mask_iou(det, mask)
            if iou > best_iou:
                best_iou, best_oid = iou, oid
        
        if best_oid is not None and best_iou >= iou_threshold:
            assigned.append(int(best_oid))
        else:
            assigned.append(-1)
            new_idxs.append(i)

    return assigned, new_idxs
