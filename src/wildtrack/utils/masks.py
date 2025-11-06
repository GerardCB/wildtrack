import cv2, numpy as np
import torch

def stable_sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 0.5 * (1.0 + np.tanh(0.5 * x))

def mask_to_binary(m, target_hw=None, thresh=0.5):
    if torch.is_tensor(m): m = m.detach().cpu().numpy()
    m = np.asarray(m)
    while m.ndim > 2:
        if m.shape[0] == 1: m = np.squeeze(m, 0)
        elif m.shape[-1] == 1: m = np.squeeze(m, -1)
        else: m = np.squeeze(m)
    m_min, m_max = float(np.nanmin(m)), float(np.nanmax(m))
    if np.isnan(m_min) or np.isnan(m_max):
        return np.zeros(target_hw or m.shape[:2], dtype=bool)
    if m_min < 0.0 or m_max > 1.0:
        m = stable_sigmoid(m)
    binmask = m >= thresh
    if target_hw is not None and binmask.shape[:2] != target_hw:
        binmask = cv2.resize((binmask*255).astype(np.uint8),
                             (target_hw[1], target_hw[0]),
                             interpolation=cv2.INTER_NEAREST) > 128
    return binmask

def compute_box_mask_iou(box_xyxy, mask):
    x0, y0, x1, y1 = map(int, box_xyxy)
    h, w = mask.shape[:2]
    x0 = max(0, min(x0, w)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h)); y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0: return 0.0
    mb = (mask > 0.5) if mask.dtype != bool else mask
    mask_in_box = mb[y0:y1, x0:x1]
    box_area = (x1-x0)*(y1-y0)
    mask_area = mb.sum()
    intersection = mask_in_box.sum()
    union = box_area + mask_area - intersection
    return 0.0 if union == 0 else float(intersection)/float(union)

def filter_new_boxes(boxes_xyxy, existing_masks, iou_threshold=0.3):
    new_boxes, idxs = [], []
    for i, b in enumerate(boxes_xyxy):
        if not any(compute_box_mask_iou(b, m) >= iou_threshold for m in existing_masks):
            new_boxes.append(b); idxs.append(i)
    if not new_boxes:
        return np.empty((0,4), np.float32), []
    return np.array(new_boxes, np.float32), idxs
