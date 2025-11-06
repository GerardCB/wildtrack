import json, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, Any, List, Literal

from ..utils.video import get_video_meta, read_frame_at_index
from ..utils.masks import mask_to_binary, filter_new_boxes
from ..utils.visualize import visualize_result
from ..detectors.base import Detector
from ..segmenters.base import VideoSegmenter

def merge_duplicate_tracks(sam2_outputs, iou_threshold=0.3, min_overlap_frames=3, verbose=True):
    """
    Merge duplicate object tracks based on average IoU over overlapping frames.
    Returns:
      - merged sam2_outputs with at most one mask per (frame, object_id)
      - merge_map: {dropped_id -> kept_id}
    """
    if not sam2_outputs:
        return sam2_outputs, {}

    # Build per-id, per-frame binary masks for IoU computation
    all_ids = sorted(set(int(i) for d in sam2_outputs for i in d["object_ids"]))
    track = {i: {} for i in all_ids}
    for d in sam2_outputs:
        f = int(d["frame_idx"])
        for oid, m in zip(d["object_ids"], d["masks"]):
            oid = int(oid)
            track[oid][f] = mask_to_binary(m)  # binary HxW

    def iou(a, b):
        if a.shape != b.shape:
            return 0.0
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return 0.0 if union == 0 else float(inter) / float(union)

    # Decide merges using average IoU across overlapping frames
    merge_map = {}
    for i in range(len(all_ids)):
        for j in range(i + 1, len(all_ids)):
            a, b = all_ids[i], all_ids[j]
            if a in merge_map or b in merge_map:
                continue
            frames = sorted(set(track[a]) & set(track[b]))
            if len(frames) < min_overlap_frames:
                continue
            ious = [iou(track[a][f], track[b][f]) for f in frames]
            if np.mean(ious) >= iou_threshold:
                keep, drop = (a, b) if a < b else (b, a)
                merge_map[drop] = keep

    # Resolve function in case of chains
    def resolve(k: int) -> int:
        seen = set()
        while k in merge_map and k not in seen:
            seen.add(k)
            k = merge_map[k]
        return k

    # Remap IDs and collapse duplicates per frame
    merged_outputs = []
    for d in sam2_outputs:
        f = int(d["frame_idx"])
        obj_ids = [int(i) for i in d["object_ids"]]
        masks = d["masks"]

        # Group by final (kept) id
        per_id_masks = {}      # final_id -> list of masks to combine
        for oid, m in zip(obj_ids, masks):
            fid = resolve(oid)
            per_id_masks.setdefault(fid, []).append(m)

        # Combine duplicates (choose ONE per id)
        new_ids = []
        new_masks = []
        for fid, ms in per_id_masks.items():
            if len(ms) == 1:
                # keep as is
                new_ids.append(fid)
                new_masks.append(ms[0])
            else:
                # union/OR of binary masks
                bin_union = None
                for _m in ms:
                    mm = np.squeeze(np.asarray(_m))
                    if mm.min() < 0.0 or mm.max() > 1.0:
                        x = np.clip(mm, -60.0, 60.0)
                        mm = 0.5 * (1.0 + np.tanh(0.5 * x))
                    mm_bin = (mm >= 0.5)
                    bin_union = mm_bin if bin_union is None else (bin_union | mm_bin)
                new_ids.append(fid)
                new_masks.append(bin_union.astype(np.uint8))  # store compactly (uint8 0/1)

        merged_outputs.append({
            "frame_idx": f,
            "object_ids": np.array(new_ids, dtype=np.int32),
            "masks": new_masks,
        })

    if verbose and merge_map:
        print(f"[merge_duplicate_tracks] Applied {len(merge_map)} merge(s): {merge_map}")

    return merged_outputs, merge_map

def save_results(video_path, outputs, detection_summary, merge_map, frame_stride, max_side, out_dir):
    clip = Path(video_path).stem
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fps, frame_count, (W,H) = get_video_meta(video_path)
    masks_path = out_dir / f"{clip}_masks.pkl"
    with open(masks_path, "wb") as f:
        pickle.dump({
            "sam2_outputs": outputs,
            "clip_name": clip,
            "video_path": str(video_path),
            "merge_map": merge_map
        }, f)
    uniq = sorted(set(int(i) for d in outputs for i in d["object_ids"]))
    meta = {
        "clip_name": clip,
        "video_path": str(video_path),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "resolution": {"width": int(W), "height": int(H)},
        "frame_stride": int(frame_stride),
        "max_side": int(max_side) if max_side else None,
        "num_objects": len(uniq),
        "num_tracked_frames": len(outputs),
        "detection_summary": detection_summary,
        "merges_applied": len(merge_map),
        "merge_map": {int(k): int(v) for k,v in merge_map.items()} if merge_map else {},
    }
    meta_path = out_dir / f"{clip}_metadata.json"
    with open(meta_path, "w") as f: json.dump(meta, f, indent=2)
    return masks_path, meta_path

def run_incremental(video_path: str,
                    detector: Detector,
                    segmenter: VideoSegmenter,
                    detection_stride=10,
                    overlap_iou_threshold=0.3,
                    frame_stride=2,
                    max_side=720,
                    debug_vis=True,
                    out_dir="data/outputs",
                    merge_duplicates=True,
                    merge_iou_threshold=0.4,
                    merge_min_frames=3,
                    jpeg_folder: str | None = None,
                    viz_mode: Literal["fast","original"] = "fast"):
    """
    Run the incremental detection + segmentation + tracking pipeline.
    """
    clip_name = Path(video_path).stem
    out_dir = Path(out_dir)/clip_name
    debug_dir = out_dir/"debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    fps, total_frames, (W,H) = get_video_meta(video_path)
    print(f"Video: {total_frames} frames @ {fps:.1f}, {W}x{H}")

    # Keep one mask per (frame, object)
    accumulated: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)  # frame_idx -> {obj_id: mask}
    next_obj_id = 0
    detection_summary: List[Dict[str,Any]] = []

    detection_frames = list(range(0, total_frames, detection_stride))
    for k, orig_idx in enumerate(detection_frames, 1):
        print(f"[{k}/{len(detection_frames)}] Detection frame: {orig_idx}/{total_frames}")
        frame = read_frame_at_index(video_path, orig_idx)
        boxes_xyxy, scores, classes = detector.detect_bgr(frame)
        if boxes_xyxy.shape[0]==0:
            print("  No animals detected.\n")
            continue
        print(f"  MegaDetector found {boxes_xyxy.shape[0]} animal(s)")

        dec_idx = orig_idx // frame_stride

        existing = [mask_to_binary(m, target_hw=(H,W)) for m in accumulated.get(dec_idx, {}).values()]  # CHANGED
        new_boxes, _ = filter_new_boxes(boxes_xyxy, existing, overlap_iou_threshold)
        print(f"  Existing animals: {boxes_xyxy.shape[0]-new_boxes.shape[0]}")
        print(f"  NEW animals: {new_boxes.shape[0]}")

        if new_boxes.shape[0]==0:
            print("  All animals already have masks.\n")
            continue

        scale = 1.0
        if max_side:
            s = max(H,W)
            if s>max_side: scale = max_side/float(s)
        boxes_scaled = new_boxes * scale

        print(f"  SAM2 propagate from decimated frame {dec_idx}...")
        sam2_out = segmenter.add_boxes_and_propagate(dec_idx, boxes_scaled, starting_obj_id=next_obj_id)
        print(f"  SAM2 produced masks for {len(sam2_out)} frames")

        # Overwrite mask per (frame, object)
        for d in sam2_out:
            f = d["frame_idx"]
            for m, oid in zip(d["masks"], d["object_ids"]):
                accumulated[f][int(oid)] = m

        next_obj_id += new_boxes.shape[0]
        detection_summary.append({
            "detection_frame": orig_idx,
            "decimated_frame": dec_idx,
            "total_detected": int(boxes_xyxy.shape[0]),
            "new_animals": int(new_boxes.shape[0]),
            "cumulative_objects": int(next_obj_id),
        })
        print(f"  Total unique animals tracked so far: {next_obj_id}\n")

    print("Converting to standard output format...")

    # Build outputs from dict (stable order)
    outputs = []
    for f in sorted(accumulated.keys()):
        oid2mask = accumulated[f]
        oids = sorted(oid2mask.keys())
        outputs.append({
            "frame_idx": f,
            "object_ids": np.array(oids, np.int32),
            "masks": [oid2mask[oid] for oid in oids],
        })

    merge_map = {}
    if merge_duplicates:
        outputs, merge_map = merge_duplicate_tracks(
            outputs, 
            iou_threshold=merge_iou_threshold,
            min_overlap_frames=merge_min_frames, 
            verbose=True
        )
    masks_p, meta_p = save_results(
        video_path, 
        outputs,
        detection_summary, 
        merge_map,
        frame_stride, 
        max_side, 
        out_dir
    )

    if debug_vis:
        vis_path = str(debug_dir/"masks_preview.mp4")
        print("Creating visualization... (this may take a while)")
        visualize_result(
            video_path,
            outputs,
            frame_stride,
            vis_path,
            use_decimated = (viz_mode == "fast"),
            jpeg_folder = jpeg_folder if viz_mode == "fast" else None,
        )
        print(f"Saved visualization: {vis_path}")

    print("Done.")
    return outputs
