# src/wildtrack/pipeline/incremental.py

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Literal, Tuple

import numpy as np

from ..utils.video import get_video_meta, read_frame_at_index
from ..utils.masks import mask_to_binary, assign_boxes_to_tracks
from ..utils.visualize import visualize_result
from ..detectors.base import Detector
from ..segmenters.base import VideoSegmenter


def merge_duplicate_tracks(
    sam2_outputs: List[Dict[str, Any]],
    iou_threshold: float = 0.3,
    min_overlap_frames: int = 3,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
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
    merge_map: Dict[int, int] = {}
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
    merged_outputs: List[Dict[str, Any]] = []
    for d in sam2_outputs:
        f = int(d["frame_idx"])
        obj_ids = [int(i) for i in d["object_ids"]]
        masks = d["masks"]

        # Group by final (kept) id
        per_id_masks: Dict[int, List[np.ndarray]] = {}  # final_id -> list of masks to combine
        for oid, m in zip(obj_ids, masks):
            fid = resolve(oid)
            per_id_masks.setdefault(fid, []).append(m)

        # Combine duplicates (choose ONE per id) via union
        new_ids: List[int] = []
        new_masks: List[np.ndarray] = []
        for fid, ms in per_id_masks.items():
            if len(ms) == 1:
                new_ids.append(fid)
                new_masks.append(ms[0])
            else:
                bin_union = None
                for _m in ms:
                    mm = np.squeeze(np.asarray(_m))
                    if mm.min() < 0.0 or mm.max() > 1.0:
                        x = np.clip(mm, -60.0, 60.0)
                        mm = 0.5 * (1.0 + np.tanh(0.5 * x))
                    mm_bin = (mm >= 0.5)
                    bin_union = mm_bin if bin_union is None else (bin_union | mm_bin)
                new_ids.append(fid)
                new_masks.append((bin_union.astype(np.uint8) if bin_union is not None else ms[0]))

        merged_outputs.append({
            "frame_idx": f,
            "object_ids": np.array(new_ids, dtype=np.int32),
            "masks": new_masks,
        })

    if verbose and merge_map:
        print(f"[merge_duplicate_tracks] Applied {len(merge_map)} merge(s): {merge_map}")

    return merged_outputs, merge_map


def save_results(
    video_path: str,
    outputs: List[Dict[str, Any]],
    detection_summary: List[Dict[str, Any]],
    merge_map: Dict[int, int],
    frame_stride: int,
    max_side: int,
    out_dir: str | Path,
    track_species: Dict[int, Tuple[str, float]] | None = None,
):
    clip = Path(video_path).stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps, frame_count, (W, H) = get_video_meta(video_path)

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
        "merge_map": {int(k): int(v) for k, v in merge_map.items()} if merge_map else {},
    }

    if track_species:
        tracks_info = []
        for track_id in uniq:
            track_info = {"track_id": int(track_id)}
            if track_id in track_species:
                species, conf = track_species[track_id]
                track_info["species"] = species
                track_info["species_confidence"] = float(conf)
            tracks_info.append(track_info)
        meta["tracks"] = tracks_info

    meta_path = out_dir / f"{clip}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return masks_path, meta_path


def run_incremental(
    video_path: str,
    detector: Detector,
    segmenter: VideoSegmenter,
    detection_stride: int = 10,
    overlap_iou_threshold: float = 0.3,
    frame_stride: int = 2,
    max_side: int = 720,
    debug_vis: bool = True,
    out_dir: str = "data/outputs",
    merge_duplicates: bool = True,
    merge_iou_threshold: float = 0.4,
    merge_min_frames: int = 3,
    jpeg_folder: str | None = None,
    viz_mode: Literal["fast", "original"] = "fast",
    classifier=None,
    species_conf_threshold: float = 0.5,
):
    """
    Run the incremental detection + classification + segmentation + tracking pipeline.
    """
    clip_name = Path(video_path).stem
    out_dir = Path(out_dir) / clip_name
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    fps, total_frames, (W, H) = get_video_meta(video_path)
    print(f"Video: {total_frames} frames @ {fps:.1f}, {W}x{H}")

    # Keep one mask per (frame, object)
    accumulated: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)  # frame_idx -> {obj_id: mask}
    next_obj_id = 0
    detection_summary: List[Dict[str, Any]] = []
    track_species: Dict[int, Tuple[str, float]] = {}  # track_id -> (species, confidence)

    safe_last = max(0, (total_frames - 1))
    detection_frames = list(range(0, safe_last, detection_stride))

    for k, orig_idx in enumerate(detection_frames, 1):
        print(f"[{k}/{len(detection_frames)}] Detection frame: {orig_idx}/{total_frames}")
        try:
            frame = read_frame_at_index(video_path, orig_idx)
        except RuntimeError as e:
            print(f"  ⚠️  Skipping unreadable frame {orig_idx}: {e}")
            continue

        boxes_xyxy, scores, classes = detector.detect_bgr(frame)
        if boxes_xyxy.shape[0] == 0:
            print("  No animals detected.\n")
            continue
        print(f"  {detector.__class__.__name__} found {boxes_xyxy.shape[0]} animal(s)")

        dec_idx = orig_idx // frame_stride

        # EXISTING masks (for this decimated frame) to link detections to current tracks
        masks_by_obj = accumulated.get(dec_idx, {})  # {obj_id: mask}
        assigned_track_ids, new_indices = assign_boxes_to_tracks(
            boxes_xyxy, masks_by_obj, iou_threshold=overlap_iou_threshold
        )

        num_new = len(new_indices)
        num_existing = int(boxes_xyxy.shape[0]) - num_new
        print(f"  Existing animals: {num_existing}")
        print(f"  NEW animals: {num_new}")

        # Map detection index -> future track_id for NEW ones, preserving order
        detidx_to_futureid: Dict[int, int] = {}
        for j, idx in enumerate(new_indices):
            detidx_to_futureid[idx] = next_obj_id + j

        # Classify ALL detections this frame (robust label over time)
        if classifier is not None and boxes_xyxy.shape[0] > 0:
            print(f"  Batch classifying {boxes_xyxy.shape[0]} detection(s) (tracks: {num_existing} existing, {num_new} new)...")
            try:
                if hasattr(classifier, "classify_batch"):
                    batch_results = classifier.classify_batch(frame, boxes_xyxy)
                else:
                    batch_results = [classifier.classify(frame, b) for b in boxes_xyxy]

                # Keep the BEST (highest-confidence) label per track across frames
                for det_i, (species, conf) in enumerate(batch_results):
                    # Determine target track id (existing or future)
                    if assigned_track_ids[det_i] != -1:
                        tid = int(assigned_track_ids[det_i])  # existing track
                    else:
                        tid = int(detidx_to_futureid[det_i])   # future id

                    prev = track_species.get(tid)
                    if prev is None or float(conf) > float(prev[1]):
                        track_species[tid] = (species, float(conf))

                # Pretty logging for NEW track candidates only (to reduce spam)
                for det_i in new_indices:
                    tid = detidx_to_futureid[det_i]
                    species, conf = batch_results[det_i]
                    if conf >= species_conf_threshold and species not in ("error", "unknown"):
                        print(f"    ID{tid}: {species} ({conf:.1%})")
                    else:
                        print(f"    ID{tid}: unknown (conf: {conf:.1%})")

            except Exception as e:
                print(f"  Warning: batch classification failed: {e}")
                # Do not overwrite any prior best labels on failure

        # If truly no new boxes, we can continue without propagating
        if num_new == 0:
            print("  No new animals to track; continue.\n")
            continue

        # Prepare ONLY the new boxes for SAM2 propagation, preserving order of new_indices
        new_boxes = boxes_xyxy[new_indices] if len(new_indices) > 0 else np.empty((0, 4), dtype=np.float32)

        scale = 1.0
        if max_side:
            s = max(H, W)
            if s > max_side:
                scale = max_side / float(s)
        boxes_scaled = new_boxes * scale

        print(f"  SAM2 propagate from decimated frame {dec_idx}...")
        sam2_out = segmenter.add_boxes_and_propagate(
            dec_idx,
            boxes_scaled,
            starting_obj_id=next_obj_id
        )
        print(f"  SAM2 produced masks for {len(sam2_out)} frames")

        # Overwrite mask per (frame, object)
        for d in sam2_out:
            f = d["frame_idx"]
            for m, oid in zip(d["masks"], d["object_ids"]):
                accumulated[f][int(oid)] = m

        next_obj_id += num_new
        detection_summary.append({
            "detection_frame": orig_idx,
            "decimated_frame": dec_idx,
            "total_detected": int(boxes_xyxy.shape[0]),
            "new_animals": int(num_new),
            "cumulative_objects": int(next_obj_id),
        })
        print(f"  Total unique animals tracked so far: {next_obj_id}\n")

    print("Converting to standard output format...")

    # Build outputs from dict (stable order)
    outputs: List[Dict[str, Any]] = []
    for f in sorted(accumulated.keys()):
        oid2mask = accumulated[f]
        oids = sorted(oid2mask.keys())
        outputs.append({
            "frame_idx": f,
            "object_ids": np.array(oids, np.int32),
            "masks": [oid2mask[oid] for oid in oids],
        })

    # Merge duplicate tracks if requested
    merge_map: Dict[int, int] = {}
    if merge_duplicates:
        outputs, merge_map = merge_duplicate_tracks(
            outputs,
            iou_threshold=merge_iou_threshold,
            min_overlap_frames=merge_min_frames,
            verbose=True
        )

    # Collapse species labels according to merge_map: keep HIGHEST confidence
    def _resolve(k: int) -> int:
        seen = set()
        while k in merge_map and k not in seen:
            seen.add(k)
            k = merge_map[k]
        return k

    if track_species:
        merged_species: Dict[int, Tuple[str, float]] = {}
        for tid, (sp, conf) in track_species.items():
            final_id = _resolve(int(tid))
            prev = merged_species.get(final_id)
            if prev is None or float(conf) > float(prev[1]):
                merged_species[final_id] = (sp, float(conf))
        track_species = merged_species

    # Persist artifacts
    masks_p, meta_p = save_results(
        video_path,
        outputs,
        detection_summary,
        merge_map,
        frame_stride,
        max_side,
        out_dir,
        track_species=track_species
    )

    # Optional visualization
    if debug_vis:
        vis_path = str((out_dir / "debug" / "masks_preview.mp4"))
        print("Creating visualization... (this may take a while)")
        visualize_result(
            video_path,
            outputs,
            frame_stride,
            vis_path,
            use_decimated=(viz_mode == "fast"),
            jpeg_folder=jpeg_folder if viz_mode == "fast" else None,
            track_species=track_species
        )
        print(f"Saved visualization: {vis_path}")

    print("Done.")
    return outputs, track_species
