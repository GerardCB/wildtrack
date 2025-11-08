import os
import sys
import json
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Literal, Tuple

import numpy as np
import cv2

from ..utils.video import get_video_meta, read_frame_at_index
from ..utils.masks import mask_to_binary, assign_boxes_to_tracks
from ..utils.visualize import visualize_result
from ..detectors.base import Detector
from ..segmenters.base import VideoSegmenter


# --------------------------- Track merge utilities --------------------------- #

def merge_duplicate_tracks(
    sam2_outputs: List[Dict[str, Any]],
    iou_threshold: float = 0.3,
    min_overlap_frames: int = 3,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    if not sam2_outputs:
        return sam2_outputs, {}

    all_ids = sorted(set(int(i) for d in sam2_outputs for i in d["object_ids"]))
    track = {i: {} for i in all_ids}
    for d in sam2_outputs:
        f = int(d["frame_idx"])
        for oid, m in zip(d["object_ids"], d["masks"]):
            oid = int(oid)
            track[oid][f] = mask_to_binary(m)

    def iou(a, b):
        if a.shape != b.shape:
            return 0.0
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return 0.0 if union == 0 else float(inter) / float(union)

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

    def resolve(k: int) -> int:
        seen = set()
        while k in merge_map and k not in seen:
            seen.add(k)
            k = merge_map[k]
        return k

    merged_outputs: List[Dict[str, Any]] = []
    for d in sam2_outputs:
        f = int(d["frame_idx"])
        obj_ids = [int(i) for i in d["object_ids"]]
        masks = d["masks"]

        per_id_masks: Dict[int, List[np.ndarray]] = {}
        for oid, m in zip(obj_ids, masks):
            fid = resolve(oid)
            per_id_masks.setdefault(fid, []).append(m)

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


# ------------------------------- Save artifacts ------------------------------ #

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


# ---------------------------- Species label helpers -------------------------- #

_NON_ANIMAL_TOKENS = {
    "blank", "empty", "unknown", "no animal", "none", "neg", "negative"
}

def _is_animal_label(label: str) -> bool:
    if not label:
        return False
    s = label.strip().lower()
    if s in _NON_ANIMAL_TOKENS:
        return False
    parts = [p.strip().lower() for p in s.split(";") if p.strip()]
    if parts:
        last = parts[-1]
        if last in _NON_ANIMAL_TOKENS:
            return False
    return True


# ----------------------------------- Main ------------------------------------ #

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
    Incremental detection + segmentation + tracking + classification pipeline.
    """
    clip_name = Path(video_path).stem
    out_dir = Path(out_dir) / clip_name
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    crops_dir = out_dir / "tmp_species_crops"
    if crops_dir.exists():
        for p in crops_dir.glob("*.jpg"):
            try: p.unlink()
            except Exception: pass
    else:
        crops_dir.mkdir(parents=True, exist_ok=True)

    crop_records: List[Tuple[str, int]] = []

    fps, total_frames, (W, H) = get_video_meta(video_path)
    print(f"Video: {total_frames} frames @ {fps:.1f}, {W}x{H}")

    accumulated: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    next_obj_id = 0
    detection_summary: List[Dict[str, Any]] = []
    track_species: Dict[int, Tuple[str, float]] = {}

    def _resize_for_speciesnet(crop_rgb: np.ndarray) -> np.ndarray:
        mv = getattr(classifier, "model_version", None) if classifier is not None else None
        if isinstance(mv, str) and mv.lower().startswith("v4.0.1a"):
            return cv2.resize(crop_rgb, (480, 480), interpolation=cv2.INTER_AREA)
        return crop_rgb

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

        masks_by_obj = accumulated.get(dec_idx, {})
        assigned_track_ids, new_indices = assign_boxes_to_tracks(
            boxes_xyxy, masks_by_obj, iou_threshold=overlap_iou_threshold
        )

        num_new = len(new_indices)
        num_existing = int(boxes_xyxy.shape[0]) - num_new
        print(f"  Existing animals: {num_existing}")
        print(f"  NEW animals: {num_new}")

        detidx_to_futureid: Dict[int, int] = {}
        for j, idx in enumerate(new_indices):
            detidx_to_futureid[idx] = next_obj_id + j

        # Save crops for ALL detections (existing + new)
        for det_i, box in enumerate(boxes_xyxy):
            if assigned_track_ids[det_i] != -1:
                tid = int(assigned_track_ids[det_i])
            else:
                tid = int(detidx_to_futureid[det_i])

            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_rgb = _resize_for_speciesnet(crop_rgb)

            fn = crops_dir / f"f{orig_idx:06d}_tid{tid}_det{det_i:03d}.jpg"
            cv2.imwrite(str(fn), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            crop_records.append((str(fn), tid))

        if num_new == 0:
            print("  No new animals to track; continue.\n")
            continue

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

    outputs: List[Dict[str, Any]] = []
    for f in sorted(accumulated.keys()):
        oid2mask = accumulated[f]
        oids = sorted(oid2mask.keys())
        outputs.append({
            "frame_idx": f,
            "object_ids": np.array(oids, np.int32),
            "masks": [oid2mask[oid] for oid in oids],
        })

    merge_map: Dict[int, int] = {}
    if merge_duplicates:
        outputs, merge_map = merge_duplicate_tracks(
            outputs,
            iou_threshold=merge_iou_threshold,
            min_overlap_frames=merge_min_frames,
            verbose=True
        )

    def _resolve(k: int) -> int:
        seen = set()
        while k in merge_map and k not in seen:
            seen.add(k)
            k = merge_map[k]
        return k

    # ---------------------- SINGLE BATCH SPECIES CLASSIFICATION ----------------------
    if classifier is not None and len(crop_records) > 0:
        print(f"\nRunning single-batch SpeciesNet on {len(crop_records)} crops...")

        used_classifier_helper = False
        mapping = None

        if hasattr(classifier, "classify_crops_folder"):
            try:
                mapping = classifier.classify_crops_folder(str(crops_dir))
                used_classifier_helper = True
            except Exception as e:
                print(f"  Warning: classifier.classify_crops_folder failed: {e}")
                mapping = None

        if mapping is None:
            # Use a TEMP FILE for predictions and delete it afterwards
            with tempfile.NamedTemporaryFile(prefix="speciesnet_preds_", suffix=".json", delete=False) as tf:
                preds_path = Path(tf.name)

            cmd = [
                sys.executable, "-m", "speciesnet.scripts.run_model",
                "--folders", str(crops_dir),
                "--predictions_json", str(preds_path),
            ]
            country = getattr(classifier, "country", None)
            admin1 = getattr(classifier, "admin1_region", None)
            if country:
                cmd += ["--country", str(country)]
            if admin1:
                cmd += ["--admin1_region", str(admin1)]

            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Parse then delete
                try:
                    with open(preds_path, "r") as f:
                        data = json.load(f)
                finally:
                    try:
                        preds_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                images = data.get("images") or data.get("predictions") or []
                mapping = {}
                for item in images:
                    fp = (
                        item.get("file")
                        or item.get("image_path")
                        or item.get("image")
                        or item.get("filepath")
                        or item.get("filename")
                    )
                    if fp and not os.path.isabs(fp):
                        fp = str((crops_dir / Path(fp).name).resolve())
                    species, score = None, None
                    if "prediction" in item and "prediction_score" in item:
                        species = item["prediction"]
                        score = float(item["prediction_score"])
                    if species is None and "classifications" in item:
                        cls = item["classifications"] or {}
                        classes = cls.get("classes") or []
                        scores  = cls.get("scores") or []
                        if classes and scores:
                            species, score = classes[0], float(scores[0])
                    if fp and (species is not None) and (score is not None):
                        mapping[Path(fp).resolve().as_posix()] = (species, score)
            except subprocess.CalledProcessError as e:
                print("  Warning: SpeciesNet CLI failed:")
                try:
                    if e.stdout: print(e.stdout.decode(errors="ignore"))
                    if e.stderr: print(e.stderr.decode(errors="ignore"))
                except Exception:
                    pass
                mapping = {}

        # Aggregate best per FINAL track id
        best_animal_per_final: Dict[int, Tuple[str, float]] = {}
        best_any_per_final: Dict[int, Tuple[str, float]] = {}

        for fp, provisional_tid in crop_records:
            final_tid = _resolve(int(provisional_tid))
            key = fp if used_classifier_helper else Path(fp).resolve().as_posix()
            sp_sc = mapping.get(key) if mapping else None
            if sp_sc is None:
                continue
            sp, sc = sp_sc
            sc = float(sc)

            prev_any = best_any_per_final.get(final_tid)
            if prev_any is None or sc > prev_any[1]:
                best_any_per_final[final_tid] = (sp, sc)

            if _is_animal_label(sp):
                prev_an = best_animal_per_final.get(final_tid)
                if prev_an is None or sc > prev_an[1]:
                    best_animal_per_final[final_tid] = (sp, sc)

        track_species = {}
        final_tids = set(list(best_any_per_final.keys()) + list(best_animal_per_final.keys()))
        for tid in sorted(final_tids):
            if tid in best_animal_per_final:
                sp, sc = best_animal_per_final[tid]
                if sc >= species_conf_threshold:
                    track_species[tid] = (sp, sc)
                else:
                    track_species[tid] = ("unknown", sc)
            else:
                sp, sc = best_any_per_final.get(tid, ("unknown", 0.0))
                track_species[tid] = ("unknown", sc)

        # Clean up crops directory now that we're done
        try:
            shutil.rmtree(crops_dir, ignore_errors=True)
        except Exception:
            pass

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
