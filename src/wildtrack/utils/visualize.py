import os
from pathlib import Path
import cv2
import numpy as np
from ..utils.video import get_video_meta

def visualize_result(
    video_path: str,
    sam2_outputs: list,
    frame_stride: int,
    out_path: str,
    alpha: float = 0.5,
    use_decimated: bool = True,
    jpeg_folder: str | None = None,
    max_side: int | None = None,
):
    """
    Render masks quickly.
    - use_decimated=True (+ jpeg_folder provided): read 000000.jpg, 000001.jpg, ... at SAM2 size -> fastest
    - use_decimated=False: read original video; optionally downscale frames once (max_side)
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Organize masks by decimated frame index
    by_frame = {d["frame_idx"]: d for d in sam2_outputs}

    # Deterministic color per object id
    def color_for(oid: int) -> tuple[int, int, int]:
        r = (oid * 97 + 53) % 196 + 60
        g = (oid * 57 + 23) % 196 + 60
        b = (oid * 31 + 89) % 196 + 60
        return int(b), int(g), int(r)
    
    # Simple ID label (species info is in metadata JSON)
    def label_for(oid: int) -> str:
        return f"ID{oid}"

    if use_decimated:
        assert jpeg_folder is not None, "jpeg_folder required with use_decimated=True"
        # Probe first JPEG to get size & fps
        first = os.path.join(jpeg_folder, "000000.jpg")
        img0 = cv2.imread(first)
        if img0 is None:
            print("⚠️  Could not read decimated frames.")
            return
        H, W = img0.shape[:2]

        # Keep original fps for smoother viewing
        fps, _, _ = get_video_meta(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        if not vw.isOpened():
            print(f"⚠️  Failed to create video writer for {out_path}"); return

        # Count total decimated frames from jpeg folder
        jpeg_files = sorted([f for f in os.listdir(jpeg_folder) if f.endswith('.jpg')])
        total_decimated_frames = len(jpeg_files)
        
        if total_decimated_frames == 0:
            print("⚠️  No JPEG frames found in decimated folder.")
            vw.release()
            return

        # Iterate through all decimated frames (not just those with detections)
        for dec_idx in range(total_decimated_frames):
            jpg_path = os.path.join(jpeg_folder, f"{dec_idx:06d}.jpg")
            frame = cv2.imread(jpg_path)
            if frame is None:
                print(f"⚠️  Could not read frame {jpg_path}")
                break

            rec = by_frame.get(dec_idx)
            if rec:
                obj_ids = rec["object_ids"]
                masks = rec["masks"]

                # Convert once to float32 for blending math
                frame_f = frame.astype(np.float32)

                for oid, m in zip(obj_ids, masks):
                    # m is already at decimated size
                    m = np.squeeze(np.asarray(m))
                    # logits -> probs -> bin
                    if m.min() < 0.0 or m.max() > 1.0:
                        # stable quick sigmoid
                        x = np.clip(m, -60.0, 60.0)
                        m = 0.5 * (1.0 + np.tanh(0.5 * x))
                    mask = (m >= 0.5)

                    if not np.any(mask):
                        continue

                    bgr = np.array(color_for(int(oid)), dtype=np.float32)
                    frame_f[mask] = frame_f[mask] * (1.0 - alpha) + bgr * alpha

                    ys, xs = np.where(mask)
                    if ys.size:
                        y0 = int(ys.min())
                        x0 = int(xs.min())
                        label = label_for(int(oid))
                        cv2.putText(frame_f, label, (x0, max(20, y0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

                frame = np.clip(frame_f, 0, 255).astype(np.uint8)
            # else: no detections for this frame, write plain frame

            vw.write(frame)

        vw.release()
        return

    # ---------- Original video path -----------
    fps, _, (W0, H0) = get_video_meta(video_path)
    if max_side is not None:
        s = max(H0, W0)
        scale = min(1.0, max_side / float(s))
        W = int(round(W0 * scale))
        H = int(round(H0 * scale))
    else:
        W, H = W0, H0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not vw.isOpened():
        print(f"⚠️  Failed to create video writer for {out_path}"); return

    cap = cv2.VideoCapture(video_path)
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (W, H) != (W0, H0):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

        dec_idx = fi // frame_stride
        rec = by_frame.get(dec_idx)
        if rec:
            frame_f = frame.astype(np.float32)
            obj_ids = rec["object_ids"]
            masks = rec["masks"]
            for oid, m in zip(obj_ids, masks):
                m = np.squeeze(np.asarray(m))
                if m.min() < 0.0 or m.max() > 1.0:
                    x = np.clip(m, -60.0, 60.0)
                    m = 0.5 * (1.0 + np.tanh(0.5 * x))
                mask = (m >= 0.5)
                if mask.shape[:2] != (H, W):
                    mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

                if not np.any(mask):
                    continue

                bgr = np.array(color_for(int(oid)), dtype=np.float32)
                frame_f[mask] = frame_f[mask] * (1.0 - alpha) + bgr * alpha

                ys, xs = np.where(mask)
                if ys.size:
                    y0 = int(ys.min())
                    x0 = int(xs.min())
                    label = label_for(int(oid))
                    cv2.putText(frame_f, label, (x0, max(20, y0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            frame = np.clip(frame_f, 0, 255).astype(np.uint8)
        # else: no detections for this frame, write plain frame

        vw.write(frame)
        fi += 1

    cap.release()
    vw.release()