import os
import cv2

def get_video_meta(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, frame_count, (w, h)


def read_frame_at_index(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return frame

def imwrite_jpeg(filename, frame, quality=95):
    try:
        return cv2.imwrite(filename, frame, params=[cv2.IMWRITE_JPEG_QUALITY, quality])
    except TypeError:
        return cv2.imwrite(filename, frame)

def export_video_to_jpeg_folder(video_path, out_dir, quality=90, max_side=None, frame_stride=2):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    write_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if (count % frame_stride) != 0:
            count += 1; continue
        if max_side is not None:
            h, w = frame.shape[:2]
            s = max(h, w)
            if s > max_side:
                scale = max_side / float(s)
                frame = cv2.resize(frame, (int(round(w*scale)), int(round(h*scale))), cv2.INTER_AREA)
        filename = os.path.join(out_dir, f"{write_idx:06d}.jpg")
        if not imwrite_jpeg(filename, frame, quality=quality):
            cap.release()
            raise RuntimeError(f"cv2.imwrite failed for {filename}")
        count += 1
        write_idx += 1
    cap.release()
    if write_idx == 0:
        raise RuntimeError("Failed to export JPEG frames; video unreadable?")
    return out_dir, write_idx
