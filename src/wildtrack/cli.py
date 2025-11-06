import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")  # avoid Triton/Inductor on Apple Silicon

import argparse
import tempfile
import shutil

from .devices import pick_device
from .utils.video import export_video_to_jpeg_folder
from .detectors.megadetector import MegaDetectorV5
from .segmenters.sam2_segmenter import SAM2Segmenter
from .pipeline.incremental import run_incremental


def main():
    ap = argparse.ArgumentParser(
        "wildtrack",
        description="WildTrack: wildlife detection + segmentation + tracking from video using MegaDetector and SAM2.",
    )

    # I/O
    ap.add_argument(
        "--video", required=True,
        help="Path to input video file (e.g., examples/djuma_zebras.mp4)."
    )
    ap.add_argument(
        "--out_dir", type=str, default="outputs",
        help="Directory to write results (masks.pkl, metadata.json, visualization). Default: ./outputs"
    )

    # Detection (MegaDetector)
    ap.add_argument(
        "--detector_conf", type=float, default=0.40,
        help="MegaDetector confidence threshold for animals. Default: 0.40"
    )
    ap.add_argument(
        "--detection_stride", type=int, default=10,
        help="Run MegaDetector every N ORIGINAL frames. Larger = faster, fewer prompts. Default: 10"
    )
    ap.add_argument(
        "--overlap_threshold", type=float, default=0.3,
        help="IoU threshold (box vs existing masks) to consider an animal as 'already tracked'. Default: 0.3"
    )

    # SAM2 / device / decimation
    ap.add_argument(
        "--device", type=str, default=pick_device("auto"),
        choices=["cpu", "mps", "cuda", "auto"],
        help="Compute device. 'auto' picks cuda > mps > cpu. Default: auto"
    )
    ap.add_argument(
        "--no_post", action="store_true",
        help="Disable SAM2 post-processing (use if you hit optional extension warnings)."
    )
    ap.add_argument(
        "--max_side", type=int, default=720,
        help="Resize frames so max(H,W)=this BEFORE SAM2/decimated export. Default: 720"
    )
    ap.add_argument(
        "--frame_stride", type=int, default=2,
        help="Decimation stride for SAM2 & visualization (every Nth frame becomes one JPEG). Larger = faster, but lower quality masks. Default: 2"
    )

    # Visualization
    ap.add_argument(
        "--debug", action="store_true",
        help="Enable visualization output (writes an annotated MP4 under outputs/<clip>/debug/)."
    )
    ap.add_argument(
        "--viz", dest="viz_mode", choices=["fast", "original"], default="fast",
        help="Visualization mode when --debug is set: "
             "'fast' overlays on decimated JPEGs (very fast), "
             "'original' overlays on original video (best quality, slower). Default: fast"
    )

    # Post-processing (merge duplicate tracks)
    ap.add_argument(
        "--no_merge", action="store_true",
        help="Disable the default duplicate-track merging (keep all tracks as-is)."
    )
    ap.add_argument(
        "--merge_iou", type=float, default=0.4,
        help="IoU threshold for merging duplicate tracks. Default: 0.4"
    )
    ap.add_argument(
        "--merge_min_frames", type=int, default=3,
        help="Minimum overlapping frames to compare when merging tracks. Default: 3"
    )

    args = ap.parse_args()
    device = pick_device(args.device)

    # 1) Export decimated JPEGs once (used by SAM2 & fast viz)
    tmp_dir = tempfile.mkdtemp(prefix="sam2_jpegs_")
    jpeg_folder, _ = export_video_to_jpeg_folder(
        args.video,
        tmp_dir,
        quality=90,
        max_side=args.max_side,
        frame_stride=args.frame_stride,
    )

    try:
        # 2) Wire detector + segmenter
        detector = MegaDetectorV5(conf_thresh=args.detector_conf, animals_only=True)
        segmenter = SAM2Segmenter(
            jpeg_folder=jpeg_folder,
            device=device,
            vos_optimized=True,
            apply_post=not args.no_post,
        )

        # 3) Run pipeline
        # Pass viz mode down via run_incremental (it will call visualize_fast with use_decimated accordingly)
        run_incremental(
            video_path=args.video,
            detector=detector,
            segmenter=segmenter,
            detection_stride=args.detection_stride,
            overlap_iou_threshold=args.overlap_threshold,
            frame_stride=args.frame_stride,
            max_side=args.max_side,
            debug_vis=args.debug,
            out_dir=args.out_dir,
            merge_duplicates=not args.no_merge,
            merge_iou_threshold=args.merge_iou,
            merge_min_frames=args.merge_min_frames,
            jpeg_folder=jpeg_folder,
            viz_mode=args.viz_mode,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
