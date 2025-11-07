import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

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
        description="WildTrack: wildlife detection, segmentation, and tracking using MegaDetector and SAM2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ==================== INPUT/OUTPUT ====================
    io_group = ap.add_argument_group('Input/Output')
    io_group.add_argument(
        "--video", "-v", required=True,
        help="Path to input video file (e.g., examples/elephants.mp4)"
    )
    io_group.add_argument(
        "--output-dir", "-o", type=str, default="outputs",
        help="Directory for output files (default: ./outputs)"
    )

    # ==================== DETECTION ====================
    detect_group = ap.add_argument_group('Detection Settings')
    detect_group.add_argument(
        "--confidence", "-c", type=float, default=0.40,
        help="Minimum confidence threshold for animal detection (default: 0.40)"
    )
    detect_group.add_argument(
        "--detect-every", type=int, default=10,
        help="Run detector every N frames (larger = faster but may miss animals, default: 10)"
    )
    detect_group.add_argument(
        "--overlap-threshold", type=float, default=0.3,
        help="IoU threshold to consider animal already tracked (default: 0.3)"
    )

    # ==================== PROCESSING ====================
    process_group = ap.add_argument_group('Processing Settings')
    process_group.add_argument(
        "--device", type=str, default="auto",
        choices=["cpu", "mps", "cuda", "auto"],
        help="Compute device (default: auto - picks best available)"
    )
    process_group.add_argument(
        "--max-resolution", type=int, default=720,
        help="Resize frames so max dimension = N pixels before processing (default: 720)"
    )
    process_group.add_argument(
        "--sample-every", type=int, default=2,
        help="Process every Nth frame for SAM2 segmentation (default: 2)"
    )
    process_group.add_argument(
        "--skip-postprocessing", action="store_true",
        help="Skip SAM2 post-processing step (use if encountering extension errors)"
    )

    # ==================== OUTPUT/VISUALIZATION ====================
    output_group = ap.add_argument_group('Output & Visualization')
    output_group.add_argument(
        "--visualize", type=str, default="none",
        choices=["none", "fast", "high-quality"],
        help="Generate annotated video: 'none' (no video), 'fast' (quick low-res preview), "
             "'high-quality' (overlay on original frames, slower) (default: none)"
    )

    # ==================== POST-PROCESSING ====================
    merge_group = ap.add_argument_group('Track Merging')
    merge_group.add_argument(
        "--skip-merge", action="store_true",
        help="Skip automatic merging of duplicate tracks"
    )
    merge_group.add_argument(
        "--merge-iou", type=float, default=0.4,
        help="IoU threshold for merging duplicate tracks (default: 0.4)"
    )
    merge_group.add_argument(
        "--merge-min-overlap", type=int, default=3,
        help="Minimum overlapping frames needed to merge tracks (default: 3)"
    )

    args = ap.parse_args()
    device = pick_device(args.device)

    # Determine if visualization is needed
    debug_vis = args.visualize != "none"
    viz_mode = "original" if args.visualize == "high-quality" else "fast"

    # 1) Export decimated JPEGs
    tmp_dir = tempfile.mkdtemp(prefix="sam2_jpegs_")
    jpeg_folder, _ = export_video_to_jpeg_folder(
        args.video,
        tmp_dir,
        quality=90,
        max_side=args.max_resolution,
        frame_stride=args.sample_every,
    )

    try:
        # 2) Initialize detector and segmenter
        detector = MegaDetectorV5(conf_thresh=args.confidence, animals_only=True)
        segmenter = SAM2Segmenter(
            jpeg_folder=jpeg_folder,
            device=device,
            vos_optimized=True,
            apply_post=not args.skip_postprocessing,
        )

        # 3) Run pipeline
        run_incremental(
            video_path=args.video,
            detector=detector,
            segmenter=segmenter,
            detection_stride=args.detect_every,
            overlap_iou_threshold=args.overlap_threshold,
            frame_stride=args.sample_every,
            max_side=args.max_resolution,
            debug_vis=debug_vis,
            out_dir=args.output_dir,
            merge_duplicates=not args.skip_merge,
            merge_iou_threshold=args.merge_iou,
            merge_min_frames=args.merge_min_overlap,
            jpeg_folder=jpeg_folder,
            viz_mode=viz_mode,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()