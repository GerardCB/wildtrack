# 🐾 WildTrack

**WildTrack** is an open-source pipeline for wildlife detection, segmentation, and tracking in videos.
It combines [MegaDetector](https://github.com/agentmorris/MegaDetector) and [SAM 2](https://github.com/facebookresearch/sam2) into a unified incremental workflow that automatically finds, segments, and follows individual animals across frames — even with camera movement, occlusions, or multiple individuals.

---

## 🔗 Why WildTrack?

* **Incremental Detection & Tracking** — intelligently adds new animals only when they appear.
* **Modular Architecture** — easily swap detectors or segmenters.
* **Optimized Performance** — supports CUDA, MPS, and CPU out of the box.
* **Lightweight CLI** — process videos with a single command.

---

## 🚀 Quickstart

### Installation

#### Option 1 – For Users

Use WildTrack directly without downloading the source code:

```bash
pip install git+https://github.com/GerardCB/wildtrack.git
```

#### Option 2 – For Developers

If you plan to edit or contribute to the code:

```bash
# Clone repository
git clone https://github.com/GerardCB/wildtrack.git
cd wildtrack

# Install dependencies
pip install -e .
```

### Run on a video

```bash
wildtrack --video path/to/video_name.mp4 --debug
```

Outputs will be saved in `./outputs/<video_name>/`, including:

* **masks_preview.mp4** — video of detected animals with painted masks (only if using --debug)
* **<video_name>_metadata.json** — summary of detections, frames, merges
* **<video_name>_masks.pkl** — serialized SAM2 outputs and masks

---

## 🔧 CLI Options

| **Category**                   | **Argument**          | **Default**  | **Description**                                                                             |
| ------------------------------ | --------------------- | ------------ | ------------------------------------------------------------------------------------------- |
| **I/O**                        | `--video`             | *(required)* | Path to input video file (e.g. `examples/djuma_zebras.mp4`).                                |
|                                | `--out_dir`           | `outputs`    | Directory to write results (`masks.pkl`, `metadata.json`, `visualization.mp4`).             |
| **Detection (MegaDetector)**   | `--detector_conf`     | `0.40`       | Confidence threshold for animal detection.                                                  |
|                                | `--detection_stride`  | `10`         | Run MegaDetector every N original frames. Larger = faster, fewer detections.                |
|                                | `--overlap_threshold` | `0.3`        | IoU threshold (box vs existing masks) to consider an animal already tracked.                |
| **SAM2 / Device / Decimation** | `--device`            | `auto`       | Compute device: `cpu`, `mps`, `cuda`, or `auto` (prefers GPU if available).                 |
|                                | `--no_post`           | *(flag)*     | Disable SAM2 post-processing.                     |
|                                | `--max_side`          | `720`        | Resize frames so max(H,W)=this before SAM2 export.                                          |
|                                | `--frame_stride`      | `2`          | Frame decimation stride for SAM2 and visualization (every Nth frame → one JPEG).            |
| **Visualization**              | `--debug`             | *(flag)*     | Enable visualization output (annotated MP4 under `outputs/<clip>/debug/`).                  |
|                                | `--viz` | `fast`       | Visualization mode: `fast` overlays on decimated frames; `original` overlays on full video (better quality, slower). |
| **Post-processing (Merging)**  | `--no_merge`          | *(flag)*     | Disable duplicate-track merging (keep all tracks).                                          |
|                                | `--merge_iou`         | `0.4`        | IoU threshold for merging duplicate tracks.                                                 |
|                                | `--merge_min_frames`  | `3`          | Minimum overlapping frames required to compare for merging.                                 |

---

## 📂 Repository Structure

```
src/wildtrack/
├── detectors/        # MegaDetector wrapper
├── segmenters/       # SAM2 wrapper
├── pipeline/         # incremental detection logic & orchestration
├── utils/            # video, masks, visualization utilities
├── cli.py            # entrypoint for wildtrack command
└── devices.py        # device selection logic
```

---

## 🛍️ License

This project is licensed under the **MIT License**.

See also the [NOTICE](NOTICE) file for details on third-party components used.

---

## 🌳 Acknowledgments

WildTrack builds on the incredible work by:

* [Microsoft AI for Earth](https://github.com/microsoft/CameraTraps) — MegaDetector
* [Meta AI Research](https://github.com/facebookresearch/sam2) — Segment Anything 2

