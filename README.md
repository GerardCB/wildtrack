# 🐾 WildTrack

![Version](https://img.shields.io/badge/version-0.4.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**WildTrack** is an open-source pipeline for wildlife detection, segmentation, tracking, and species classification in videos.
It combines your choice of wildlife detection models with [SAM 2](https://github.com/facebookresearch/sam2) and [SpeciesNet](https://github.com/google/cameratrapai) classification into a unified incremental workflow that automatically finds, segments, follows, and identifies individual animals across frames — even with camera movement, occlusions, or multiple individuals.

<p align="center">
  <img src="demo.gif" alt="WildTrack tracking elephants" width="720"/><br>
  <em>Example: WildTrack automatically detects, segments, tracks, and identifies elephants across video frames!</em>
</p>

---

## 🔗 Why WildTrack?

* **Flexible Detection** — choose from multiple models (MegaDetector, Community Fish Detector, or bring your own)
* **Incremental Tracking** — intelligently adds new animals only when they appear
* **High-Quality Segmentation** — SAM2-powered masks for precise individual tracking
* **Species Classification** — optional SpeciesNet integration for automatic species ID
* **Hardware Optimized** — supports CUDA, MPS, and CPU out of the box
* **Simple CLI** — process videos with a single command

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
# Basic tracking
wildtrack --video path/to/video_name.mp4 --visualize high-quality

# With species classification
wildtrack --video path/to/video_name.mp4 --classify --visualize high-quality
```

Outputs will be saved in `./outputs/<video_name>/`, including:

* **masks_preview.mp4** — video of detected animals with painted masks
* **<video_name>_metadata.json** — summary of detections, frames, merges, and per-track species
* **<video_name>_masks.pkl** — serialized SAM2 outputs and masks

---

## 🔍 Detector Support

WildTrack supports multiple detection models. Choose the best detector for your use case:

```bash
# List available detectors
wildtrack --list-detectors

# Use specific detector
wildtrack -v video.mp4 --detector megadetector-v5 --visualize fast
```

### Available Detectors

| Detector | Best For | Notes |
|----------|----------|-------|
| `megadetector-v5` | Terrestrial wildlife, camera traps | Default detector |
| `community-fish` | Underwater footage, marine life | Requires [ultralytics](https://pypi.org/project/ultralytics/) |

### Examples

```bash
# Terrestrial wildlife (MegaDetector as default)
wildtrack -v safari.mp4 --visualize fast

# Underwater/marine life
wildtrack -v reef.mp4 --detector community-fish --visualize fast
```

Want to add your own detector? See our [detector contribution guide](docs/ADDING_DETECTORS.md)!

---

## 🏷️ Species Classification

WildTrack includes optional species classification powered by Google's [SpeciesNet](https://github.com/google/cameratrapai), trained on 65M+ camera trap images with 2000+ species coverage.

### Installation

Species classification requires the SpeciesNet package:

```bash
# Install SpeciesNet
pip install speciesnet

# On Mac, if you encounter errors:
pip install speciesnet --use-pep517
```

### Basic Usage

```bash
# List available classifiers
wildtrack --list-classifiers

# Enable classification
wildtrack -v safari.mp4 --classify --visualize fast

# With custom confidence threshold
wildtrack -v video.mp4 --classify --species-confidence 0.7 --visualize fast
```

### Geographic Filtering

SpeciesNet supports geographic filtering to improve accuracy by filtering predictions to species found in specific regions:

```bash
# Country-level filtering
wildtrack -v safari.mp4 --classify --country KEN --visualize fast

# USA with state-level filtering
wildtrack -v yellowstone.mp4 --classify --country USA --admin1-region WY --visualize fast
```

**Supported codes:**
- **Country:** ISO 3166-1 alpha-3 codes
- **State/Region:** USA only, use 2-letter codes

See [SpeciesNet](https://github.com/google/cameratrapai) for more details on geographic filtering.

---

## 🔧 CLI Options

| **Category** | **Argument** | **Default** | **Description** |
|--------------|--------------|-------------|-----------------|
| **Input/Output** | `-v`, `--video` | *(required)* | Path to input video file |
| | `-o`, `--output-dir` | `outputs` | Directory for output files |
| **Detection** | `--detector` | `megadetector-v5` | Detection model to use (run `--list-detectors` for options) |
| | `-c`, `--confidence` | `0.40` | Minimum confidence threshold for detections |
| | `--detect-every` | `10` | Run detector every N frames (larger = faster) |
| | `--overlap-threshold` | `0.3` | IoU threshold to consider animal already tracked |
| **Classification** | `--classify` | *(flag)* | Enable species classification |
| | `--classifier` | `speciesnet` | Classifier to use (run `--list-classifiers` for options) |
| | `--species-confidence` | `0.5` | Minimum confidence for species predictions |
| | `--country` | *(optional)* | ISO 3166-1 alpha-3 country code |
| | `--admin1-region` | *(optional)* | State/region 2 letter code (USA only) |
| **Processing** | `--device` | `auto` | Compute device: `cpu`, `mps`, `cuda`, or `auto` |
| | `--max-resolution` | `720` | Resize frames so max dimension = N pixels (smaller = faster) |
| | `--sample-every` | `2` | Process every Nth frame for segmentation (larger = faster)|
| | `--skip-postprocessing` | *(flag)* | Skip SAM2 post-processing step |
| **Visualization** | `--visualize` | `none` | Output mode: `none`, `fast` (quick preview), or `high-quality` (overlay on original) |
| **Track Merging** | `--skip-merge` | *(flag)* | Disable automatic merging of duplicate tracks |
| | `--merge-iou` | `0.4` | IoU threshold for merging duplicate tracks |
| | `--merge-min-overlap` | `3` | Minimum overlapping frames to merge tracks |

### Common Usage Examples

```bash
# Basic usage with quick, low-resolution video preview
wildtrack -v elephants.mp4 --visualize fast

# With species classification
wildtrack -v safari.mp4 --classify --country KEN --visualize fast

# Underwater fish detection and classification
wildtrack -v reef_dive.mp4 --detector community-fish --classify --visualize fast

# High-quality video output with classification in custom directory
wildtrack -v elephants.mp4 --classify --country KEN -o results/ --visualize high-quality

# Fast processing (no visualization)
wildtrack -v video.mp4 --classify --detect-every 20 --sample-every 4

# Sensitive detection for hard-to-spot animals
wildtrack -v nocturnal.mp4 --confidence 0.25 --classify --species-confidence 0.3

# Full control over processing with classification
wildtrack -v test.mp4 \
  --detector megadetector-v5 \
  --confidence 0.35 \
  --detect-every 20 \
  --sample-every 3 \
  --classify \
  --classifier speciesnet \
  --species-confidence 0.6 \
  --country USA \
  --admin1-region CA \
  --visualize high-quality \
  --skip-merge
```

### Understanding Key Parameters

**Detection vs Sampling:**
- `--detect-every N`: How often to look for *new* animals (affects detection only)
- `--sample-every N`: How often to process frames for *tracking* (affects segmentation quality)

**Visualization Modes:**
- `none`: No video output (fastest, for batch processing)
- `fast`: Quick preview using processed frames (~2-5x faster)
- `high-quality`: Overlay masks on original video (best quality, slower)

**Device Selection:**
- `auto`: Automatically picks CUDA → MPS → CPU
- `cuda`: Force NVIDIA GPU (fastest)
- `mps`: Force Apple Silicon GPU
- `cpu`: Force CPU (slowest, but works everywhere)

**Classification:**
- Enable with `--classify` flag
- Add `--country` for geographic filtering
- Lower `--species-confidence` to 0.3-0.4 for harder-to-identify animals

---

## 📂 Repository Structure

```
src/wildtrack/
├── classifiers/      # Species classification (SpeciesNet, etc.)
├── detectors/        # Detection models (MegaDetector, Community Fish, etc.)
├── segmenters/       # SAM2 wrapper
├── pipeline/         # Incremental detection, tracking, classification logic
├── utils/            # Video, masks, visualization utilities
├── cli.py            # Entrypoint for wildtrack command
└── devices.py        # Device selection logic
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
* [WildHackers Community](https://github.com/WildHackers/community-fish-detector) — Community Fish Detector
* [Google CameraTrap AI](https://github.com/google/cameratrapai) — SpeciesNet species classifier
