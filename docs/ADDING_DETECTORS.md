# Adding a New Detector to WildTrack

WildTrack's modular architecture makes it easy to add new detection models. This guide shows you how to contribute a detector!

## Quick Start

1. **Create your detector file** in `src/wildtrack/detectors/`
2. **Implement the `Detector` interface** (just one method!)
3. **Register it** in `registry.py`
4. **Test it** with a sample video
5. **Submit a PR**!

## Real Example: Community Fish Detector

The best way to learn is by example. Check out `src/wildtrack/detectors/community_fish_detector.py` for a complete, working implementation.

## Step-by-Step Guide

### 1. Create Detector File

Create `src/wildtrack/detectors/your_detector.py`:

```python
"""
Your detector description and documentation.

Source: https://github.com/your/detector
"""

import numpy as np
from .base import Detector


class YourDetector(Detector):
    """
    Brief description of your detector.
    
    Args:
        conf_thresh: Confidence threshold for detections (0-1)
        # Add your detector-specific args here
        **kwargs: Extra parameters (for CLI compatibility)
    """
    
    def __init__(self, conf_thresh: float = 0.4, **kwargs):
        self.conf_thresh = conf_thresh
        # Initialize your model here
        self.model = self._load_model()
    
    def _load_model(self):
        """Load your detection model."""
        # Your model loading code
        # Consider caching to ~/.cache/wildtrack/your_detector/
        pass
    
    def detect_bgr(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects in a BGR image (OpenCV format).
        
        Args:
            frame_bgr: BGR image as numpy array (H, W, 3), dtype uint8
        
        Returns:
            boxes_xyxy: Bounding boxes [x_min, y_min, x_max, y_max], shape (N, 4), dtype float32
            scores: Confidence scores, shape (N,), dtype float32, range [0, 1]
            classes: Class IDs, shape (N,), dtype int32 (typically 1 for "animal")
        """
        # Handle invalid input
        if frame_bgr is None or frame_bgr.size == 0:
            return self._empty_outputs()
        
        # Your detection logic here:
        # 1. Run inference
        # 2. Filter by confidence
        # 3. Convert to required format
        # 4. Return boxes, scores, classes
        
        # Example empty return (when no detections):
        if no_detections:
            return self._empty_outputs()
        
        return boxes_xyxy, scores, classes
    
    def _empty_outputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standard empty output format."""
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32)
        )
```

### 2. Register Your Detector

Add to `src/wildtrack/detectors/registry.py`:

```python
from .your_detector import YourDetector

DETECTOR_REGISTRY = {
    # ... existing detectors ...
    
    "your-detector": {
        "class": YourDetector,
        "description": "Brief one-line description",
        "source": "https://github.com/your/detector",
        "categories": {
            1: "animal",  # or "fish", "bird", etc.
        }
    },
}
```

### 3. Update Imports

Add to `src/wildtrack/detectors/__init__.py`:

```python
from .your_detector import YourDetector

__all__ = [
    # ... existing exports ...
    "YourDetector",
]
```

### 4. Test Your Detector

```bash
# List detectors (should show yours)
wildtrack --list-detectors

# Test on a video
wildtrack -v test_video.mp4 --detector your-detector --visualize fast
```

## Critical Requirements

### Output Format

**Your `detect_bgr()` method MUST return exactly this:**

```python
boxes_xyxy, scores, classes = detector.detect_bgr(frame_bgr)
```

Where:

1. **`boxes_xyxy`**: `np.ndarray` of shape `(N, 4)`, dtype `float32`
   - Format: `[x_min, y_min, x_max, y_max]` in **absolute pixel coordinates**
   - NOT normalized (0-1), NOT xywh format

2. **`scores`**: `np.ndarray` of shape `(N,)`, dtype `float32`
   - Range: [0, 1]
   - Confidence/probability for each detection

3. **`classes`**: `np.ndarray` of shape `(N,)`, dtype `int32`
   - Class IDs (typically `1` for "animal")

### Handling Edge Cases

```python
def detect_bgr(self, frame_bgr):
    # 1. Handle None/empty input
    if frame_bgr is None or frame_bgr.size == 0:
        return self._empty_outputs()
    
    # 2. Run your inference
    detections = self.model.predict(frame_bgr)
    
    # 3. Parse results
    boxes, scores, classes = self._parse_detections(detections)
    
    # 4. Filter by confidence
    keep = scores >= self.conf_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    
    # 5. Handle empty results after filtering
    if len(boxes) == 0:
        return self._empty_outputs()
    
    return boxes, scores, classes
```

## Common Patterns

### Pattern 1: Wrapping a YOLO Model

```python
from ultralytics import YOLO
import cv2

class YOLODetector(Detector):
    def __init__(self, conf_thresh: float = 0.4, model_path: str = "yolov8n.pt", **kwargs):
        self.conf_thresh = conf_thresh
        self.model = YOLO(model_path)
    
    def detect_bgr(self, frame_bgr: np.ndarray):
        if frame_bgr is None or frame_bgr.size == 0:
            return self._empty_outputs()
        
        # YOLO inference
        results = self.model(frame_bgr, conf=self.conf_thresh, verbose=False)[0]
        
        if len(results.boxes) == 0:
            return self._empty_outputs()
        
        # Extract (already in xyxy format)
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = results.boxes.conf.cpu().numpy().astype(np.float32)
        classes = np.ones(len(boxes), dtype=np.int32)  # All class 1
        
        return boxes, scores, classes
```

### Pattern 2: Auto-Downloading Models

```python
from pathlib import Path
import urllib.request

class AutoDownloadDetector(Detector):
    MODEL_URL = "https://github.com/user/repo/releases/download/v1.0/model.pt"
    
    def _get_model_path(self) -> str:
        """Download model to cache if needed."""
        cache_dir = Path.home() / ".cache" / "wildtrack" / "your_detector"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = cache_dir / "model.pt"
        
        if model_file.exists():
            return str(model_file)
        
        print(f"Downloading model to {model_file} (one-time, ~100MB)...")
        try:
            urllib.request.urlretrieve(self.MODEL_URL, str(model_file))
            print("✓ Download complete!")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
        
        return str(model_file)
```

### Pattern 3: Converting Box Formats

```python
def _convert_boxes_to_xyxy(self, boxes_xywh: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Convert normalized xywh boxes to absolute xyxy.
    
    Args:
        boxes_xywh: [x_center, y_center, width, height] normalized (0-1)
        W, H: Image dimensions
    
    Returns:
        boxes_xyxy: [x_min, y_min, x_max, y_max] in pixels
    """
    x_center = boxes_xywh[:, 0] * W
    y_center = boxes_xywh[:, 1] * H
    width = boxes_xywh[:, 2] * W
    height = boxes_xywh[:, 3] * H
    
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    
    return np.stack([x_min, y_min, x_max, y_max], axis=1).astype(np.float32)
```

## Best Practices

### Model Loading

1. **Cache models** to `~/.cache/wildtrack/detector_name/`
2. **Show progress** for downloads
3. **Handle failures gracefully** with clear error messages
4. **Support custom paths** via constructor argument

### Performance

1. **Minimize imports** - Import heavy libraries inside methods if possible
2. **Lazy loading** - Only load model when first used
3. **Device selection** - Support CPU/CUDA/MPS where possible

### Documentation

1. **Docstrings** - Explain what your detector does and when to use it
2. **Source links** - Link to papers, repos, datasets
3. **Examples** - Show usage in docstring

## Testing

### Manual Testing

```bash
# Basic test
wildtrack -v test.mp4 --detector your-detector --visualize fast

# Test with different parameters
wildtrack -v test.mp4 --detector your-detector --confidence 0.2

# Compare with MegaDetector
wildtrack -v test.mp4 --detector megadetector-v5 -o results_mega/
wildtrack -v test.mp4 --detector your-detector -o results_yours/
```

## Real-World Example: Community Fish Detector

See `src/wildtrack/detectors/community_fish_detector.py` for a complete implementation that shows:

- ✅ Auto-downloading model from GitHub releases
- ✅ Caching to `~/.cache/wildtrack/`
- ✅ Wrapping Ultralytics YOLO
- ✅ Proper error handling
- ✅ Device selection (CUDA/MPS/CPU)
- ✅ Complete documentation
- ✅ Accepting extra kwargs for compatibility

Key features:
```python
class CommunityFishDetector(Detector):
    DEFAULT_MODEL_URL = "https://github.com/..."
    
    def __init__(self, conf_thresh=0.4, model_path=None, imgsz=1024, device="auto", **kwargs):
        # Auto-download if model_path is None
        # Support custom device selection
        # Accept **kwargs for CLI compatibility
    
    def _get_default_model(self):
        # Download to cache with progress
        # Reuse if already downloaded
    
    def detect_bgr(self, frame_bgr):
        # Convert BGR to RGB for YOLO
        # Run inference with confidence threshold
        # Return in correct format
```

## Submitting Your Detector

1. **Test thoroughly** on diverse videos
2. **Document use cases** in the detector docstring
3. **Update CHANGELOG.md**
4. **Create a PR** with:
   - Detector implementation
   - Registry update
   - Example usage
   - Performance notes (optional)

### PR Template

```markdown
## Add [Detector Name]

**Description:**
[Brief description of what this detector does]

**Use cases:**
- [Use case 1]
- [Use case 2]

**Testing:**
- [x] Tested on [dataset/video type]
- [x] Output format validated
- [x] Works with all CLI flags
- [x] Documentation added

```

## Questions?

- Check `community_fish_detector.py` for a real example
- Open an issue on GitHub
- Tag @GerardCB in your PR for review

Thank you for contributing to WildTrack! 🐾