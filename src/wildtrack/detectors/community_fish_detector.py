"""
Community Fish Detector for WildTrack.

A YOLO-based detector trained on 1.9M+ images from 17 datasets,
designed to detect fish in any environment (freshwater, marine, lab).

Source: https://github.com/WildHackers/community-fish-detector
"""

import os
import numpy as np
from pathlib import Path
from .base import Detector


class CommunityFishDetector(Detector):
    """
    Community Fish Detector - YOLO model for detecting fish in any environment.
    
    Trained on Community Fish Detection Dataset with 1.9M+ images spanning:
    - Freshwater environments
    - Marine/ocean environments  
    - Laboratory settings
    
    Args:
        conf_thresh: Confidence threshold for detections (default: 0.4)
        model_path: Path to .pt model file. If None, uses default cached model.
        imgsz: Input image size for YOLO (default: 1024, as per official recommendation)
        device: Device to run on ('auto', 'cpu', 'cuda', 'mps'). Auto selects best available.
    
    Example:
        >>> detector = CommunityFishDetector(conf_thresh=0.3, imgsz=1024)
        >>> boxes, scores, classes = detector.detect_bgr(frame_bgr)
    """
    
    DEFAULT_MODEL_URL = "https://github.com/WildHackers/community-fish-detector/releases/download/cfd-1.00-yolov12x/cfd-yolov12x-1.00.pt"
    DEFAULT_MODEL_NAME = "cfd-yolov12x-1.00.pt"
    
    def __init__(
        self, 
        conf_thresh: float = 0.4,
        model_path: str = None,
        imgsz: int = 1024,
        device: str = "auto",
        **kwargs  # Accept extra parameters (e.g., animals_only) and ignore them
    ):
        """
        Initialize the Community Fish Detector.
        
        Args:
            conf_thresh: Minimum confidence for detections (0-1)
            model_path: Path to .pt model file (auto-downloads if None)
            imgsz: YOLO input size (default: 1024)
            device: Compute device ('auto', 'cpu', 'cuda', 'mps')
            **kwargs: Extra parameters (ignored, for compatibility)
        """
        self.conf_thresh = float(conf_thresh)
        self.imgsz = int(imgsz)
        self.device = self._resolve_device(device)
        
        # Load model
        if model_path is None:
            model_path = self._get_default_model()
        
        self.model = self._load_model(model_path)
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _get_default_model(self) -> str:
        """
        Get path to default model, downloading if necessary.
        
        Returns:
            Path to cached model file
        """
        cache_dir = Path.home() / ".cache" / "wildtrack" / "community_fish_detector"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = cache_dir / self.DEFAULT_MODEL_NAME
        
        # Check if model exists
        if model_file.exists():
            print(f"Using cached Community Fish Detector model from {model_file}")
            return str(model_file)
        
        # Download model
        print(f"Downloading Community Fish Detector model to {model_file}")
        print("This is a one-time download (~238MB)...")
        
        try:
            import urllib.request
            urllib.request.urlretrieve(self.DEFAULT_MODEL_URL, str(model_file))
            print("✓ Download complete!")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Community Fish Detector model: {e}\n\n"
                f"You can manually download from:\n"
                f"{self.DEFAULT_MODEL_URL}\n\n"
                f"and place it at:\n"
                f"{model_file}"
            )
        
        return str(model_file)
    
    def _load_model(self, model_path: str):
        """
        Load YOLO model from .pt file.
        
        Args:
            model_path: Path to .pt model weights
            
        Returns:
            Loaded YOLO model
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Community Fish Detector requires 'ultralytics' package.\n"
                "Install with: pip install ultralytics"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Specify a valid model_path or let it auto-download."
            )
        
        print(f"Loading Community Fish Detector from {model_path}")
        model = YOLO(model_path)
        
        # Set device
        model.to(self.device)
        
        return model
    
    def detect_bgr(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect fish in a BGR image.
        
        Args:
            frame_bgr: BGR image (OpenCV format), shape (H, W, 3), dtype uint8
        
        Returns:
            boxes_xyxy: Bounding boxes in [x_min, y_min, x_max, y_max] format
                       Shape: (N, 4), dtype: float32, in absolute pixel coordinates
            scores: Confidence scores, shape (N,), dtype float32, range [0, 1]
            classes: Class IDs (all 1 for "fish"), shape (N,), dtype int32
        """
        # Handle invalid input
        if frame_bgr is None or frame_bgr.size == 0:
            return self._empty_outputs()
        
        # YOLO expects RGB
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Run inference
        # verbose=False suppresses YOLO's print statements
        results = self.model.predict(
            source=frame_rgb,
            imgsz=self.imgsz,
            conf=self.conf_thresh,
            verbose=False,
            device=self.device
        )
        
        # Extract first result (single image)
        result = results[0]
        
        # Check if any detections
        if len(result.boxes) == 0:
            return self._empty_outputs()
        
        # Extract boxes (already in xyxy format)
        boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        
        # Extract confidence scores
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        
        # All detections are class 1 (fish) for this detector
        classes = np.ones(len(boxes_xyxy), dtype=np.int32)
        
        return boxes_xyxy, scores, classes
    
    def _empty_outputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return empty arrays when no detections are found.
        
        Returns:
            Empty arrays in the correct format
        """
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32)
        )