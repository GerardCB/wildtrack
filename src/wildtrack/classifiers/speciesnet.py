"""
SpeciesNet classifier for WildTrack.

Google's SpeciesNet is a species classifier trained on 65M+ camera trap images,
covering 2000+ species. It's designed to work with MegaDetector.

Source: https://github.com/google/cameratrapai
Paper: https://doi.org/10.1049/cvi2.12318
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from .base import Classifier


class SpeciesNetClassifier(Classifier):
    """
    SpeciesNet species classifier from Google CameraTrap AI.
    
    Trained on 65M+ camera trap images with 2000+ labels including species,
    higher-level taxa (genus, family, order), and non-animal classes.
    
    Args:
        model_version: SpeciesNet model version (default: "v4.0.1a")
            - "v4.0.1a": Always-crop model (expects cropped bounding boxes)
            - "v4.0.1b": Full-image model (processes whole images)
        conf_thresh: Minimum confidence threshold (0-1)
        country: Optional ISO 3166-1 alpha-3 country code for geographic filtering (e.g., "USA", "GBR", "KEN")
        admin1_region: Optional state/region code (currently only USA, e.g., "CA", "NY")
        device: Device to run on ("auto", "cpu", "cuda")
        **kwargs: Extra parameters (for compatibility)
    
    Example:
        >>> classifier = SpeciesNetClassifier(conf_thresh=0.6, country="USA", admin1_region="CA")
        >>> species, conf = classifier.classify(frame_bgr, bbox)
        >>> print(f"Detected: {species} ({conf:.1%})")
    
    Notes:
        - Requires: pip install speciesnet
        - First run will download model weights (~500MB)
        - Model expects cropped images at 480x480 resolution
        - Geographic filtering helps improve accuracy
    """
    
    def __init__(
        self,
        model_version: str = "v4.0.1a",
        conf_thresh: float = 0.5,
        country: Optional[str] = None,
        admin1_region: Optional[str] = None,
        device: str = "auto",
        **kwargs
    ):
        """Initialize SpeciesNet classifier."""
        self.model_version = model_version
        self.conf_thresh = conf_thresh
        self.country = country
        self.admin1_region = admin1_region
        self.device = self._resolve_device(device)
        
        # Load SpeciesNet classifier
        self.classifier = self._load_classifier()
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string."""
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _load_classifier(self):
        """
        Load SpeciesNet classifier.
        
        Returns:
            SpeciesNetClassifier instance
        """
        try:
            from speciesnet.classifier import SpeciesNetClassifier as SNClassifier
        except ImportError:
            raise ImportError(
                "SpeciesNet not installed. Install with:\n"
                "  pip install speciesnet\n\n"
                "If on Mac and receive an error during this step, use:\n"
                "  pip install speciesnet --use-pep517\n\n"
                "More info: https://github.com/google/cameratrapai"
            )
        
        print(f"Loading SpeciesNet {self.model_version}...")
        print("(First run will download model weights ~500MB)")
        
        # Model specification
        model_path = f"kaggle:google/speciesnet/pyTorch/{self.model_version}"
        
        # Initialize classifier
        classifier = SNClassifier(
            model_path=model_path,
            device=self.device
        )
        
        print(f"✓ SpeciesNet loaded on {self.device.upper()}")
        return classifier
    
    def classify(self, image_bgr: np.ndarray, bbox: np.ndarray) -> Tuple[str, float]:
        """
        Classify species within a bounding box.
        
        Args:
            image_bgr: BGR image (H, W, 3), dtype uint8
            bbox: Bounding box [x_min, y_min, x_max, y_max], dtype float32
        
        Returns:
            species: Species name (e.g., "odocoileus_virginianus")
            confidence: Confidence score [0, 1]
        """
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validate bbox
        H, W = image_bgr.shape[:2]
        x1 = max(0, min(x1, W))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H))
        y2 = max(0, min(y2, H))
        
        # Check if bbox is valid
        if x2 <= x1 or y2 <= y1:
            return "error", 0.0
        
        # Crop to bounding box
        crop = image_bgr[y1:y2, x1:x2]
        
        # Check minimum size
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return "unknown", 0.0
        
        # SpeciesNet expects RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Prepare request format
        request = {
            "instances": [{
                "image": crop_rgb,  # RGB numpy array
            }]
        }
        
        # Add geographic information if provided
        if self.country:
            request["instances"][0]["country"] = self.country
        if self.admin1_region:
            request["instances"][0]["admin1_region"] = self.admin1_region
        
        try:
            # Run classifier (just classification, no detection)
            response = self.classifier.predict(request)
            
            # Parse response
            predictions = response.get("predictions", [])
            if not predictions:
                return "unknown", 0.0
            
            pred = predictions[0]
            
            # Check for failures
            if "failures" in pred and "CLASSIFIER" in pred["failures"]:
                return "error", 0.0
            
            # Get classifications
            classifications = pred.get("classifications", {})
            classes = classifications.get("classes", [])
            scores = classifications.get("scores", [])
            
            if not classes or not scores:
                return "unknown", 0.0
            
            # Get top prediction
            top_species = classes[0]
            top_confidence = scores[0]
            
            # Filter by confidence
            if top_confidence < self.conf_thresh:
                return "unknown", top_confidence
            
            # Clean up species name (remove underscores, capitalize)
            species_name = self._format_species_name(top_species)
            
            return species_name, top_confidence
            
        except Exception as e:
            print(f"  Warning: SpeciesNet classification failed: {e}")
            return "error", 0.0
    
    def _format_species_name(self, species: str) -> str:
        """
        Format species name for display.
        
        Args:
            species: Raw species name from SpeciesNet (e.g., "odocoileus_virginianus")
        
        Returns:
            Formatted name (e.g., "Odocoileus virginianus" or "White-tailed Deer")
        """
        # Handle special cases
        if species in ["blank", "unknown", "animal"]:
            return species
        
        # Replace underscores with spaces and capitalize
        formatted = species.replace("_", " ").title()
        
        return formatted
    
    def classify_batch(
        self, 
        image_bgr: np.ndarray, 
        bboxes: np.ndarray
    ) -> list[Tuple[str, float]]:
        """
        Classify multiple bounding boxes in the same image.
        
        Overridden for potential batch efficiency with SpeciesNet.
        Currently just loops (but could be optimized in future).
        
        Args:
            image_bgr: BGR image (H, W, 3)
            bboxes: Array of boxes (N, 4) [x_min, y_min, x_max, y_max]
        
        Returns:
            List of (species, confidence) tuples, one per box
        """
        # For now, just loop (could batch in future for efficiency)
        results = []
        for bbox in bboxes:
            species, conf = self.classify(image_bgr, bbox)
            results.append((species, conf))
        return results