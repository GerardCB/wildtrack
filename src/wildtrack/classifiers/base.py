"""
Base classifier interface for WildTrack.

All classifiers must implement the classify() method.
"""

from typing import Tuple
import numpy as np


class Classifier:
    """
    Interface for species classification.
    
    Classifiers take an image and bounding box and return a species prediction.
    """
    
    def classify(self, image_bgr: np.ndarray, bbox: np.ndarray) -> Tuple[str, float]:
        """
        Classify species within a bounding box.
        
        Args:
            image_bgr: BGR image (H, W, 3), dtype uint8
            bbox: Bounding box [x_min, y_min, x_max, y_max], dtype float32
        
        Returns:
            species: Species name (e.g., "plains_zebra", "african_elephant")
            confidence: Confidence score [0, 1]
        
        Example:
            >>> classifier = MyClassifier()
            >>> species, conf = classifier.classify(frame, [100, 200, 300, 400])
            >>> print(f"Detected {species} with {conf:.2%} confidence")
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement classify()"
        )
    
    def classify_batch(
        self, 
        image_bgr: np.ndarray, 
        bboxes: np.ndarray
    ) -> list[Tuple[str, float]]:
        """
        Classify multiple bounding boxes in the same image.
        
        Default implementation: loop over boxes and call classify().
        Override for batch efficiency if your model supports it.
        
        Args:
            image_bgr: BGR image (H, W, 3)
            bboxes: Array of boxes (N, 4) [x_min, y_min, x_max, y_max]
        
        Returns:
            List of (species, confidence) tuples, one per box
        """
        results = []
        for bbox in bboxes:
            species, conf = self.classify(image_bgr, bbox)
            results.append((species, conf))
        return results