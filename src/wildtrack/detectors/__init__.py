"""
Wildlife detection modules.

Usage:
    from wildtrack.detectors import get_detector, list_detectors
    
    # List available detectors
    list_detectors()
    
    # Create a detector
    detector = get_detector("megadetector-v5", conf_thresh=0.4)
"""

from .base import Detector
from .megadetector import MegaDetectorV5
from .registry import (
    get_detector,
    list_detectors,
    register_detector,
    get_available_detectors,
    DETECTOR_REGISTRY,
)

__all__ = [
    "Detector",
    "MegaDetectorV5",
    "get_detector",
    "list_detectors", 
    "register_detector",
    "get_available_detectors",
    "DETECTOR_REGISTRY",
]