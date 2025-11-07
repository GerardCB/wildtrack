"""
Wildlife detection modules.

Usage:
    from wildtrack.detectors import get_detector, list_detectors
    
    # List available detectors
    list_detectors()
    
    # Create a detector
    detector = get_detector("megadetector-v5", conf_thresh=0.4)
    detector = get_detector("community-fish", conf_thresh=0.3)
"""

from .base import Detector
from .megadetector import MegaDetectorV5
from .community_fish_detector import CommunityFishDetector
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
    "CommunityFishDetector",
    "get_detector",
    "list_detectors", 
    "register_detector",
    "get_available_detectors",
    "DETECTOR_REGISTRY",
]