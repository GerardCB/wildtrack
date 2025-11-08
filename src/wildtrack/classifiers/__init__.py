"""
Species classification modules for WildTrack.

Usage:
    from wildtrack.classifiers import get_classifier, list_classifiers
    
    # List available classifiers
    list_classifiers()
    
    # Create a classifier
    classifier = get_classifier("megaclassifier", conf_thresh=0.6)
"""

from .base import Classifier
from .registry import (
    get_classifier,
    list_classifiers,
    register_classifier,
    get_available_classifiers,
    CLASSIFIER_REGISTRY,
)

__all__ = [
    "Classifier",
    "get_classifier",
    "list_classifiers",
    "register_classifier",
    "get_available_classifiers",
    "CLASSIFIER_REGISTRY",
]