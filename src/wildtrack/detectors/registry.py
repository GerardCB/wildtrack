"""
Detector registry for WildTrack.

Makes it easy to:
- Add new detectors
- List available detectors  
- Instantiate detectors by name
"""

from typing import Dict, Type, Any
from .base import Detector
from .megadetector import MegaDetectorV5


# Registry structure: name -> {class, description, metadata}
DETECTOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "megadetector-v5": {
        "class": MegaDetectorV5,
        "description": "MegaDetector v5 - General wildlife detector",
        "source": "https://github.com/agentmorris/MegaDetector",
        "categories": {
            1: "animal",
            2: "person", 
            3: "vehicle"
        }
    },
    # More detectors will be added here
    # "community-fish": {...},
    # "yolov8-wildlife": {...},
}


def register_detector(
    name: str,
    detector_class: Type[Detector],
    description: str,
    source: str = "",
    **metadata
):
    """
    Register a new detector.
    
    Args:
        name: Unique identifier (e.g., "my-detector")
        detector_class: Class that implements Detector interface
        description: Short description of the detector
        source: URL to paper or GitHub repo
        **metadata: Additional metadata (categories, etc.)
    
    Example:
        register_detector(
            name="community-fish",
            detector_class=CommunityFishDetector,
            description="Specialized underwater fish detector",
            source="https://github.com/WildHackers/community-fish-detector"
        )
    """
    if name in DETECTOR_REGISTRY:
        raise ValueError(f"Detector '{name}' already registered")
    
    if not issubclass(detector_class, Detector):
        raise TypeError(f"detector_class must inherit from Detector")
    
    DETECTOR_REGISTRY[name] = {
        "class": detector_class,
        "description": description,
        "source": source,
        **metadata
    }


def get_detector(name: str, **kwargs) -> Detector:
    """
    Instantiate a detector by name.
    
    Args:
        name: Detector name from registry
        **kwargs: Arguments passed to detector constructor
                 (e.g., conf_thresh=0.3, animals_only=True)
    
    Returns:
        Initialized Detector instance
    
    Example:
        detector = get_detector("megadetector-v5", conf_thresh=0.4)
    """
    if name not in DETECTOR_REGISTRY:
        available = ", ".join(DETECTOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown detector '{name}'.\n"
            f"Available detectors: {available}\n"
            f"Run 'wildtrack --list-detectors' to see details."
        )
    
    detector_class = DETECTOR_REGISTRY[name]["class"]
    return detector_class(**kwargs)


def list_detectors(verbose: bool = False) -> None:
    """
    Print available detectors and their details.
    
    Args:
        verbose: If True, show additional metadata
    """
    print("\n🔍 Available Detectors:\n")
    
    for name, info in DETECTOR_REGISTRY.items():
        print(f"  {name}")
        print(f"    {info['description']}")
        
        if verbose and info.get('source'):
            print(f"    Source: {info['source']}")
        
        if verbose and info.get('categories'):
            print(f"    Categories: {info['categories']}")
        
        print()


def get_available_detectors() -> list[str]:
    """Return list of available detector names."""
    return list(DETECTOR_REGISTRY.keys())


# Aliases for convenience
get_available = get_available_detectors
create_detector = get_detector