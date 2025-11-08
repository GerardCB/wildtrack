"""
Classifier registry for WildTrack.

Makes it easy to:
- Add new classifiers
- List available classifiers
- Instantiate classifiers by name
"""

from typing import Dict, Type, Any
from .base import Classifier
from .speciesnet import SpeciesNetClassifier


# Registry structure: name -> {class, description, metadata}
CLASSIFIER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "speciesnet": {
        "class": SpeciesNetClassifier,
        "description": "Google SpeciesNet trained on 65M+ camera trap images (2000+ species)",
        "source": "https://github.com/google/cameratrapai",
        "notes": "Requires: pip install speciesnet"
    },
}


def register_classifier(
    name: str,
    classifier_class: Type[Classifier],
    description: str,
    source: str = "",
    **metadata
):
    """
    Register a new classifier.
    
    Args:
        name: Unique identifier (e.g., "my-classifier")
        classifier_class: Class that implements Classifier interface
        description: Short description of the classifier
        source: URL to paper or GitHub repo
        **metadata: Additional metadata (species_list, etc.)
    """
    if name in CLASSIFIER_REGISTRY:
        raise ValueError(f"Classifier '{name}' already registered")
    
    if not issubclass(classifier_class, Classifier):
        raise TypeError(f"classifier_class must inherit from Classifier")
    
    CLASSIFIER_REGISTRY[name] = {
        "class": classifier_class,
        "description": description,
        "source": source,
        **metadata
    }


def get_classifier(name: str, **kwargs) -> Classifier:
    """
    Instantiate a classifier by name.
    
    Args:
        name: Classifier name from registry
        **kwargs: Arguments passed to classifier constructor
                 (e.g., conf_thresh=0.5)
    
    Returns:
        Initialized Classifier instance
    
    Example:
        classifier = get_classifier("megaclassifier", conf_thresh=0.6)
    """
    if name not in CLASSIFIER_REGISTRY:
        available = ", ".join(CLASSIFIER_REGISTRY.keys()) if CLASSIFIER_REGISTRY else "none"
        raise ValueError(
            f"Unknown classifier '{name}'.\n"
            f"Available classifiers: {available}\n"
            f"Run 'wildtrack --list-classifiers' to see details."
        )
    
    classifier_class = CLASSIFIER_REGISTRY[name]["class"]
    return classifier_class(**kwargs)


def list_classifiers(verbose: bool = False) -> None:
    """
    Print available classifiers and their details.
    
    Args:
        verbose: If True, show additional metadata
    """
    print("\n🏷️  Available Classifiers:\n")
    
    if not CLASSIFIER_REGISTRY:
        print("  No classifiers available yet.")
        print("  Check back soon or add your own!")
        print()
        return
    
    for name, info in CLASSIFIER_REGISTRY.items():
        print(f"  {name}")
        print(f"    {info['description']}")
        
        if verbose and info.get('source'):
            print(f"    Source: {info['source']}")
        
        print()


def get_available_classifiers() -> list[str]:
    """Return list of available classifier names."""
    return list(CLASSIFIER_REGISTRY.keys())


# Aliases for convenience
get_available = get_available_classifiers
create_classifier = get_classifier