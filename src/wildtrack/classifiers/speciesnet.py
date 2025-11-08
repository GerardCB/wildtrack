"""
SpeciesNet classifier for WildTrack.

Google's SpeciesNet is a species classifier trained on 65M+ camera trap images,
covering 2000+ species. It's designed to work with MegaDetector.

Source: https://github.com/google/cameratrapai
Paper: https://doi.org/10.1049/cvi2.12318
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import cv2
from typing import Tuple, Optional, List
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
        """Resolve device string ('auto', 'cpu', 'cuda', 'mps')."""
        if device != "auto":
            return device
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except Exception:
            return "cpu"

    def _load_classifier(self):
        """
        Load SpeciesNet classifier from the external package, adapting to its
        constructor signature to avoid unexpected-kwarg errors.
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

        # Canonical model spec we control (the external lib may accept different names)
        model_spec = f"kaggle:google/speciesnet/pyTorch/{self.model_version}"

        # Introspect the external classifier's __init__ to determine accepted kwargs
        import inspect
        init_sig = inspect.signature(SNClassifier.__init__)
        init_params = init_sig.parameters

        # Build kwargs dynamically to match what the external class accepts
        ctor_kwargs = {}

        # Most likely names for the model spec that appear across versions
        if "model" in init_params:
            ctor_kwargs["model"] = model_spec
        elif "model_spec" in init_params:
            ctor_kwargs["model_spec"] = model_spec
        elif "model_id" in init_params:
            ctor_kwargs["model_id"] = model_spec
        # (Do NOT pass 'model_path' – that caused the crash)

        # Device handling (many versions accept 'device'; if not, we'll retry)
        if "device" in init_params:
            ctor_kwargs["device"] = self.device

        # First attempt: kwargs only
        try:
            classifier = SNClassifier(**ctor_kwargs)
            print(f"✓ SpeciesNet loaded on {self.device.upper()}")
            return classifier
        except TypeError as e:
            # Retry strategies: (1) add positional model_spec, (2) drop device kwarg if unsupported
            last_err = e

        # Second attempt: positional model_spec + (optional) device kwarg
        try:
            if "device" in init_params:
                classifier = SNClassifier(model_spec, device=self.device)
            else:
                classifier = SNClassifier(model_spec)
            print(f"✓ SpeciesNet loaded on {self.device.upper()}")
            return classifier
        except TypeError as e:
            last_err = e

        # Third attempt: positional only (some versions may ignore device or set it later)
        try:
            classifier = SNClassifier(model_spec)
            print(f"✓ SpeciesNet loaded (device set separately if supported)")
            return classifier
        except TypeError as e:
            last_err = e

        # If we get here, surface a clear error with the detected signature
        raise TypeError(
            "Could not construct speciesnet.classifier.SpeciesNetClassifier with any of the tried signatures.\n"
            f"Detected __init__ signature: {init_sig}\n"
            f"Last error: {last_err}"
        )

    
    def _resize_for_version(self, crop_rgb: np.ndarray) -> np.ndarray:
        """
        For 'v4.0.1a' (always-crop), resize to 480x480 so the ensemble gets a consistent view.
        For 'v4.0.1b' (full image), we still pass crops (SpeciesNet is an ensemble), but no strict size.
        """
        if self.model_version.lower().startswith("v4.0.1a"):
            return cv2.resize(crop_rgb, (480, 480), interpolation=cv2.INTER_AREA)
        return crop_rgb

    def classify(self, image_bgr: np.ndarray, bbox: np.ndarray) -> Tuple[str, float]:
        """Single-box helper that uses the batch path under the hood."""
        out = self.classify_batch(image_bgr, np.asarray([bbox], dtype=np.float32))
        return out[0] if out else ("error", 0.0)

    def classify_batch(
        self,
        image_bgr: np.ndarray,
        bboxes: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Batch classify bounding boxes by:
          1) cropping to temp folder
          2) invoking SpeciesNet CLI
          3) parsing predictions.json
        """
        H, W = image_bgr.shape[:2]
        if bboxes is None or len(bboxes) == 0:
            return []

        # 1) Prepare temp workspace
        tmp_root = tempfile.mkdtemp(prefix="speciesnet_")
        imgs_dir = Path(tmp_root) / "images"
        out_json = Path(tmp_root) / "predictions.json"
        imgs_dir.mkdir(parents=True, exist_ok=True)

        # 2) Write crops
        file_map = []  # [(index_in_batch, filepath)]
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                # write sentinel blank so we preserve indexing
                file_map.append((i, None))
                continue

            crop = image_bgr[y1:y2, x1:x2]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                file_map.append((i, None))
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_rgb = self._resize_for_version(crop_rgb)

            # Save as JPEG
            fn = imgs_dir / f"box_{i:04d}.jpg"
            cv2.imwrite(str(fn), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            file_map.append((i, str(fn)))

        # If all invalid, short-circuit
        if all(p is None for _, p in file_map):
            return [("unknown", 0.0) for _ in range(len(bboxes))]

        # 3) Build CLI command
        # Minimal required args (SpeciesNet detects and classifies; it expects folders)
        # We pass country/admin1_region only if provided (geo filter belongs to ensemble stage)
        cmd = [
            sys.executable, "-m", "speciesnet.scripts.run_model",
            "--folders", str(imgs_dir),
            "--predictions_json", str(out_json),
        ]
        if self.country:
            cmd += ["--country", self.country]
        if self.admin1_region:
            cmd += ["--admin1_region", self.admin1_region]
        # If the script supports selecting model version, add here (kept optional)
        # e.g., cmd += ["--model_version", self.model_version]

        # 4) Run CLI
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("  Warning: SpeciesNet CLI failed:")
            try:
                if e.stdout: print(e.stdout.decode(errors="ignore"))
                if e.stderr: print(e.stderr.decode(errors="ignore"))
            except Exception:
                pass
            # Return safe defaults
            return [("error", 0.0) for _ in range(len(bboxes))]
        except Exception as e:
            print(f"  Warning: SpeciesNet CLI invocation error: {e}")
            return [("error", 0.0) for _ in range(len(bboxes))]

        # 5) Parse predictions.json and map back to input order
        pred_by_file = {}
        try:
            with open(out_json, "r") as f:
                data = json.load(f)

            # SpeciesNet outputs can be keyed per file; handle common shapes:
            #  - {"images": [{"file": "...", "prediction": "...", "prediction_score": 0.xxx, ...}, ...]}
            #  - or {"predictions": [{"image_path": "...", "classifications": {"classes": [...], "scores": [...]}}, ...]}
            if isinstance(data, dict):
                images = data.get("images") or data.get("predictions") or []
                for item in images:
                    # Try to extract a filepath
                    fp = item.get("file") or item.get("image_path") or item.get("image") or item.get("filepath")
                    if not fp:
                        # Sometimes only a basename is given; try best-effort
                        fp = item.get("filename")
                    # Normalize to absolute path we wrote
                    if fp and not os.path.isabs(fp):
                        fp = str((imgs_dir / Path(fp).name).resolve())

                    # Get class + score
                    species, score = None, None
                    # 1) flat fields
                    if "prediction" in item and "prediction_score" in item:
                        species = item["prediction"]
                        score = float(item["prediction_score"])
                    # 2) nested classifications
                    if (species is None) and ("classifications" in item):
                        cls = item["classifications"] or {}
                        classes = cls.get("classes") or []
                        scores = cls.get("scores") or []
                        if classes and scores:
                            species, score = classes[0], float(scores[0])

                    if fp and (species is not None) and (score is not None):
                        pred_by_file[Path(fp).resolve().as_posix()] = (species, score)
        except Exception as e:
            print(f"  Warning: Failed to parse SpeciesNet predictions: {e}")
            return [("error", 0.0) for _ in range(len(bboxes))]

        # 6) Rebuild outputs in the original bbox order
        results: List[Tuple[str, float]] = []
        for i, fp in file_map:
            if fp is None:
                results.append(("unknown", 0.0))
            else:
                key = Path(fp).resolve().as_posix()
                if key in pred_by_file:
                    sp, sc = pred_by_file[key]
                    if sc < self.conf_thresh:
                        results.append(("unknown", float(sc)))
                    else:
                        results.append((self._format_species_name(sp), float(sc)))
                else:
                    results.append(("unknown", 0.0))

        # 7) Cleanup temp dir
        try:
            import shutil
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass

        return results

    def _parse_speciesnet_label(self, label: str):
        """
        Parse SpeciesNet semicolon-delimited taxonomy strings:
        UUID;Class;Order;Family;Genus;Species;CommonName
        Returns a dict with fields (may be empty strings).
        """
        if label is None:
            label = ""
        parts = [p.strip() for p in label.split(";")]
        parts += [""] * (7 - len(parts))  # pad to length 7
        return {
            "uuid": parts[0],
            "clazz": parts[1],   # avoid 'class' keyword
            "order": parts[2],
            "family": parts[3],
            "genus": parts[4],
            "species": parts[5],
            "common": parts[6],
            "raw": label,
        }

    def _format_species_name(self, label: str) -> str:
        """
        Choose the most specific/human-friendly display name.
        Priority: Common name > binomial species > genus > family > raw.
        """
        if not label:
            return "Unknown"

        simple = label.lower()
        if simple in {"blank", "unknown", "animal"}:
            # Preserve simple tokens with leading capital
            return simple.capitalize()

        info = self._parse_speciesnet_label(label)

        # 1) Common name
        if info["common"]:
            # Title-case the common name
            return info["common"].strip().title()

        # 2) Species (binomial)
        if info["species"]:
            sp = info["species"].replace("_", " ").strip()
            toks = sp.split()
            if len(toks) >= 2:
                genus = toks[0].capitalize()
                epithet = " ".join(t.lower() for t in toks[1:])
                return f"{genus} {epithet}"
            # If it's a single token, just title-case it
            return sp.title()

        # 3) Genus only
        if info["genus"]:
            return f"{info['genus'].strip().title()} (genus)"

        # 4) Family only
        if info["family"]:
            fam = info["family"].strip()
            # Normalize to “Elephantidae (family)”
            fam = fam.replace(" family", "").replace(" Family", "")
            return f"{fam} (family)"

        # 5) Fallback
        return label.replace("_", " ").strip().title()

