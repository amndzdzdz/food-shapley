"""Shapley-value analysis of food-ingredient segments in vision-language models.

Only reproducibility infrastructure is implemented so far: experiment
configuration, deterministic seeding, artifact layout, and run provenance.
The research pipeline itself (FoodSeg103 -> coalitions -> CLIP -> shapiq) is
not implemented yet; see ``docs/`` for the weekly plans.
"""

from food_shapley.config import ExperimentConfig, config_fingerprint, load_config
from food_shapley.paths import RunLayout, artifact_root
from food_shapley.provenance import RunProvenance, capture_provenance, write_run_metadata
from food_shapley.seeding import derive_seed, seed_everything

__all__ = [
    "ExperimentConfig",
    "RunLayout",
    "RunProvenance",
    "artifact_root",
    "capture_provenance",
    "config_fingerprint",
    "derive_seed",
    "load_config",
    "seed_everything",
    "write_run_metadata",
]

__version__ = "0.1.0"
