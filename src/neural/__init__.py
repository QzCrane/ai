"""Layer 2 (Neural): Route B Time-and-Structure-First PyTorch Backbone & Training Pipeline."""
from src.brain import (
    ablation_harness,
    config_profiles,
    prepare_seed_corpus,
    sovereign_brain_pytorch,
    training_engine,
)

__all__ = [
    "ablation_harness",
    "config_profiles",
    "prepare_seed_corpus",
    "sovereign_brain_pytorch",
    "training_engine",
]
