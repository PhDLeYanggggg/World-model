from __future__ import annotations

from src.training.train_baseline_failure_predictor import train_predictor


def evaluate_predictor():
    # Training writes train/val/test metrics from causal features; this wrapper
    # exists for the Stage 6 command surface.
    return train_predictor()

