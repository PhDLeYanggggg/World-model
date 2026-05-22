from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class FailurePredictorHead:
    threshold: float = 0.35

    def predict(self, features: Dict[str, np.ndarray], horizon: int) -> Dict[str, np.ndarray]:
        speed = features.get("speed", 0.0)
        speed_change = features.get("speed_change", 0.0)
        nearest = features.get("nearest_neighbor_distance", np.full_like(speed, 999.0))
        density = features.get("local_density", np.zeros_like(speed))
        horizon_term = min(float(horizon) / 100.0, 1.0)
        score = (
            0.20 * horizon_term
            + 0.22 * np.clip(speed / 2.0, 0.0, 1.0)
            + 0.22 * np.clip(speed_change / 0.6, 0.0, 1.0)
            + 0.16 * np.clip(density / 8.0, 0.0, 1.0)
            + 0.20 * np.clip(1.0 - nearest / 4.0, 0.0, 1.0)
        )
        failure_prob = np.clip(score, 0.0, 1.0)
        correction_needed = np.clip((failure_prob - self.threshold) / max(1.0 - self.threshold, 1e-6), 0.0, 1.0)
        return {
            "failure_probability": failure_prob,
            "correction_needed_probability": correction_needed,
            "failure_type_distribution": np.stack([1.0 - failure_prob, failure_prob], axis=1),
        }

