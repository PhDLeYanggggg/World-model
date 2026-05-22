from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Stage15BaselinePreservingWorldModel:
    residual_clip: float = 0.05
    intervention_threshold: float = 0.5

    def predict(self, baseline: np.ndarray, failure_score: np.ndarray | None = None) -> dict[str, np.ndarray]:
        score = failure_score if failure_score is not None else np.zeros(baseline.shape[:2], dtype=np.float32)
        intervention = 1.0 / (1.0 + np.exp(-(score - self.intervention_threshold)))
        residual = np.zeros_like(baseline, dtype=np.float32)
        return {"prediction": baseline + intervention[..., None] * residual, "intervention": intervention, "bounded_residual": residual}

