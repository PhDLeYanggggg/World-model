from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class Stage14MultimodalWorldModel:
    """Deterministic bounded-residual scaffold; not latent generative and not true 3D."""

    residual_clip: float = 0.1
    alpha_default: float = 0.0

    def predict(self, strongest_causal_baseline: np.ndarray) -> Dict[str, np.ndarray]:
        residual = np.zeros_like(strongest_causal_baseline, dtype=np.float32)
        alpha = np.full(strongest_causal_baseline.shape[:2], self.alpha_default, dtype=np.float32)
        prediction = strongest_causal_baseline.astype(np.float32) + alpha[..., None] * residual
        return {"prediction": prediction, "alpha": alpha, "bounded_residual": residual}

