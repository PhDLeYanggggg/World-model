from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class BoundedResidualDecoder:
    residual_clip: float = 0.25

    def decode(self, features: Dict[str, np.ndarray], failure_outputs: Dict[str, np.ndarray], horizon: int) -> Dict[str, np.ndarray]:
        velocity = features.get("last_velocity")
        if velocity is None:
            n = len(failure_outputs["failure_probability"])
            velocity = np.zeros((n, 2), dtype=np.float64)
        norm = np.linalg.norm(velocity, axis=1, keepdims=True)
        direction = np.divide(velocity, np.maximum(norm, 1e-6), out=np.zeros_like(velocity), where=norm > 1e-6)
        magnitude = self.residual_clip * failure_outputs["correction_needed_probability"][:, None]
        return {"bounded_residual": direction * magnitude, "residual_norm": magnitude[:, 0]}

