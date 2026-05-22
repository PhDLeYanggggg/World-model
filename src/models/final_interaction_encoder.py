from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class InteractionFeatureEncoder:
    """Past-only scalar interaction features for variable-agent scenes."""

    def encode(self, past_states: np.ndarray, valid_mask: np.ndarray) -> Dict[str, np.ndarray]:
        states = np.asarray(past_states, dtype=np.float64)
        mask = np.asarray(valid_mask, dtype=bool)
        n = states.shape[1]
        last_valid = mask[-1] if len(mask) else np.zeros(n, dtype=bool)
        pos = states[-1, :, :2]
        vel = states[-1, :, 2:4] if states.shape[2] >= 4 else np.zeros((n, 2))
        nearest = np.full(n, 999.0, dtype=np.float64)
        density = np.zeros(n, dtype=np.float64)
        closing = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if not last_valid[i]:
                continue
            diffs = pos - pos[i]
            dists = np.linalg.norm(diffs, axis=1)
            candidates = (dists > 1e-6) & last_valid
            if candidates.any():
                nearest[i] = float(dists[candidates].min())
                density[i] = float((dists[candidates] < 3.0).sum())
                j = int(np.where(candidates)[0][np.argmin(dists[candidates])])
                rel_pos = pos[j] - pos[i]
                rel_vel = vel[j] - vel[i]
                closing[i] = float(-np.dot(rel_pos, rel_vel) / max(np.linalg.norm(rel_pos), 1e-6))
        return {"nearest_neighbor_distance": nearest, "local_density": density, "closing_speed": closing}

