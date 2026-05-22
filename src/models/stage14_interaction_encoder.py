from __future__ import annotations

import numpy as np


def scalar_interaction_features(states: np.ndarray, mask: np.ndarray, past_index: int = 9) -> np.ndarray:
    if states.ndim != 3 or states.shape[1] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    visible = mask[past_index].astype(bool)
    positions = states[past_index, :, :2]
    out = np.zeros((states.shape[1], 2), dtype=np.float32)
    for i in range(states.shape[1]):
        if not visible[i]:
            continue
        other = visible.copy()
        other[i] = False
        if not other.any():
            out[i] = [999.0, 0.0]
            continue
        d = np.linalg.norm(positions[other] - positions[i], axis=1)
        out[i] = [float(np.min(d)), float(other.sum())]
    return out

