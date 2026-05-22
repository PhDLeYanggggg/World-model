from __future__ import annotations

import numpy as np


def encode_agent_history(states: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask.astype(bool)
    safe = np.where(valid[..., None], states[..., :4], 0.0)
    denom = np.maximum(valid.sum(axis=0, keepdims=True).T, 1)
    return safe.sum(axis=0) / denom

