from __future__ import annotations

import numpy as np


def encode_agent_history(history: np.ndarray, mask: np.ndarray, agent_idx: int, horizon: int, max_horizon: int = 10) -> np.ndarray:
    valid = mask[:, agent_idx]
    if not valid.any():
        return np.zeros(12, dtype=float)
    last_idx = int(np.where(valid)[0][-1])
    last = history[last_idx, agent_idx]
    prev = history[int(np.where(valid)[0][-2]), agent_idx] if valid.sum() >= 2 else last
    heading_rate = float(np.angle(np.exp(1j * (last[6] - prev[6]))))
    speed_change = float(last[7] - prev[7])
    displacement = last[0:2] - history[int(np.where(valid)[0][0]), agent_idx, 0:2]
    return np.asarray(
        [
            last[2],
            last[3],
            last[4],
            last[5],
            last[7],
            np.sin(last[6]),
            np.cos(last[6]),
            heading_rate,
            speed_change,
            float(valid.mean()),
            float(np.linalg.norm(displacement)),
            float(horizon / max(max_horizon, 1)),
        ],
        dtype=float,
    )
