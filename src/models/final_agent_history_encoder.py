from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class AgentHistoryEncoder:
    """Causal encoder for per-agent past kinematics."""

    def encode(self, past_states: np.ndarray, valid_mask: np.ndarray) -> Dict[str, np.ndarray]:
        states = np.asarray(past_states, dtype=np.float64)
        mask = np.asarray(valid_mask, dtype=bool)
        if states.ndim != 3:
            raise ValueError("past_states must have shape [time, agents, features]")
        if mask.shape != states.shape[:2]:
            raise ValueError("valid_mask must have shape [time, agents]")
        last_idx = np.maximum(mask.sum(axis=0) - 1, 0)
        agent_idx = np.arange(states.shape[1])
        last = states[last_idx, agent_idx]
        speed = np.linalg.norm(last[:, 2:4], axis=1)
        accel = np.linalg.norm(last[:, 4:6], axis=1) if states.shape[2] >= 6 else np.zeros(states.shape[1])
        valid_fraction = mask.mean(axis=0)
        heading = last[:, 6] if states.shape[2] > 6 else np.zeros(states.shape[1])
        speed_hist = np.linalg.norm(states[:, :, 2:4], axis=2) if states.shape[2] >= 4 else np.zeros(states.shape[:2])
        speed_change = np.where(mask.any(axis=0), np.nanmax(np.where(mask, speed_hist, np.nan), axis=0) - np.nanmin(np.where(mask, speed_hist, np.nan), axis=0), 0.0)
        speed_change = np.nan_to_num(speed_change)
        return {
            "last_position": last[:, :2],
            "last_velocity": last[:, 2:4] if states.shape[2] >= 4 else np.zeros((states.shape[1], 2)),
            "speed": speed,
            "acceleration_norm": accel,
            "heading": heading,
            "speed_change": speed_change,
            "valid_fraction": valid_fraction,
        }

