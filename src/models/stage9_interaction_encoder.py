from __future__ import annotations

import numpy as np


def encode_interaction(history: np.ndarray, mask: np.ndarray, agent_idx: int) -> np.ndarray:
    last = history[-1]
    valid = mask[-1].copy()
    if agent_idx >= len(valid) or not valid[agent_idx]:
        return np.zeros(10, dtype=float)
    valid[agent_idx] = False
    pos_i = last[agent_idx, 0:2]
    vel_i = last[agent_idx, 2:4]
    pos = last[valid, 0:2]
    vel = last[valid, 2:4]
    if len(pos) == 0:
        return np.zeros(10, dtype=float)
    rel = pos - pos_i[None, :]
    dist = np.linalg.norm(rel, axis=1)
    nearest = int(np.argmin(dist))
    rel_v = vel - vel_i[None, :]
    closing = -np.sum(rel * rel_v, axis=1) / np.maximum(dist, 1e-6)
    ttc = dist / np.maximum(closing, 1e-3)
    bbox = np.maximum(pos.max(axis=0) - pos.min(axis=0), 1.0) if len(pos) > 1 else np.ones(2)
    density = (len(pos) + 1) / max(float(np.prod(bbox)), 1.0)
    return np.asarray(
        [
            min(float(dist[nearest]), 50.0) / 50.0,
            min(float(dist.mean()), 50.0) / 50.0,
            min(float(np.min(ttc)), 50.0) / 50.0,
            min(float(np.max(closing)), 20.0) / 20.0,
            min(float(density), 1.0),
            min(float(len(pos) + 1) / 64.0, 1.0),
            float(rel[nearest, 0] / 50.0),
            float(rel[nearest, 1] / 50.0),
            float(rel_v[nearest, 0] / 10.0),
            float(rel_v[nearest, 1] / 10.0),
        ],
        dtype=float,
    )
