from __future__ import annotations

import numpy as np


def clip_vectors(vectors: np.ndarray, max_norm: float | np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    limit = np.asarray(max_norm, dtype=np.float32)
    if limit.ndim == 1:
        limit = limit[:, None]
    scale = np.minimum(1.0, limit / np.maximum(norms, 1e-6))
    return vectors * scale


def unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def heading(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    return np.arctan2(vy, vx)


def integrate_state(state: np.ndarray, accel: np.ndarray, dt: float, max_speed: float) -> np.ndarray:
    out = state.copy()
    out[:, 4:6] = accel
    per_agent_max_speed = np.minimum(float(max_speed), state[:, 8])
    out[:, 2:4] = clip_vectors(state[:, 2:4] + accel * dt, per_agent_max_speed)
    out[:, 0:2] = state[:, 0:2] + out[:, 2:4] * dt
    out[:, 6] = heading(out[:, 2], out[:, 3])
    return out
