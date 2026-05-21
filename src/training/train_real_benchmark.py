from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.features.world_state_features import stage3_agent_feature_vector
from src.physics.social_force import compute_social_force_acceleration


@dataclass
class LinearResidualModel:
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    residual_std: np.ndarray
    trained: bool = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = (features - self.feature_mean) / np.maximum(self.feature_std, 1e-6)
        x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        return x @ self.weights


def train_real_residual_model(episodes: List[Dict], history_steps: int = 6, ridge: float = 1e-3) -> Dict:
    features, targets = [], []
    for episode in episodes:
        states, scene = episode["states"], episode["scene"]
        if states.shape[0] <= history_steps + 1:
            continue
        for t in range(history_steps - 1, states.shape[0] - 1):
            current = states[t]
            base = compute_social_force_acceleration(current, scene, float(np.max(current[:, 9])))
            residual = states[t + 1, :, 4:6] - base
            for i in range(current.shape[0]):
                features.append(real_agent_features(current, scene, i))
                targets.append(residual[i])
    if not features:
        return {"deterministic": empty_model(), "stochastic": empty_model(), "training": {"trained": False, "reason": "no train samples"}}
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    xn = (x - mean) / std
    xb = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=np.float32)], axis=1)
    reg = ridge * np.eye(xb.shape[1], dtype=np.float32)
    weights = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
    pred = xb @ weights
    residual_std = np.std(y - pred, axis=0) + 1e-4
    model = LinearResidualModel(weights=weights, feature_mean=mean, feature_std=std, residual_std=residual_std)
    return {
        "deterministic": model,
        "stochastic": model,
        "training": {
            "trained": True,
            "samples": int(x.shape[0]),
            "feature_dim": int(x.shape[1]),
            "target": "real residual acceleration = true_next_acceleration - hand_physics_acceleration",
            "residual_std": residual_std.tolist(),
        },
    }


def empty_model() -> LinearResidualModel:
    return LinearResidualModel(np.zeros((1, 2), dtype=np.float32), np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32), np.ones(2, dtype=np.float32), trained=False)


def real_agent_features(state: np.ndarray, scene, index: int) -> np.ndarray:
    agent = state[index]
    speed = np.linalg.norm(agent[2:4])
    accel = np.linalg.norm(agent[4:6])
    base = np.asarray(
        [
            agent[0] / max(1.0, scene.width),
            agent[1] / max(1.0, scene.height),
            agent[2] / max(1e-6, agent[8]),
            agent[3] / max(1e-6, agent[8]),
            agent[4] / max(1e-6, agent[9]),
            agent[5] / max(1e-6, agent[9]),
            speed / max(1e-6, agent[8]),
            accel / max(1e-6, agent[9]),
            np.sin(agent[6]),
            np.cos(agent[6]),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, stage3_agent_feature_vector(state, scene, index)], axis=0).astype(np.float32)
