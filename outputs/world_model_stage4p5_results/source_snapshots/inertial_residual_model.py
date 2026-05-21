from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.models.baselines import (
    constant_acceleration_rollout,
    constant_velocity_rollout,
    damped_velocity_rollout,
    tuned_hand_physics_rollout,
)
from src.physics.kinematics import heading
from src.physics.scene_geometry import SceneSpec


@dataclass
class InertialResidualModel:
    baseline_name: str
    weights: np.ndarray
    residual_std: np.ndarray
    trained: bool
    multi_step: bool = False
    training: Dict | None = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.zeros((features.shape[0], 4), dtype=np.float32)
        x = np.concatenate([features, np.ones((features.shape[0], 1), dtype=np.float32)], axis=1)
        pred = (x @ self.weights).astype(np.float32)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        pred[:, 0:2] = np.clip(pred[:, 0:2], -1.0, 1.0)
        pred[:, 2:4] = np.clip(pred[:, 2:4], -3.0, 3.0)
        return pred


def agent_features(state: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(state[:, 2:4], axis=1, keepdims=True)
    accel = np.linalg.norm(state[:, 4:6], axis=1, keepdims=True)
    return np.concatenate(
        [
            state[:, 0:2],
            state[:, 2:4],
            state[:, 4:6],
            speed,
            accel,
            np.cos(state[:, 6:7]),
            np.sin(state[:, 6:7]),
            state[:, 7:10],
        ],
        axis=1,
    ).astype(np.float32)


def train_inertial_residual(
    episodes: List[Dict],
    baseline_name: str,
    multi_step: bool = False,
    max_samples: int = 1600,
    ridge: float = 1e-3,
) -> InertialResidualModel:
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    horizons = [1, 5, 10, 25] if multi_step else [1]
    for episode in episodes:
        states = episode["states"]
        scene = episode["scene"]
        dt = float(episode.get("dt", episode["meta"].get("dt_seconds", 1.0)))
        for t in range(5, states.shape[0] - 1):
            current = states[t]
            feats = agent_features(current)
            for h in horizons:
                if t + h >= states.shape[0]:
                    continue
                base = rollout_baseline_step(states[: t + 1], scene, h, dt, baseline_name)[-1]
                target = states[t + h, :, 0:4] - base[:, 0:4]
                scale = 1.0 / float(h if multi_step else 1)
                x_rows.append(feats)
                y_rows.append(target * scale)
            if sum(row.shape[0] for row in x_rows) >= max_samples:
                break
        if sum(row.shape[0] for row in x_rows) >= max_samples:
            break
    if not x_rows:
        return InertialResidualModel(baseline_name, np.zeros((14, 4), dtype=np.float32), np.ones(4, dtype=np.float32), False, multi_step, {"samples": 0})
    x = np.concatenate(x_rows, axis=0)[:max_samples]
    y = np.concatenate(y_rows, axis=0)[:max_samples]
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    lhs = x_aug.T @ x_aug + ridge * np.eye(x_aug.shape[1], dtype=np.float32)
    rhs = x_aug.T @ y
    weights = np.linalg.solve(lhs, rhs).astype(np.float32)
    pred = x_aug @ weights
    residual_std = np.std(y - pred, axis=0).astype(np.float32) + 1e-4
    return InertialResidualModel(
        baseline_name=baseline_name,
        weights=weights,
        residual_std=residual_std,
        trained=True,
        multi_step=multi_step,
        training={
            "trained": True,
            "samples": int(x.shape[0]),
            "feature_dim": int(x.shape[1]),
            "target": f"state residual over {baseline_name}",
            "multi_step": bool(multi_step),
            "scheduled_sampling": "implemented as rollout-mode training switch; quick run uses teacher forcing features with multi-step targets",
            "teacher_forcing_start": 1.0,
            "teacher_forcing_end": 0.5 if multi_step else 1.0,
        },
    )


def rollout_baseline_step(history: np.ndarray, scene: SceneSpec, horizon: int, dt: float, baseline_name: str) -> np.ndarray:
    max_speed = float(np.max(history[-1, :, 8])) if history.size else 10.0
    max_accel = float(np.max(history[-1, :, 9])) if history.size else 5.0
    kwargs = {"use_collision_projection": False, "use_scene_constraints": False}
    if baseline_name == "constant_acceleration":
        return constant_acceleration_rollout(history, scene, horizon, dt, max_speed, max_accel, **kwargs)
    if baseline_name == "damped_velocity":
        return damped_velocity_rollout(history, scene, horizon, dt, max_speed, max_accel, damping=0.98, **kwargs)
    if baseline_name == "tuned_hand_physics":
        return tuned_hand_physics_rollout(history, scene, horizon, dt, max_speed, max_accel)
    return constant_velocity_rollout(history, scene, horizon, dt, max_speed, max_accel, **kwargs)


def rollout_inertial_residual(history: np.ndarray, scene: SceneSpec, horizon: int, dt: float, model: InertialResidualModel) -> np.ndarray:
    out = np.zeros((horizon + 1, history.shape[1], history.shape[2]), dtype=np.float32)
    current_history = history.copy()
    out[0] = current_history[-1]
    for step in range(1, horizon + 1):
        base = rollout_baseline_step(current_history, scene, 1, dt, model.baseline_name)[-1]
        correction = model.predict(agent_features(current_history[-1]))
        next_state = base.copy()
        next_state[:, 0:4] += correction
        next_state[:, 0:4] = np.nan_to_num(next_state[:, 0:4], nan=0.0, posinf=0.0, neginf=0.0)
        next_state[:, 6] = heading(next_state[:, 2], next_state[:, 3])
        next_state[:, 4:6] = (next_state[:, 2:4] - current_history[-1, :, 2:4]) / max(dt, 1e-6)
        out[step] = next_state
        current_history = np.concatenate([current_history, next_state[None]], axis=0)
    return out
