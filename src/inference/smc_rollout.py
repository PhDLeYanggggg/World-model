from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from src.models.neural_residual_world_model import denormalize, entity_sequence_features, neighbor_features, normalize, obstacle_features
from src.physics.collision import project_collisions
from src.physics.constraints import project_scene_constraints
from src.physics.kinematics import clip_vectors, integrate_state
from src.physics.scene_geometry import SceneSpec
from src.physics.social_force import compute_social_force_acceleration
from src.utils.metrics import effective_sample_size, normalize_log_weights, systematic_resample_indexes


PROPOSAL_NAMES = {
    "hand_physics_proposal": "hand_physics_SMC",
    "learned_neural_proposal": "learned_neural_SMC",
    "physics_plus_neural_residual_proposal": "physics_plus_neural_residual_SMC",
}


def rollout_smc(
    history: np.ndarray,
    scene: SceneSpec,
    cfg: Dict,
    model: torch.nn.Module | None,
    normalization: Dict | None,
    proposal: str,
    particles: int,
    horizon: int = 100,
    stochastic: bool = False,
    seed: int = 0,
) -> Dict:
    rng = np.random.default_rng(seed)
    dt = float(cfg["world"]["dt"])
    max_speed = float(cfg["world"]["max_speed_mps"])
    max_accel = float(cfg["world"]["max_accel_mps2"])
    histories = np.repeat(history[None].astype(np.float32), particles, axis=0)
    histories[:, -1, :, :2] += rng.normal(0.0, 0.025, size=histories[:, -1, :, :2].shape).astype(np.float32)
    histories = initialize_latent_goals(histories, scene, cfg, rng)
    trajectories = np.zeros((particles, horizon + 1, history.shape[1], history.shape[2]), dtype=np.float32)
    trajectories[:, 0] = histories[:, -1]
    log_weights = np.zeros(particles, dtype=np.float64)
    diagnostics = []

    for step in range(1, horizon + 1):
        accel_batch, proposal_logp = proposal_acceleration_batch(
            histories, scene, cfg, model, normalization, proposal, rng, stochastic=stochastic
        )
        for p in range(particles):
            current = histories[p, -1]
            next_state = integrate_state(current, accel_batch[p], dt, max_speed)
            after_collision, cinfo = project_collisions(next_state)
            after_scene, sinfo = project_scene_constraints(after_collision, scene)
            after_scene[:, 4:6] = clip_vectors((after_scene[:, 2:4] - current[:, 2:4]) / dt, after_scene[:, 9])
            log_weights[p] += proposal_logp[p]
            log_weights[p] -= 3.5 * float(cinfo["collision_projection_cost"])
            log_weights[p] -= 2.2 * float(sinfo["obstacle_projection_cost"])
            log_weights[p] -= 1.4 * float(sinfo["boundary_violation"])
            histories[p] = np.concatenate([histories[p, 1:], after_scene[None]], axis=0)
            trajectories[p, step] = after_scene

        weights = normalize_log_weights(log_weights)
        if step in set(int(h) for h in cfg["inference"]["horizons"]) or step == horizon:
            diagnostics.append(
                {
                    "step": int(step),
                    "ESS": round(float(effective_sample_size(weights)), 4),
                    "min_log_weight": round(float(np.min(log_weights)), 4),
                    "max_log_weight": round(float(np.max(log_weights)), 4),
                }
            )
        if effective_sample_size(weights) < particles * 0.45 and step < horizon:
            indexes = systematic_resample_indexes(weights, rng)
            histories = histories[indexes]
            trajectories = trajectories[indexes]
            histories = rejuvenate_latent_goals(histories, scene, cfg, rng)
            log_weights = np.zeros(particles, dtype=np.float64)
        elif step % 25 == 0 and step < horizon:
            histories = rejuvenate_latent_goals(histories, scene, cfg, rng, probability_scale=0.35)

    weights = normalize_log_weights(log_weights)
    return {
        "proposal": proposal,
        "trajectories": trajectories,
        "weights": weights,
        "diagnostics": diagnostics,
        "particles": int(particles),
    }


def initialize_latent_goals(histories: np.ndarray, scene: SceneSpec, cfg: Dict, rng: np.random.Generator) -> np.ndarray:
    out = histories.copy()
    keep_probability = float(cfg["inference"].get("latent_goal_keep_probability", 0.5))
    exit_probability = float(cfg["inference"].get("latent_goal_exit_probability", 0.3))
    noise_m = float(cfg["inference"].get("latent_goal_noise_m", 2.0))
    for p in range(out.shape[0]):
        for i in range(out.shape[2]):
            current_goal = out[p, -1, i, 10:12].copy()
            draw = rng.random()
            if draw < keep_probability:
                goal = current_goal + rng.normal(0.0, noise_m * 0.18, size=2)
            elif draw < keep_probability + exit_probability:
                goal = sample_exit_goal(out[p, -1, i], scene, rng)
            else:
                velocity_goal = out[p, -1, i, :2] + goal_direction_from_velocity(out[p, -1, i]) * rng.uniform(8.0, 20.0)
                goal = clamp_goal(velocity_goal + rng.normal(0.0, noise_m, size=2), scene)
            out[p, :, i, 10:12] = goal
    return out.astype(np.float32)


def rejuvenate_latent_goals(
    histories: np.ndarray,
    scene: SceneSpec,
    cfg: Dict,
    rng: np.random.Generator,
    probability_scale: float = 1.0,
) -> np.ndarray:
    out = histories.copy()
    probability = float(cfg["inference"].get("latent_goal_rejuvenation_probability", 0.08)) * probability_scale
    noise_m = float(cfg["inference"].get("latent_goal_noise_m", 2.0))
    for p in range(out.shape[0]):
        for i in range(out.shape[2]):
            if rng.random() >= probability:
                continue
            if rng.random() < 0.55:
                goal = sample_exit_goal(out[p, -1, i], scene, rng)
            else:
                goal = clamp_goal(out[p, -1, i, 10:12] + rng.normal(0.0, noise_m, size=2), scene)
            out[p, :, i, 10:12] = goal
    return out.astype(np.float32)


def sample_exit_goal(agent: np.ndarray, scene: SceneSpec, rng: np.random.Generator) -> np.ndarray:
    exits = list(scene.exits.values())
    heading = goal_direction_from_velocity(agent)
    scores = []
    for exit_point in exits:
        direction = np.asarray(exit_point, dtype=np.float32) - agent[:2]
        norm = np.linalg.norm(direction)
        direction = direction / max(1e-6, norm)
        scores.append(max(0.05, float(np.dot(direction, heading)) + 0.35))
    probs = np.asarray(scores, dtype=np.float64)
    probs /= probs.sum()
    goal = np.asarray(exits[int(rng.choice(len(exits), p=probs))], dtype=np.float32)
    return clamp_goal(goal + rng.normal(0.0, 0.75, size=2), scene)


def goal_direction_from_velocity(agent: np.ndarray) -> np.ndarray:
    speed = float(np.linalg.norm(agent[2:4]))
    if speed < 1e-4:
        direction = agent[10:12] - agent[:2]
    else:
        direction = agent[2:4]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return (direction / norm).astype(np.float32)


def clamp_goal(goal: np.ndarray, scene: SceneSpec) -> np.ndarray:
    return np.asarray([np.clip(goal[0], 0.5, scene.width - 0.5), np.clip(goal[1], 0.5, scene.height - 0.5)], dtype=np.float32)


def rollout_learned_single(
    history: np.ndarray,
    scene: SceneSpec,
    cfg: Dict,
    model: torch.nn.Module,
    normalization: Dict,
    horizon: int = 100,
    stochastic: bool = False,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = float(cfg["world"]["dt"])
    max_speed = float(cfg["world"]["max_speed_mps"])
    max_accel = float(cfg["world"]["max_accel_mps2"])
    out = np.zeros((horizon + 1, history.shape[1], history.shape[2]), dtype=np.float32)
    rolling = history.astype(np.float32).copy()
    out[0] = rolling[-1]
    current = rolling[-1].copy()
    for step in range(1, horizon + 1):
        physics = compute_social_force_acceleration(current, scene, max_accel)
        residual = predict_residual_acceleration(rolling, scene, model, normalization, stochastic=stochastic, rng=rng)
        accel = clip_vectors(physics + residual, current[:, 9])
        current = integrate_state(current, accel, dt, max_speed)
        current, _ = project_collisions(current)
        current, _ = project_scene_constraints(current, scene)
        rolling = np.concatenate([rolling[1:], current[None]], axis=0)
        out[step] = current
    return out


def proposal_acceleration_batch(
    histories: np.ndarray,
    scene: SceneSpec,
    cfg: Dict,
    model: torch.nn.Module | None,
    normalization: Dict | None,
    proposal: str,
    rng: np.random.Generator,
    stochastic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    particles, _, agents, _ = histories.shape
    max_accel = float(cfg["world"]["max_accel_mps2"])
    base = np.zeros((particles, agents, 2), dtype=np.float32)
    for p in range(particles):
        base[p] = compute_social_force_acceleration(histories[p, -1], scene, max_accel)

    if proposal == "hand_physics_proposal" or model is None or normalization is None:
        noise_std = 0.10
        accel = base
    else:
        residual = predict_residual_acceleration_batch(histories, scene, model, normalization, stochastic=stochastic, rng=rng)
        if proposal == "learned_neural_proposal":
            accel = residual
            noise_std = 0.17
        elif proposal == "physics_plus_neural_residual_proposal":
            accel = base + residual
            noise_std = 0.09
        else:
            raise ValueError(f"Unknown proposal: {proposal}")

    noise = rng.normal(0.0, noise_std, size=accel.shape).astype(np.float32)
    logp = -0.5 * np.mean((noise / noise_std) ** 2, axis=(1, 2))
    clipped = np.stack([clip_vectors(accel[p] + noise[p], histories[p, -1, :, 9]) for p in range(particles)]).astype(np.float32)
    return clipped, logp.astype(np.float64)


def predict_residual_acceleration_batch(
    histories: np.ndarray,
    scene: SceneSpec,
    model: torch.nn.Module,
    normalization: Dict,
    stochastic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    particles, _, agents, _ = histories.shape
    entity_rows, neighbor_rows, obstacle_rows = [], [], []
    for p in range(particles):
        current = histories[p, -1]
        for i in range(agents):
            entity_rows.append(entity_sequence_features(histories[p], scene, i))
            neighbor_rows.append(neighbor_features(current, i))
            obstacle_rows.append(obstacle_features(current, scene, i))
    pred = _predict_residual_arrays(
        np.asarray(entity_rows, dtype=np.float32),
        np.asarray(neighbor_rows, dtype=np.float32),
        np.asarray(obstacle_rows, dtype=np.float32),
        model,
        normalization,
        stochastic,
        rng,
    )
    return pred.reshape(particles, agents, 2).astype(np.float32)


def predict_residual_acceleration(
    history: np.ndarray,
    scene: SceneSpec,
    model: torch.nn.Module,
    normalization: Dict,
    stochastic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    current = history[-1]
    entity_rows, neighbor_rows, obstacle_rows = [], [], []
    for i in range(current.shape[0]):
        entity_rows.append(entity_sequence_features(history, scene, i))
        neighbor_rows.append(neighbor_features(current, i))
        obstacle_rows.append(obstacle_features(current, scene, i))
    return _predict_residual_arrays(
        np.asarray(entity_rows, dtype=np.float32),
        np.asarray(neighbor_rows, dtype=np.float32),
        np.asarray(obstacle_rows, dtype=np.float32),
        model,
        normalization,
        stochastic,
        rng,
    ).astype(np.float32)


def _predict_residual_arrays(
    entity: np.ndarray,
    neighbor: np.ndarray,
    obstacle: np.ndarray,
    model: torch.nn.Module,
    normalization: Dict,
    stochastic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    model.eval()
    entity_n = normalize(entity, normalization["entity_mean"], normalization["entity_std"])
    neighbor_n = normalize(neighbor, normalization["neighbor_mean"], normalization["neighbor_std"])
    obstacle_n = normalize(obstacle, normalization["obstacle_mean"], normalization["obstacle_std"])
    preds = []
    latent_dim = int(getattr(model, "latent_dim", 0))
    with torch.no_grad():
        for start in range(0, entity_n.shape[0], 4096):
            end = min(entity_n.shape[0], start + 4096)
            latent = None
            if latent_dim:
                scale = 0.55 if stochastic else 0.0
                latent = torch.tensor(rng.normal(0.0, scale, size=(end - start, latent_dim)), dtype=torch.float32)
            pred, _ = model(
                torch.tensor(entity_n[start:end], dtype=torch.float32),
                torch.tensor(neighbor_n[start:end], dtype=torch.float32),
                torch.tensor(obstacle_n[start:end], dtype=torch.float32),
                latent,
            )
            preds.append(pred.cpu().numpy())
    pred_scaled = np.concatenate(preds, axis=0)
    return denormalize(pred_scaled, normalization["target_mean"], normalization["target_std"])
