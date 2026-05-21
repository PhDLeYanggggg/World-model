from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from src.physics.collision import min_gap_and_collisions, project_collisions
from src.physics.constraints import project_scene_constraints
from src.physics.kinematics import clip_vectors, integrate_state
from src.physics.social_force import compute_social_force_acceleration
from src.training.train_real_benchmark import LinearResidualModel, real_agent_features


REAL_MODEL_NAMES = [
    "constant_velocity_baseline",
    "hand_physics_baseline",
    "deterministic_neural_residual",
    "stochastic_neural_residual",
    "physics_plus_neural_residual",
    "hand_physics_SMC",
    "physics_plus_neural_residual_SMC",
]


def evaluate_real_benchmark(episodes: List[Dict], model_bundle: Dict, output_dir: str | Path, quick: bool = False) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test = [e for e in episodes if e["meta"]["split"] == "real_test"]
    horizons = [1, 10, 25, 50, 100]
    particles = 16 if quick else 64
    results_by_model = {name: [] for name in REAL_MODEL_NAMES}
    if not test:
        metrics = empty_metrics("No real_test episodes were built.")
        write_metric_outputs(output_dir, metrics)
        return metrics
    for episode in test[: (3 if quick else len(test))]:
        horizon = min(100, episode["states"].shape[0] - 6)
        if horizon <= 1:
            continue
        rollouts = rollout_all_real_models(episode, model_bundle, horizon=horizon, particles=particles)
        true_future = episode["states"][5 : 5 + horizon + 1]
        for name, payload in rollouts.items():
            results_by_model[name].append(real_metrics(payload["trajectories"], true_future, episode["scene"], horizons, payload["weights"]))
    summary = {name: aggregate(rows, horizons) for name, rows in results_by_model.items()}
    write_metric_outputs(output_dir, summary)
    return summary


def rollout_all_real_models(episode: Dict, model_bundle: Dict, horizon: int, particles: int) -> Dict:
    states = episode["states"]
    scene = episode["scene"]
    history = states[:6]
    det = model_bundle["deterministic"]
    sto = model_bundle["stochastic"]
    return {
        "constant_velocity_baseline": one_payload(constant_velocity(history, scene, horizon)),
        "hand_physics_baseline": one_payload(hand_physics(history, scene, horizon)),
        "deterministic_neural_residual": one_payload(learned_residual(history, scene, horizon, det, stochastic=False)),
        "stochastic_neural_residual": multi_payload([learned_residual(history, scene, horizon, sto, stochastic=True, seed=100 + i) for i in range(min(8, particles))]),
        "physics_plus_neural_residual": one_payload(learned_residual(history, scene, horizon, det, stochastic=False)),
        "hand_physics_SMC": multi_payload([hand_physics(history, scene, horizon, noise=0.12, seed=200 + i) for i in range(particles)]),
        "physics_plus_neural_residual_SMC": multi_payload([learned_residual(history, scene, horizon, sto, stochastic=True, noise=0.08, seed=300 + i) for i in range(particles)]),
    }


def one_payload(traj: np.ndarray) -> Dict:
    return {"trajectories": traj[None], "weights": np.ones(1)}


def multi_payload(trajs: List[np.ndarray]) -> Dict:
    arr = np.stack(trajs, axis=0)
    return {"trajectories": arr, "weights": np.ones(arr.shape[0]) / arr.shape[0]}


def constant_velocity(history: np.ndarray, scene, horizon: int) -> np.ndarray:
    out = np.zeros((horizon + 1, history.shape[1], history.shape[2]), dtype=np.float32)
    current = history[-1].copy()
    out[0] = current
    dt = 1.0
    for h in range(1, horizon + 1):
        current = current.copy()
        current[:, :2] += current[:, 2:4] * dt
        current, _ = project_collisions(current)
        current, _ = project_scene_constraints(current, scene)
        out[h] = current
    return out


def hand_physics(history: np.ndarray, scene, horizon: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((horizon + 1, history.shape[1], history.shape[2]), dtype=np.float32)
    current = history[-1].copy()
    out[0] = current
    for h in range(1, horizon + 1):
        accel = compute_social_force_acceleration(current, scene, float(np.max(current[:, 9])))
        if noise:
            accel += rng.normal(0.0, noise, size=accel.shape).astype(np.float32)
        current = integrate_state(current, clip_vectors(accel, current[:, 9]), 1.0, float(np.max(current[:, 8])))
        current, _ = project_collisions(current)
        current, _ = project_scene_constraints(current, scene)
        out[h] = current
    return out


def learned_residual(history: np.ndarray, scene, horizon: int, model: LinearResidualModel, stochastic: bool, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((horizon + 1, history.shape[1], history.shape[2]), dtype=np.float32)
    current = history[-1].copy()
    out[0] = current
    for h in range(1, horizon + 1):
        base = compute_social_force_acceleration(current, scene, float(np.max(current[:, 9])))
        if model.trained:
            feats = np.stack([real_agent_features(current, scene, i) for i in range(current.shape[0])], axis=0)
            residual = model.predict(feats)
        else:
            residual = np.zeros_like(base)
        if stochastic:
            residual += rng.normal(0.0, model.residual_std, size=residual.shape).astype(np.float32)
        if noise:
            residual += rng.normal(0.0, noise, size=residual.shape).astype(np.float32)
        accel = clip_vectors(base + residual, current[:, 9])
        current = integrate_state(current, accel, 1.0, float(np.max(current[:, 8])))
        current, _ = project_collisions(current)
        current, _ = project_scene_constraints(current, scene)
        out[h] = current
    return out


def real_metrics(trajectories: np.ndarray, truth: np.ndarray, scene, horizons: Iterable[int], weights: np.ndarray) -> Dict:
    weights = weights / max(1e-12, weights.sum())
    mean_traj = np.tensordot(weights, trajectories, axes=(0, 0))
    max_h = truth.shape[0] - 1
    out = {"branch_count": int(trajectories.shape[0]), "horizons": {}}
    for horizon in horizons:
        if horizon > max_h:
            continue
        err = np.linalg.norm(mean_traj[1 : horizon + 1, :, :2] - truth[1 : horizon + 1, :, :2], axis=2)
        fde = np.linalg.norm(mean_traj[horizon, :, :2] - truth[horizon, :, :2], axis=1)
        branch_ade = np.mean(np.linalg.norm(trajectories[:, 1 : horizon + 1, :, :2] - truth[None, 1 : horizon + 1, :, :2], axis=3), axis=(1, 2))
        branch_fde = np.mean(np.linalg.norm(trajectories[:, horizon, :, :2] - truth[None, horizon, :, :2], axis=2), axis=1)
        out["horizons"][str(horizon)] = {
            "ADE": float(np.mean(err)),
            "FDE": float(np.mean(fde)),
            f"minADE@{trajectories.shape[0]}": float(np.min(branch_ade)),
            f"minFDE@{trajectories.shape[0]}": float(np.min(branch_fde)),
        }
    h_eval = max([h for h in horizons if h <= max_h], default=max_h)
    branch_fde = np.mean(np.linalg.norm(trajectories[:, h_eval, :, :2] - truth[None, h_eval, :, :2], axis=2), axis=1)
    for threshold in [1, 2, 5, 10]:
        out[f"coverage_FDE_lt_{threshold}m"] = float(np.mean(branch_fde < threshold))
    out.update(physical_metrics(trajectories, scene))
    out["cluster_diversity_score"] = endpoint_diversity(trajectories)
    out["semantic_event_accuracy"] = None
    out["NLL_endpoint"] = endpoint_nll(trajectories[:, h_eval, :, :2], weights, truth[h_eval, :, :2])
    return out


def physical_metrics(trajectories: np.ndarray, scene) -> Dict:
    frames = trajectories.shape[0] * trajectories.shape[1]
    collision = 0
    boundary = 0
    for particle in trajectories:
        for frame in particle:
            min_gap, collisions = min_gap_and_collisions(frame)
            if collisions or min_gap < -1e-4:
                collision += 1
            if np.any(frame[:, 0] < 0) or np.any(frame[:, 0] > scene.width) or np.any(frame[:, 1] < 0) or np.any(frame[:, 1] > scene.height):
                boundary += 1
    collision_rate = collision / max(1, frames)
    boundary_rate = boundary / max(1, frames)
    return {
        "collision_violation_rate": collision_rate,
        "boundary_violation_rate": boundary_rate,
        "obstacle_violation_rate": None,
        "physical_validity_rate": max(0.0, 1.0 - max(collision_rate, boundary_rate)),
    }


def endpoint_diversity(trajectories: np.ndarray) -> float:
    if trajectories.shape[0] <= 1:
        return 0.0
    endpoints = trajectories[:, -1, :, :2].mean(axis=1)
    spread = float(np.mean(np.linalg.norm(endpoints - endpoints.mean(axis=0), axis=1)))
    return min(1.0, spread / 5.0)


def endpoint_nll(endpoints: np.ndarray, weights: np.ndarray, truth: np.ndarray, sigma: float = 1.5) -> float:
    values = []
    for i in range(truth.shape[0]):
        d2 = np.sum((endpoints[:, i, :] - truth[i]) ** 2, axis=1)
        comp = np.log(np.maximum(weights, 1e-12)) - math.log(2 * math.pi * sigma * sigma) - 0.5 * d2 / (sigma * sigma)
        peak = np.max(comp)
        values.append(peak + math.log(float(np.exp(comp - peak).sum())))
    return float(-np.mean(values))


def aggregate(rows: List[Dict], horizons: Iterable[int]) -> Dict:
    if not rows:
        return {"available": False}
    out = {"available": True, "branch_count": int(np.mean([r["branch_count"] for r in rows])), "horizons": {}}
    for horizon in horizons:
        key = str(horizon)
        if key not in rows[0]["horizons"]:
            continue
        metrics = rows[0]["horizons"][key].keys()
        out["horizons"][key] = {m: round(float(np.mean([r["horizons"][key][m] for r in rows])), 5) for m in metrics}
    scalar_keys = ["coverage_FDE_lt_1m", "coverage_FDE_lt_2m", "coverage_FDE_lt_5m", "coverage_FDE_lt_10m", "collision_violation_rate", "boundary_violation_rate", "physical_validity_rate", "cluster_diversity_score", "NLL_endpoint"]
    for key in scalar_keys:
        out[key] = round(float(np.mean([r[key] for r in rows if isinstance(r.get(key), (int, float))])), 5)
    out["obstacle_violation_rate"] = None
    out["semantic_event_accuracy"] = None
    return out


def empty_metrics(reason: str) -> Dict:
    return {"error": reason, "models": {name: {"available": False} for name in REAL_MODEL_NAMES}}


def write_metric_outputs(output_dir: Path, metrics: Dict) -> None:
    (output_dir / "metrics_stage4_real.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rows = flatten_metrics(metrics)
    if rows:
        with (output_dir / "metrics_stage4_real.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / "metrics_table_stage4_real.md").write_text(markdown_table(rows), encoding="utf-8")
    else:
        (output_dir / "metrics_stage4_real.csv").write_text("", encoding="utf-8")
        (output_dir / "metrics_table_stage4_real.md").write_text("_No real benchmark metrics available._\n", encoding="utf-8")


def flatten_metrics(metrics: Dict) -> List[Dict]:
    rows = []
    if "models" in metrics:
        return rows
    for model, payload in metrics.items():
        if not payload.get("available"):
            continue
        row = {"model": model, "branch_count": payload["branch_count"]}
        for h in [1, 10, 25, 50, 100]:
            if str(h) in payload["horizons"]:
                row[f"ADE@{h}"] = payload["horizons"][str(h)]["ADE"]
                row[f"FDE@{h}"] = payload["horizons"][str(h)]["FDE"]
                minfde_key = [k for k in payload["horizons"][str(h)] if k.startswith("minFDE@")][0]
                row[f"minFDE@N@{h}"] = payload["horizons"][str(h)][minfde_key]
        for key in ["coverage_FDE_lt_1m", "coverage_FDE_lt_2m", "coverage_FDE_lt_5m", "coverage_FDE_lt_10m", "collision_violation_rate", "boundary_violation_rate", "physical_validity_rate", "cluster_diversity_score", "NLL_endpoint"]:
            row[key] = payload.get(key)
        rows.append(row)
    return rows


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._"
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(round(row.get(k), 5)) if isinstance(row.get(k), float) else str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)
