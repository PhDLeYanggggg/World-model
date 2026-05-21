from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


FEATURES = ["x", "y", "vx", "vy", "ax", "ay", "heading", "speed", "body_radius"]
BASELINES = [
    "constant_position",
    "constant_velocity_causal_fd",
    "damped_velocity",
    "constant_acceleration_causal",
    "constant_turn_rate_velocity",
    "identity_hand_physics",
    "type_specific_kinematic_baseline",
]


def available_datasets(root: str | Path = "data/stage5b_episodes") -> List[str]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(path.name for path in base.iterdir() if path.is_dir() and list(path.glob("episode_*.npz")))


def load_dataset_episodes(dataset: str, split: str = "test", root: str | Path = "data/stage5b_episodes") -> List[Dict]:
    episodes = []
    for path in sorted((Path(root) / dataset).glob("episode_*.npz")):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"].item()))
        if split != "all" and meta.get("split") != split:
            continue
        episodes.append({"states": data["states"].astype(np.float32), "meta": meta, "path": str(path)})
    return episodes


def rollout(history: np.ndarray, horizon: int, dt: float, model: str, agent_type: str = "unknown") -> np.ndarray:
    last = history[-1].copy()
    out = np.zeros((horizon + 1, history.shape[1], history.shape[2]), dtype=np.float32)
    out[0] = last
    prev = history[-2] if history.shape[0] >= 2 else last
    speed = np.linalg.norm(last[:, 2:4], axis=1)
    theta = last[:, 6].copy()
    dtheta = np.angle(np.exp(1j * (last[:, 6] - prev[:, 6]))) if history.shape[0] >= 2 else np.zeros(history.shape[1], dtype=np.float32)
    current = last.copy()
    for t in range(1, horizon + 1):
        current = current.copy()
        if model == "constant_position":
            current[:, 2:6] = 0.0
        elif model in {"constant_velocity_causal_fd", "identity_hand_physics"}:
            current[:, 0:2] += current[:, 2:4] * dt
        elif model == "damped_velocity":
            current[:, 2:4] *= 0.985
            current[:, 0:2] += current[:, 2:4] * dt
        elif model == "constant_acceleration_causal":
            current[:, 0:2] += current[:, 2:4] * dt + 0.5 * current[:, 4:6] * dt * dt
            current[:, 2:4] += current[:, 4:6] * dt
        elif model == "constant_turn_rate_velocity":
            theta = theta + dtheta
            current[:, 2] = speed * np.cos(theta)
            current[:, 3] = speed * np.sin(theta)
            current[:, 0:2] += current[:, 2:4] * dt
            current[:, 6] = theta
        elif model == "type_specific_kinematic_baseline":
            if "vehicle" in agent_type or "traffic" in agent_type:
                theta = theta + dtheta
                current[:, 2] = speed * np.cos(theta)
                current[:, 3] = speed * np.sin(theta)
                current[:, 0:2] += current[:, 2:4] * dt
                current[:, 6] = theta
            else:
                current[:, 0:2] += current[:, 2:4] * dt
        else:
            raise ValueError(model)
        current[:, 6] = np.arctan2(current[:, 3], current[:, 2])
        current[:, 7] = np.linalg.norm(current[:, 2:4], axis=1)
        out[t] = current
    return out


def horizons_for_episodes(episodes: List[Dict]) -> List[int]:
    if not episodes:
        return []
    future = min(ep["states"].shape[0] - int(ep["meta"].get("past_horizon", 10)) for ep in episodes)
    return [h for h in [1, 10, 25, 50, 100] if h <= future]


def evaluate_model(dataset: str, model: str, episodes: List[Dict]) -> Dict:
    horizons = horizons_for_episodes(episodes)
    by_h = {}
    all_speed_violations = []
    all_accel_violations = []
    all_collision_violations = []
    for h in horizons:
        ade_values = []
        fde_values = []
        for ep in episodes:
            states = ep["states"]
            past = int(ep["meta"].get("past_horizon", 10))
            dt = float(ep["meta"].get("dt_s", 1.0))
            agent_type = str(ep["meta"].get("agent_type", dataset))
            hist = states[:past]
            true = states[past : past + h]
            pred = rollout(hist, h, dt, model, agent_type=agent_type)[1 : h + 1]
            err = np.linalg.norm(pred[:, :, 0:2] - true[:, :, 0:2], axis=2)
            ade_values.append(float(err.mean()))
            fde_values.append(float(err[-1].mean()))
            speed = np.linalg.norm(pred[:, :, 2:4], axis=2)
            accel = np.linalg.norm(np.diff(pred[:, :, 2:4], axis=0) / max(dt, 1e-6), axis=2) if pred.shape[0] > 1 else np.zeros_like(speed)
            true_speed = np.linalg.norm(true[:, :, 2:4], axis=2)
            speed_limit = max(1.0, float(np.nanpercentile(true_speed, 99.0)) * 2.0)
            accel_limit = max(1.0, float(np.nanpercentile(np.linalg.norm(true[:, :, 4:6], axis=2), 99.0)) * 3.0)
            all_speed_violations.append(float(np.mean(speed > speed_limit)))
            all_accel_violations.append(float(np.mean(accel > accel_limit)) if accel.size else 0.0)
            all_collision_violations.append(float(collision_violation_rate(pred)))
        by_h[str(h)] = {
            "ADE": round(float(np.mean(ade_values)), 6) if ade_values else math.nan,
            "FDE": round(float(np.mean(fde_values)), 6) if fde_values else math.nan,
        }
    physical_violation = float(np.mean(all_speed_violations + all_accel_violations + all_collision_violations)) if all_speed_violations else 0.0
    return {
        "dataset": dataset,
        "model": model,
        "episodes": len(episodes),
        "horizons": by_h,
        "physical_validity_rate": round(float(max(0.0, 1.0 - physical_violation)), 6),
        "collision_violation_rate": round(float(np.mean(all_collision_violations)), 6) if all_collision_violations else 0.0,
        "speed_violation_rate": round(float(np.mean(all_speed_violations)), 6) if all_speed_violations else 0.0,
        "acceleration_violation_rate": round(float(np.mean(all_accel_violations)), 6) if all_accel_violations else 0.0,
    }


def collision_violation_rate(pred: np.ndarray) -> float:
    if pred.shape[1] < 2:
        return 0.0
    violations = []
    for frame in pred:
        positions = frame[:, 0:2]
        radii = np.maximum(frame[:, 8], 0.2)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                gap = np.linalg.norm(positions[i] - positions[j]) - (radii[i] + radii[j])
                violations.append(gap < 0.0)
    return float(np.mean(violations)) if violations else 0.0


def benchmark_dataset(dataset: str, split: str = "test") -> Dict:
    episodes = load_dataset_episodes(dataset, split=split)
    if not episodes and split != "all":
        episodes = load_dataset_episodes(dataset, split="all")
    results = {model: evaluate_model(dataset, model, episodes) for model in BASELINES}
    horizons = horizons_for_episodes(episodes)
    target_h = str(max(horizons)) if horizons else "0"
    strongest = min(BASELINES, key=lambda model: results[model]["horizons"].get(target_h, {}).get("FDE", float("inf")))
    return {
        "dataset": dataset,
        "split": split,
        "episodes": len(episodes),
        "official_horizons": horizons,
        "target_horizon_for_strongest": int(target_h),
        "strongest_causal_baseline": strongest,
        "strongest_metrics": results[strongest],
        "all_baselines": results,
    }


def write_outputs(rows: Iterable[Dict]) -> Dict:
    payload = {"datasets": {row["dataset"]: row for row in rows}}
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage5b_baseline_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_lines = ["dataset,model,horizon,ADE,FDE,physical_validity_rate,collision_violation_rate,speed_violation_rate,acceleration_violation_rate"]
    md_rows = []
    strongest_rows = []
    for dataset, row in payload["datasets"].items():
        strongest = row["strongest_causal_baseline"]
        strongest_h = str(row["target_horizon_for_strongest"])
        strongest_metrics = row["all_baselines"][strongest]["horizons"][strongest_h]
        strongest_rows.append(
            {
                "dataset": dataset,
                "strongest_causal_baseline": strongest,
                f"FDE@{strongest_h}": strongest_metrics["FDE"],
                f"ADE@{strongest_h}": strongest_metrics["ADE"],
            }
        )
        for model, metrics in row["all_baselines"].items():
            for horizon, hv in metrics["horizons"].items():
                csv_lines.append(
                    f"{dataset},{model},{horizon},{hv['ADE']},{hv['FDE']},{metrics['physical_validity_rate']},"
                    f"{metrics['collision_violation_rate']},{metrics['speed_violation_rate']},{metrics['acceleration_violation_rate']}"
                )
            h100 = metrics["horizons"].get("100", {})
            md_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "ADE@10": metrics["horizons"].get("10", {}).get("ADE", ""),
                    "FDE@10": metrics["horizons"].get("10", {}).get("FDE", ""),
                    "ADE@100": h100.get("ADE", "n/a"),
                    "FDE@100": h100.get("FDE", "n/a"),
                    "physical_validity": metrics["physical_validity_rate"],
                }
            )
    (out / "stage5b_baseline_metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (out / "stage5b_baseline_table.md").write_text(markdown_table(md_rows), encoding="utf-8")
    (out / "stage5b_strongest_causal_baselines.md").write_text(markdown_table(strongest_rows), encoding="utf-8")
    return payload


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"
