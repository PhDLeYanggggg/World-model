from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.evaluation.stage9_data_audit import available_stage9_datasets, load_stage9_episodes


REPORT_DIR = Path("outputs/reports")
BASELINES = [
    "constant_position",
    "constant_velocity_causal_fd",
    "damped_velocity",
    "constant_acceleration_causal",
    "constant_turn_rate_velocity",
    "goal_directed_baseline",
    "scene_clamped_baseline",
    "nearest_goal_baseline_diagnostic",
]
OFFICIAL_BASELINES = [b for b in BASELINES if b != "nearest_goal_baseline_diagnostic"]


def rollout_baseline(ep: Dict, horizon: int, model: str) -> np.ndarray:
    states = ep["states"]
    meta = ep["meta"]
    past = int(meta["past_horizon"])
    dt = float(meta.get("dt_s", 1.0))
    hist = states[:past].copy()
    last = hist[-1].copy()
    prev = hist[-2].copy() if hist.shape[0] >= 2 else last.copy()
    out = np.zeros((horizon, states.shape[1], states.shape[2]), dtype=np.float32)
    cur = last.copy()
    speed = np.linalg.norm(cur[:, 2:4], axis=1)
    theta = cur[:, 6].copy()
    dtheta = np.angle(np.exp(1j * (cur[:, 6] - prev[:, 6]))) if hist.shape[0] >= 2 else np.zeros(cur.shape[0], dtype=float)
    goals = load_goal_centers(ep)
    boundary = load_boundary(ep)
    for t in range(horizon):
        cur = cur.copy()
        if model == "constant_position":
            cur[:, 2:6] = 0.0
        elif model == "constant_velocity_causal_fd":
            cur[:, 0:2] += cur[:, 2:4] * dt
        elif model == "damped_velocity":
            cur[:, 2:4] *= 0.985
            cur[:, 0:2] += cur[:, 2:4] * dt
        elif model == "constant_acceleration_causal":
            cur[:, 0:2] += cur[:, 2:4] * dt + 0.5 * cur[:, 4:6] * dt * dt
            cur[:, 2:4] += cur[:, 4:6] * dt
        elif model == "constant_turn_rate_velocity":
            theta = theta + dtheta
            cur[:, 2] = speed * np.cos(theta)
            cur[:, 3] = speed * np.sin(theta)
            cur[:, 0:2] += cur[:, 2:4] * dt
        elif model in {"goal_directed_baseline", "nearest_goal_baseline_diagnostic"}:
            cur[:, 2:4] = goal_directed_velocity(cur, goals, speed)
            cur[:, 0:2] += cur[:, 2:4] * dt
        elif model == "scene_clamped_baseline":
            cur[:, 0:2] += cur[:, 2:4] * dt
            if boundary is not None:
                cur[:, 0] = np.clip(cur[:, 0], boundary[0], boundary[2])
                cur[:, 1] = np.clip(cur[:, 1], boundary[1], boundary[3])
        else:
            raise ValueError(model)
        cur[:, 6] = np.arctan2(cur[:, 3], cur[:, 2])
        cur[:, 7] = np.linalg.norm(cur[:, 2:4], axis=1)
        out[t] = cur
    return out


def load_goal_centers(ep: Dict) -> np.ndarray:
    try:
        data = np.load(ep["path"], allow_pickle=True)
        goals = json.loads(str(data["goal_candidates"].item()))
        centers = [g.get("center", [0.0, 0.0]) for g in goals]
        return np.asarray(centers, dtype=float) if centers else np.zeros((0, 2), dtype=float)
    except Exception:  # noqa: BLE001
        return np.zeros((0, 2), dtype=float)


def load_boundary(ep: Dict) -> Tuple[float, float, float, float] | None:
    scene = Path("data/stage8p5_scene_gold_packs") / ep["meta"]["dataset_name"] / ep["meta"]["scene_id"] / "scene_gold_pack.json"
    if not scene.exists():
        return None
    pack = json.loads(scene.read_text(encoding="utf-8"))
    pts = np.asarray(pack.get("boundary_polygon", []), dtype=float)
    if len(pts) < 3:
        return None
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def goal_directed_velocity(cur: np.ndarray, goals: np.ndarray, speed: np.ndarray) -> np.ndarray:
    if goals.size == 0:
        return cur[:, 2:4]
    vec = goals[None, :, :] - cur[:, None, 0:2]
    dist = np.linalg.norm(vec, axis=2)
    nearest = np.argmin(dist, axis=1)
    direction = vec[np.arange(cur.shape[0]), nearest]
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    return direction / np.maximum(norm, 1e-6) * speed[:, None]


def valid_future_mask(ep: Dict, horizon: int) -> np.ndarray:
    past = int(ep["meta"]["past_horizon"])
    return ep["mask"][past : past + horizon]


def evaluate_baseline(dataset: str, model: str, episodes: List[Dict]) -> Dict:
    horizons = available_horizons(episodes)
    by_h = {}
    subset_errors = {"easy": [], "hard": [], "baseline_failure": []}
    for h in horizons:
        ade_vals = []
        fde_vals = []
        scene_fde_vals = []
        validity = []
        for ep in episodes:
            past = int(ep["meta"]["past_horizon"])
            true = ep["states"][past : past + h]
            mask = valid_future_mask(ep, h)
            pred = rollout_baseline(ep, h, model)
            ade, fde = masked_ade_fde(pred, true, mask)
            ade_vals.append(ade)
            fde_vals.append(fde)
            scene_fde_vals.append(scene_fde(pred, true, mask))
            validity.append(physical_validity(pred, mask))
            target_subset = "hard" if ep["meta"].get("hard_interaction") else "easy"
            subset_errors.setdefault(target_subset, []).append(fde)
            if ep["meta"].get("baseline_failure_proxy"):
                subset_errors["baseline_failure"].append(fde)
        by_h[str(h)] = {
            "per_agent_ADE": round(float(np.nanmean(ade_vals)), 6),
            "per_agent_FDE": round(float(np.nanmean(fde_vals)), 6),
            "all_agent_scene_FDE": round(float(np.nanmean(scene_fde_vals)), 6),
            "physical_validity": round(float(np.nanmean(validity)), 6),
        }
    target_h = str(max([int(h) for h in by_h], default=0))
    return {
        "dataset": dataset,
        "model": model,
        "episodes": len(episodes),
        "horizons": by_h,
        "target_horizon": int(target_h) if target_h != "0" else 0,
        "target_FDE": by_h.get(target_h, {}).get("per_agent_FDE", math.inf),
        "hard_subset_FDE": round(float(np.nanmean(subset_errors["hard"])), 6) if subset_errors["hard"] else None,
        "baseline_failure_subset_FDE": round(float(np.nanmean(subset_errors["baseline_failure"])), 6) if subset_errors["baseline_failure"] else None,
        "easy_subset_FDE": round(float(np.nanmean(subset_errors["easy"])), 6) if subset_errors["easy"] else None,
    }


def available_horizons(episodes: List[Dict]) -> List[int]:
    if not episodes:
        return []
    future = min(int(ep["meta"].get("future_horizon", 0)) for ep in episodes)
    return [h for h in [1, 5, 10, 25, 50, 100] if h <= future]


def masked_ade_fde(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    err = np.linalg.norm(pred[:, :, 0:2] - true[:, :, 0:2], axis=2)
    if not mask.any():
        return math.nan, math.nan
    ade = float(err[mask].mean())
    final_mask = mask[-1]
    fde = float(err[-1][final_mask].mean()) if final_mask.any() else math.nan
    return ade, fde


def scene_fde(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    final_mask = mask[-1]
    if not final_mask.any():
        return math.nan
    pred_center = pred[-1, final_mask, 0:2].mean(axis=0)
    true_center = true[-1, final_mask, 0:2].mean(axis=0)
    return float(np.linalg.norm(pred_center - true_center))


def physical_validity(pred: np.ndarray, mask: np.ndarray) -> float:
    if pred.shape[0] < 2:
        return 1.0
    speeds = np.linalg.norm(pred[:, :, 2:4], axis=2)
    valid_speeds = speeds[mask]
    if valid_speeds.size == 0:
        return 1.0
    limit = max(float(np.nanpercentile(valid_speeds, 99)) * 2.0, 1.0)
    speed_ok = float(np.mean(valid_speeds <= limit))
    collision_ok = 1.0 - collision_proxy(pred, mask)
    return float(0.5 * speed_ok + 0.5 * collision_ok)


def collision_proxy(pred: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for frame, m in zip(pred, mask):
        pos = frame[m, 0:2]
        if len(pos) < 2:
            continue
        d = np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=2)
        d[d == 0] = np.inf
        vals.append(float(np.min(d) < 0.5))
    return float(np.mean(vals)) if vals else 0.0


def run_stage9_baselines(datasets: List[str] | None = None) -> Dict:
    datasets = datasets or available_stage9_datasets()
    out = {"stage": "9", "datasets": {}}
    for dataset in datasets:
        episodes = load_stage9_episodes(dataset, split="test")
        if not episodes:
            episodes = load_stage9_episodes(dataset, split="val")
        results = {model: evaluate_baseline(dataset, model, episodes) for model in BASELINES}
        strongest = min(OFFICIAL_BASELINES, key=lambda m: results[m]["target_FDE"])
        out["datasets"][dataset] = {
            "episodes": len(episodes),
            "strongest_causal_baseline": strongest,
            "target_horizon": results[strongest]["target_horizon"],
            "strongest_metrics": results[strongest],
            "all_baselines": results,
        }
    return out


def write_stage9_baselines(payload: Dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage9_per_agent_baseline_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for dataset, drow in payload["datasets"].items():
        strongest = drow["strongest_causal_baseline"]
        for model, row in drow["all_baselines"].items():
            h = str(row["target_horizon"])
            hv = row["horizons"].get(h, {})
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "target_horizon": h,
                    "per_agent_FDE": hv.get("per_agent_FDE", ""),
                    "per_agent_ADE": hv.get("per_agent_ADE", ""),
                    "scene_FDE": hv.get("all_agent_scene_FDE", ""),
                    "hard_FDE": row.get("hard_subset_FDE"),
                    "failure_FDE": row.get("baseline_failure_subset_FDE"),
                    "easy_FDE": row.get("easy_subset_FDE"),
                    "strongest": model == strongest,
                }
            )
    (REPORT_DIR / "stage9_per_agent_baseline_table.md").write_text(markdown_table(rows), encoding="utf-8")


def markdown_table(rows: Iterable[Dict]) -> str:
    rows = list(rows)
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"
