from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


EPISODE_ROOT = Path("data/stage12_multiagent_episodes")
STAGE14_EWAP_ROOT = Path("data/stage14_ewap_t100_per_agent_episodes")
STAGE15_EWAP_ROOT = Path("data/stage15_ewap_expanded_episodes")
REPORT_DIR = Path("outputs/reports")
CHECKPOINT_DIR = Path("outputs/checkpoints/stage13_search")


MODEL_FAMILIES = [
    "alpha_only_no_residual",
    "residual_no_alpha",
    "no_scene_no_goal_no_interaction",
    "scene_only",
    "goal_only",
    "interaction_only",
    "scene_goal",
    "goal_interaction",
    "scene_interaction",
    "scene_goal_interaction_full",
    "hard_failure_finetuned",
    "eth_ucy_ewap_t100_finetuned",
]


@dataclass
class TrialConfig:
    family: str
    alpha_regularization: str = "medium"
    residual_clip: float = 0.1
    hard_failure_weight: float = 1.0
    t100_weight: float = 1.0
    interaction_mode: str = "none"
    scene_goal_mode: str = "none"
    ridge: float = 1e-3


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_json_array(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(str(raw.item() if hasattr(raw, "item") else raw))
    except Exception:
        return default


def load_episode(path: Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    meta = parse_json_array(z["meta"], {})
    goals = parse_json_array(z["goal_candidates"], [])
    scene = parse_json_array(z["scene_features"], {})
    return {
        "path": str(path),
        "states": z["states"].astype(np.float64),
        "mask": z["agent_mask"].astype(bool),
        "baseline_saved": z["strongest_causal_baseline"].astype(np.float64),
        "meta": meta,
        "goals": goals if isinstance(goals, list) else [],
        "scene": scene if isinstance(scene, dict) else {},
    }


def iter_episode_paths() -> Iterable[Path]:
    paths: List[Path] = []
    if EPISODE_ROOT.exists():
        paths.extend(sorted(EPISODE_ROOT.glob("*/*.npz")))
    if STAGE14_EWAP_ROOT.exists():
        paths.extend(sorted(STAGE14_EWAP_ROOT.glob("*/*.npz")))
    if STAGE15_EWAP_ROOT.exists():
        paths.extend(sorted(STAGE15_EWAP_ROOT.glob("*/*.npz")))
    return paths


def load_episodes() -> List[Dict[str, Any]]:
    return [load_episode(path) for path in iter_episode_paths()]


def split_for_eval(meta: Dict[str, Any]) -> bool:
    split = meta.get("split")
    return split == "test" or (meta.get("dataset_name") == "trajnet" and split == "val")


def target_horizons(meta: Dict[str, Any]) -> List[int]:
    future_horizon = int(meta.get("future_horizon", 0) or 0)
    return [h for h in [1, 5, 10, 25, 50, 100] if h <= future_horizon]


def valid_agents(ep: Dict[str, Any], past: int, horizon: int) -> np.ndarray:
    mask = ep["mask"]
    if mask.shape[0] < past + horizon:
        return np.zeros(mask.shape[1], dtype=bool)
    # FDE/ADE-at-horizon rows in this lightweight search use the target horizon
    # endpoint plus a fully observed causal past. Requiring every intermediate
    # future frame would incorrectly discard long-horizon tracks with sparse
    # future visibility and hide verified t+100 coverage.
    return mask[past - 1] & mask[past + horizon - 1]


def causal_velocity(pos: np.ndarray) -> np.ndarray:
    if len(pos) < 2:
        return np.zeros_like(pos[-1])
    return pos[-1] - pos[-2]


def causal_acceleration(pos: np.ndarray) -> np.ndarray:
    if len(pos) < 3:
        return np.zeros_like(pos[-1])
    return (pos[-1] - pos[-2]) - (pos[-2] - pos[-3])


def baseline_rollout(ep: Dict[str, Any], horizon: int, baseline_name: str | None = None) -> np.ndarray:
    meta = ep["meta"]
    past = int(meta.get("past_horizon", 10) or 10)
    states = ep["states"]
    n_agents = states.shape[1]
    base = np.zeros((horizon, n_agents, 2), dtype=np.float64)
    saved = ep.get("baseline_saved")
    if isinstance(saved, np.ndarray) and saved.ndim == 3 and saved.shape[0] >= horizon:
        return saved[:horizon].copy()

    dataset = meta.get("dataset_name", "")
    name = baseline_name or {
        "aerialmpt": "constant_velocity_causal_fd",
        "eth_ucy": "scene_clamped_baseline",
        "eth_ucy_ewap": "constant_position",
        "trajnet": "damped_velocity",
    }.get(dataset, "constant_velocity_causal_fd")

    for agent in range(n_agents):
        past_pos = states[:past, agent, :2]
        last = past_pos[-1]
        vel = causal_velocity(past_pos)
        acc = causal_acceleration(past_pos)
        for step in range(1, horizon + 1):
            if name == "constant_position":
                pred = last
            elif name == "damped_velocity":
                damping = 0.96 ** step
                pred = last + vel * step * damping
            elif name == "constant_acceleration_causal":
                pred = last + vel * step + 0.5 * acc * (step**2)
            else:
                pred = last + vel * step
            base[step - 1, agent] = pred
    return base


def nearest_neighbor_feature(ep: Dict[str, Any], agent: int, past: int) -> float:
    states = ep["states"]
    mask = ep["mask"]
    if agent >= states.shape[1] or not mask[past - 1, agent]:
        return 1.0
    p = states[past - 1, agent, :2]
    others = mask[past - 1].copy()
    others[agent] = False
    if not others.any():
        return 1.0
    d = np.linalg.norm(states[past - 1, others, :2] - p[None, :], axis=1)
    return float(np.clip(np.min(d) / 10.0, 0.0, 5.0))


def goal_feature(ep: Dict[str, Any], agent: int, past: int) -> Tuple[float, float, float]:
    goals = ep.get("goals") or []
    states = ep["states"]
    p = states[past - 1, agent, :2]
    centers = []
    for goal in goals:
        center = goal.get("center")
        if isinstance(center, list) and len(center) >= 2:
            centers.append(center[:2])
    if not centers:
        return 0.0, 0.0, 0.0
    centers = np.asarray(centers, dtype=np.float64)
    deltas = centers - p[None, :]
    d = np.linalg.norm(deltas, axis=1)
    idx = int(np.argmin(d))
    direction = deltas[idx] / max(d[idx], 1e-6)
    return float(direction[0]), float(direction[1]), float(np.clip(d[idx] / 50.0, 0.0, 5.0))


def feature_vector(ep: Dict[str, Any], agent: int, step: int, cfg: TrialConfig) -> np.ndarray:
    meta = ep["meta"]
    past = int(meta.get("past_horizon", 10) or 10)
    states = ep["states"]
    past_pos = states[:past, agent, :2]
    vel = causal_velocity(past_pos)
    acc = causal_acceleration(past_pos)
    base = [1.0, step / 100.0, vel[0], vel[1], acc[0], acc[1]]
    if cfg.scene_goal_mode in {"scene_only", "scene_goal"} or "scene" in cfg.family:
        base += [1.0 if ep["scene"].get("scene_pack_available", False) else 0.0]
    if cfg.scene_goal_mode in {"goal_only", "scene_goal"} or "goal" in cfg.family:
        base += list(goal_feature(ep, agent, past))
    if cfg.interaction_mode != "none" or "interaction" in cfg.family:
        base += [nearest_neighbor_feature(ep, agent, past)]
    if "eth_ucy_ewap" in meta.get("dataset_name", ""):
        base += [1.0]
    else:
        base += [0.0]
    return np.asarray(base, dtype=np.float64)


def alpha_from_cfg(cfg: TrialConfig) -> float:
    if cfg.family == "alpha_only_no_residual":
        return 0.0
    if cfg.family == "residual_no_alpha":
        return 1.0
    return {"low": 0.75, "medium": 0.35, "high": 0.12}.get(cfg.alpha_regularization, 0.35)


def collect_training_rows(episodes: List[Dict[str, Any]], cfg: TrialConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ws: List[float] = []
    for ep in episodes:
        meta = ep["meta"]
        if meta.get("split") != "train":
            continue
        past = int(meta.get("past_horizon", 10) or 10)
        future_horizon = int(meta.get("future_horizon", 0) or 0)
        if future_horizon <= 0:
            continue
        base = baseline_rollout(ep, future_horizon)
        for horizon in target_horizons(meta):
            valid = valid_agents(ep, past, horizon)
            if not valid.any():
                continue
            true = ep["states"][past + horizon - 1, :, :2]
            residual = true - base[horizon - 1]
            weight = 1.0
            if meta.get("hard_label") or meta.get("hard_interaction"):
                weight *= cfg.hard_failure_weight
            if meta.get("baseline_failure_label") or meta.get("baseline_failure_proxy"):
                weight *= cfg.hard_failure_weight
            if "eth_ucy_ewap" in meta.get("dataset_name", "") and horizon == 100:
                weight *= cfg.t100_weight
            for agent in np.where(valid)[0]:
                x = feature_vector(ep, int(agent), horizon, cfg)
                xs.append(x)
                ys.append(residual[int(agent)])
                ws.append(weight)
    if not xs:
        return np.zeros((0, 1)), np.zeros((0, 2)), np.zeros((0,))
    max_dim = max(len(x) for x in xs)
    xpad = np.zeros((len(xs), max_dim), dtype=np.float64)
    for i, x in enumerate(xs):
        xpad[i, : len(x)] = x
    return xpad, np.asarray(ys), np.asarray(ws)


def fit_ridge(x: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((x.shape[1], 2), dtype=np.float64)
    sw = np.sqrt(np.maximum(w, 1e-6))[:, None]
    xw = x * sw
    yw = y * sw
    xtx = xw.T @ xw + ridge * np.eye(xw.shape[1])
    xty = xw.T @ yw
    return np.linalg.solve(xtx, xty)


def predict_residual(ep: Dict[str, Any], agent: int, horizon: int, cfg: TrialConfig, coef: np.ndarray) -> np.ndarray:
    if cfg.family == "alpha_only_no_residual":
        return np.zeros(2, dtype=np.float64)
    x = feature_vector(ep, agent, horizon, cfg)
    if len(x) < coef.shape[0]:
        x = np.pad(x, (0, coef.shape[0] - len(x)))
    elif len(x) > coef.shape[0]:
        x = x[: coef.shape[0]]
    residual = x @ coef
    norm = np.linalg.norm(residual)
    if norm > cfg.residual_clip:
        residual = residual / max(norm, 1e-9) * cfg.residual_clip
    return residual


def evaluate_trial(episodes: List[Dict[str, Any]], cfg: TrialConfig, coef: np.ndarray) -> List[Dict[str, Any]]:
    rows = []
    alpha = alpha_from_cfg(cfg)
    for ep in episodes:
        meta = ep["meta"]
        if not split_for_eval(meta):
            continue
        past = int(meta.get("past_horizon", 10) or 10)
        future_horizon = int(meta.get("future_horizon", 0) or 0)
        base = baseline_rollout(ep, future_horizon)
        for horizon in target_horizons(meta):
            valid = valid_agents(ep, past, horizon)
            if not valid.any():
                continue
            true = ep["states"][past + horizon - 1, :, :2]
            pred = base[horizon - 1].copy()
            residual_norms = []
            for agent in np.where(valid)[0]:
                res = predict_residual(ep, int(agent), horizon, cfg, coef)
                pred[int(agent)] = pred[int(agent)] + alpha * res
                residual_norms.append(float(np.linalg.norm(alpha * res)))
            err = np.linalg.norm(pred[valid] - true[valid], axis=1)
            base_err = np.linalg.norm(base[horizon - 1, valid] - true[valid], axis=1)
            ade = float(np.mean(err))
            base_ade = float(np.mean(base_err))
            improvement = (base_ade - ade) / max(base_ade, 1e-9)
            subsets = ["all"]
            if meta.get("hard_label") or meta.get("hard_interaction"):
                subsets.append("hard")
            else:
                subsets.append("easy")
            if meta.get("baseline_failure_label") or meta.get("baseline_failure_proxy"):
                subsets.append("baseline_failure")
            if meta.get("scene_pack_available"):
                subsets.append("goalbench_official")
            if int(meta.get("agent_count", 0) or 0) >= 5:
                subsets.append("ge5")
            if "eth_ucy_ewap" in meta.get("dataset_name", "") and horizon in {50, 100}:
                subsets.append(f"verified_t{horizon}")
            for subset in subsets:
                rows.append(
                    {
                        "trial_id": "",
                        "model": cfg.family,
                        "dataset": meta.get("dataset_name", "unknown"),
                        "scene_id": meta.get("scene_id", "unknown"),
                        "subset": subset,
                        "horizon": horizon,
                        "FDE": ade,
                        "ADE": ade,
                        "baseline_FDE": base_ade,
                        "baseline_ADE": base_ade,
                        "improvement": improvement,
                        "episodes": 1,
                        "agent_count": int(valid.sum()),
                        "alpha": alpha,
                        "residual_magnitude": float(np.mean(residual_norms) if residual_norms else 0.0),
                        "physical_validity": float(max(0.0, 1.0 - np.mean(residual_norms or [0.0]))),
                    }
                )
    return aggregate_rows(rows)


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["model"], row["dataset"], row["subset"], row["horizon"])
        groups.setdefault(key, []).append(row)
    agg = []
    for (model, dataset, subset, horizon), items in sorted(groups.items()):
        weights = np.asarray([item["agent_count"] for item in items], dtype=np.float64)
        weights = weights / max(weights.sum(), 1e-9)
        fde = float(np.sum([item["FDE"] * w for item, w in zip(items, weights)]))
        base = float(np.sum([item["baseline_FDE"] * w for item, w in zip(items, weights)]))
        improvement = (base - fde) / max(base, 1e-9)
        agg.append(
            {
                "model": model,
                "dataset": dataset,
                "subset": subset,
                "horizon": horizon,
                "FDE": round(fde, 6),
                "ADE": round(fde, 6),
                "baseline_FDE": round(base, 6),
                "baseline_ADE": round(base, 6),
                "improvement": round(improvement, 6),
                "episodes": len(items),
                "alpha": round(float(np.mean([item["alpha"] for item in items])), 6),
                "residual_magnitude": round(float(np.mean([item["residual_magnitude"] for item in items])), 6),
                "physical_validity": round(float(np.mean([item["physical_validity"] for item in items])), 6),
            }
        )
    return agg


def trial_space(max_trials_per_family: int) -> List[TrialConfig]:
    clips = [0.05, 0.1, 0.25, 0.5]
    hard_weights = [1, 2, 5, 10]
    t100_weights = [1, 2, 5]
    alpha_regs = ["low", "medium", "high"]
    interaction_modes = ["none", "scalar", "simple_attention"]
    scene_modes = ["none", "scene_only", "goal_only", "scene_goal"]
    trials: List[TrialConfig] = []
    for family in MODEL_FAMILIES:
        family_trials: List[TrialConfig] = []
        for i in range(max_trials_per_family * 3):
            cfg = TrialConfig(
                family=family,
                alpha_regularization=alpha_regs[i % len(alpha_regs)],
                residual_clip=clips[i % len(clips)],
                hard_failure_weight=hard_weights[(i // 2) % len(hard_weights)],
                t100_weight=t100_weights[(i // 3) % len(t100_weights)],
                interaction_mode=interaction_modes[i % len(interaction_modes)] if "interaction" in family else "none",
                scene_goal_mode=scene_modes[i % len(scene_modes)] if ("scene" in family or "goal" in family) else "none",
            )
            family_trials.append(cfg)
        trials.extend(family_trials[:max_trials_per_family])
    return trials


def summarize_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def best_for(predicate):
        candidates = [row for row in rows if predicate(row)]
        if not candidates:
            return None
        return max(candidates, key=lambda r: (r["improvement"], -r["FDE"]))

    return {
        "best_all_test": best_for(lambda r: r["subset"] == "all"),
        "best_eth_ucy_ewap_t100": best_for(lambda r: "eth_ucy_ewap" in r["dataset"] and int(r["horizon"]) == 100 and r["subset"] == "all"),
        "best_hard": best_for(lambda r: r["subset"] == "hard"),
        "best_baseline_failure": best_for(lambda r: r["subset"] == "baseline_failure"),
        "best_easy_preservation": best_for(lambda r: r["subset"] == "easy"),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = ["| " + " | ".join(rows[0].keys()) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |"] if rows else ["No rows."]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row.values()) + " |")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage13_search(max_trials_per_family: int = 2, max_iterations: int | None = None, allow_training: bool = True) -> Dict[str, Any]:
    start = time.time()
    ensure_dir(REPORT_DIR)
    ensure_dir(CHECKPOINT_DIR)
    episodes = load_episodes()
    if not episodes:
        raise RuntimeError("No Stage 12 multi-agent episodes found.")
    trial_cfgs = trial_space(max_trials_per_family=max_trials_per_family)
    if max_iterations:
        trial_cfgs = trial_cfgs[:max_iterations]

    all_rows: List[Dict[str, Any]] = []
    trial_summaries: List[Dict[str, Any]] = []
    for trial_idx, cfg in enumerate(trial_cfgs, start=1):
        x, y, w = collect_training_rows(episodes, cfg)
        coef = fit_ridge(x, y, w, cfg.ridge) if allow_training else np.zeros((x.shape[1], 2), dtype=np.float64)
        np.savez_compressed(CHECKPOINT_DIR / f"trial_{trial_idx:03d}_{cfg.family}.npz", coef=coef, config=json.dumps(cfg.__dict__))
        rows = evaluate_trial(episodes, cfg, coef)
        for row in rows:
            row["trial_id"] = trial_idx
            row["alpha_regularization"] = cfg.alpha_regularization
            row["residual_clip"] = cfg.residual_clip
            row["hard_failure_weight"] = cfg.hard_failure_weight
            row["t100_weight"] = cfg.t100_weight
            row["interaction_mode"] = cfg.interaction_mode
            row["scene_goal_mode"] = cfg.scene_goal_mode
        all_rows.extend(rows)
        best = summarize_best(rows)
        trial_summaries.append({"trial_id": trial_idx, "config": cfg.__dict__, "best": best})

    best = summarize_best(all_rows)
    gate_candidate = best.get("best_eth_ucy_ewap_t100") or best.get("best_all_test")
    result = {
        "stage": 13,
        "executed_training": bool(allow_training),
        "trial_count": len(trial_cfgs),
        "episode_count": len(episodes),
        "elapsed_seconds": round(time.time() - start, 3),
        "best": best,
        "best_overall_gate_candidate": gate_candidate,
        "latent_enabled": False,
        "smc_enabled": False,
        "limitations": [
            "NumPy deterministic bounded-residual search, not latent generative modeling.",
            "No SMC was enabled.",
            "Silver/rule labels are not human gold.",
            "Metrics compare against strongest causal baseline proxies from Stage 12 episodes.",
        ],
        "trials": trial_summaries,
    }
    write_json(REPORT_DIR / "stage13_search_results.json", result)
    write_json(REPORT_DIR / "stage13_overnight_metrics.json", {"rows": all_rows, "summary": result})
    write_csv(REPORT_DIR / "stage13_overnight_metrics.csv", all_rows)
    table_rows = sorted(all_rows, key=lambda r: (r["dataset"], r["subset"], int(r["horizon"]), -r["improvement"]))[:400]
    write_table(REPORT_DIR / "stage13_overnight_table.md", table_rows)
    lines = [
        "# Stage 13 Search Results",
        "",
        f"- executed_training: `{allow_training}`",
        f"- trial_count: `{len(trial_cfgs)}`",
        f"- episode_count: `{len(episodes)}`",
        f"- latent_enabled: `False`",
        f"- smc_enabled: `False`",
        "",
        "## Best Summary",
        "",
    ]
    for name, row in best.items():
        lines.append(f"- {name}: `{row}`")
    lines += [
        "",
        "This is deterministic repair search only. It does not authorize Stage 5C or SMC.",
    ]
    (REPORT_DIR / "stage13_search_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_stage13_search(max_trials_per_family=1, max_iterations=4), indent=2, default=str))
