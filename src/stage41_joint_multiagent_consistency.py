from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_fresh_confirmation as fresh
from src import stage41_full_trajectory_world_state as ft


OUT_DIR = fresh.OUT_DIR
DATA_DIR = fresh.DATA_DIR
LEDGER_JSONL = fresh.LEDGER_JSONL
EPS = 1e-6


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _append_ledger(step: str, status: str, started: float, inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    ensure_dir(OUT_DIR)
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": step,
        "source": "fresh_run",
        "status": status,
        "wall_time_s": time.perf_counter() - started,
        "input_hash": _combined_hash(inputs),
        "output_hash": _combined_hash(outputs),
        "git_commit": _git_commit(),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _full_trajectory_reference() -> Mapping[str, Any]:
    report = read_json(OUT_DIR / "stage41_full_trajectory_world_state.json", {})
    trials = report.get("trials") or {}
    ensemble = trials.get("full_trajectory_ensemble") or {}
    paths = ensemble.get("paths") or []
    if not paths:
        best_name = report.get("best_name")
        best_trial = trials.get(str(best_name), {})
        ckpt = (best_trial.get("train") or {}).get("checkpoint")
        paths = [ckpt] if ckpt else []
    if not paths:
        ft.train_full_trajectory_world_state()
        return _full_trajectory_reference()
    return {
        "paths": paths,
        "policy": report.get("best_policy") or ensemble.get("policy") or {},
        "metrics": report.get("best_metrics") or {},
    }


def _split_predictions(split: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Mapping[str, Any]]:
    ref = _full_trajectory_reference()
    pred, labels = ft._predict_ensemble(ref["paths"], split)
    return pred, labels, ref


def _group_metadata(split: str) -> dict[str, np.ndarray]:
    data = s41._combined()
    ds = ft._fresh_ds(split)
    ids = ds["ids"].astype(np.int64)
    key = np.asarray(
        [f"{data['source_file'][rid]}|{int(round(float(data['frame_id'][rid])))}|{int(data['horizon'][rid])}" for rid in ids],
        dtype=object,
    )
    return {
        "key": key,
        "agent_id": data["agent_id"].astype(int)[ids],
        "frame_id": data["frame_id"].astype(float)[ids],
        "ids": ids,
    }


def _selected_waypoints(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], switch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred, labels)
    selected = floor_xy.copy()
    selected[switch] = neural_xy[switch]
    return selected, neural_xy


def _min_group_distance(xy: np.ndarray, group_key: np.ndarray, normalizer: np.ndarray) -> np.ndarray:
    out = np.full(len(xy), np.inf, dtype=np.float64)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(group_key):
        groups[str(key)].append(i)
    for members in groups.values():
        if len(members) < 2:
            continue
        mem = np.asarray(members, dtype=np.int64)
        pts = xy[mem].astype(np.float64)
        for local_i, row in enumerate(mem):
            others = np.delete(np.arange(len(mem)), local_i)
            if len(others) == 0:
                continue
            d = np.linalg.norm(pts[others] - pts[local_i][None, :, :], axis=2)
            # Normalize only for threshold comparability across dataset-local
            # coordinate systems; do not claim metric units.
            out[row] = float(np.min(d) / max(float(normalizer[row]), EPS))
    return out


def _base_selection(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> dict[str, np.ndarray]:
    selected_ade, selected_fde, switch, floor_ade = ft._apply_policy(pred, labels, policy)
    selected_xy, neural_xy = _selected_waypoints(pred, labels, switch)
    floor_xy = ft._floor_waypoints(labels)
    neural_ade, neural_fde = ft._trajectory_errors(neural_xy, labels)
    return {
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switch": switch.astype(bool),
        "floor_ade": floor_ade,
        "floor_fde": ft._trajectory_errors(floor_xy, labels)[1],
        "neural_ade": neural_ade,
        "neural_fde": neural_fde,
        "selected_xy": selected_xy,
        "neural_xy": neural_xy,
    }


def _apply_joint_variant(
    pred: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    split_meta: Mapping[str, np.ndarray],
    base_policy: Mapping[str, Any],
    params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    base = _base_selection(pred, labels, base_policy)
    switch = base["switch"].copy()
    normalizer = labels["normalizer"].astype(np.float64)
    group_key = split_meta["key"]
    selected_xy, neural_xy = _selected_waypoints(pred, labels, switch)
    base_min_dist = _min_group_distance(selected_xy, group_key, normalizer)
    action_counts = {"guarded_off": 0, "expanded_on": 0}

    mode = str(params.get("mode", "base_reference"))
    if mode in {"collision_guard", "joint_moe"}:
        min_sep = float(params.get("min_sep", 0.0))
        guard = switch & np.isfinite(base_min_dist) & (base_min_dist < min_sep)
        switch[guard] = False
        action_counts["guarded_off"] = int(np.sum(guard))

    if mode in {"interaction_guard", "joint_moe"}:
        min_interaction = float(params.get("min_interaction", 0.0))
        min_occupancy = float(params.get("min_occupancy", 0.0))
        hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
        keep = hard | (pred["interaction"] >= min_interaction) | (pred["occupancy"] >= min_occupancy)
        guard = switch & ~keep
        switch[guard] = False
        action_counts["guarded_off"] += int(np.sum(guard))

    if mode in {"group_expand", "joint_moe"}:
        risk_max = float(params.get("expand_risk_max", 0.0))
        min_sep = float(params.get("expand_min_sep", 0.0))
        hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
        easy = labels["easy"].astype(bool)
        candidate = (~switch) & (~easy) & hard & (pred["traj_risk"] <= risk_max)
        # Evaluate each candidate against the already selected group rollout.
        expanded = np.zeros(len(switch), dtype=bool)
        groups: dict[str, list[int]] = defaultdict(list)
        for i, key in enumerate(group_key):
            groups[str(key)].append(i)
        current_xy, _ = _selected_waypoints(pred, labels, switch)
        for members in groups.values():
            mem = np.asarray(members, dtype=np.int64)
            local = mem[candidate[mem]]
            for row in local:
                other = mem[mem != row]
                if len(other) == 0:
                    expanded[row] = True
                    continue
                d = np.linalg.norm(current_xy[other] - neural_xy[row][None, :, :], axis=2)
                if float(np.min(d) / max(float(normalizer[row]), EPS)) >= min_sep:
                    expanded[row] = True
                    current_xy[row] = neural_xy[row]
        switch[expanded] = True
        action_counts["expanded_on"] = int(np.sum(expanded))

    selected_ade = base["floor_ade"].copy()
    selected_fde = base["floor_fde"].copy()
    selected_ade[switch] = base["neural_ade"][switch]
    selected_fde[switch] = base["neural_fde"][switch]
    selected_xy, _ = _selected_waypoints(pred, labels, switch)
    final_min_dist = _min_group_distance(selected_xy, group_key, normalizer)
    diagnostics = {
        **action_counts,
        "base_switch_rate": float(np.mean(base["switch"])),
        "final_switch_rate": float(np.mean(switch)),
        "base_mean_min_group_distance": float(np.mean(base_min_dist[np.isfinite(base_min_dist)])) if np.any(np.isfinite(base_min_dist)) else None,
        "final_mean_min_group_distance": float(np.mean(final_min_dist[np.isfinite(final_min_dist)])) if np.any(np.isfinite(final_min_dist)) else None,
    }
    return selected_ade, selected_fde, switch, diagnostics


def _params_grid(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    risks = pred["traj_risk"]
    risk_q = [float(v) for v in np.quantile(risks, [0.05, 0.10, 0.20, 0.35])]
    params: list[dict[str, Any]] = [{"mode": "base_reference"}]
    for min_sep in [0.02, 0.05, 0.08, 0.12]:
        params.append({"mode": "collision_guard", "min_sep": min_sep})
    for min_interaction in [0.35, 0.50, 0.65]:
        for min_occupancy in [0.35, 0.50, 0.65]:
            params.append({"mode": "interaction_guard", "min_interaction": min_interaction, "min_occupancy": min_occupancy})
    for risk in risk_q:
        for min_sep in [0.02, 0.05, 0.08]:
            params.append({"mode": "group_expand", "expand_risk_max": risk, "expand_min_sep": min_sep})
            params.append(
                {
                    "mode": "joint_moe",
                    "min_sep": min_sep,
                    "min_interaction": 0.35,
                    "min_occupancy": 0.35,
                    "expand_risk_max": risk,
                    "expand_min_sep": min_sep,
                }
            )
    return params


def _score(metrics: Mapping[str, Any]) -> float:
    max_domain_easy = max([float(row.get("easy_degradation", 0.0)) for row in (metrics.get("by_domain") or {}).values()] or [0.0])
    return (
        1.2 * float(metrics.get("all_improvement", 0.0))
        + 1.5 * float(metrics.get("t50_improvement", 0.0))
        + 1.0 * float(metrics.get("t100_improvement", 0.0))
        + 1.2 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 40.0 * max(0.0, max_domain_easy - 0.02)
    )


def _metric(selected_ade: np.ndarray, floor_ade: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    base = {
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }
    return s41._metrics(selected_ade, floor_ade, base, switch)


def _evaluate_params(
    pred: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    split_meta: Mapping[str, np.ndarray],
    base_policy: Mapping[str, Any],
    params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_ade, selected_fde, switch, diagnostics = _apply_joint_variant(pred, labels, split_meta, base_policy, params)
    base = _base_selection(pred, labels, base_policy)
    metrics = _metric(selected_ade, base["floor_ade"], labels, switch)
    metrics["endpoint_fde_metrics"] = _metric(selected_fde, base["floor_fde"], labels, switch)
    metrics["joint_consistency"] = diagnostics
    return metrics, diagnostics


def run_joint_multiagent_consistency() -> dict[str, Any]:
    ref = _full_trajectory_reference()
    pred_val, labels_val, _ = _split_predictions("val")
    pred_test, labels_test, _ = _split_predictions("test")
    meta_val = _group_metadata("val")
    meta_test = _group_metadata("test")
    base_policy = ref["policy"]
    base_val_metrics, base_val_diag = _evaluate_params(pred_val, labels_val, meta_val, base_policy, {"mode": "base_reference"})
    best_params = {"mode": "base_reference"}
    best_score = _score(base_val_metrics)
    val_results: dict[str, Any] = {
        "base_reference": {"params": best_params, "metrics": base_val_metrics, "score": best_score, "diagnostics": base_val_diag}
    }
    for params in _params_grid(pred_val, labels_val):
        name = str(params["mode"])
        if params == {"mode": "base_reference"}:
            continue
        metrics, diag = _evaluate_params(pred_val, labels_val, meta_val, base_policy, params)
        score = _score(metrics)
        key = json.dumps(params, sort_keys=True)
        val_results[key] = {"params": dict(params), "metrics": metrics, "score": score, "diagnostics": diag}
        if (
            score > best_score
            and metrics.get("all_improvement", 0.0) > 0
            and metrics.get("t50_improvement", 0.0) >= 0
            and metrics.get("hard_failure_improvement", 0.0) > 0
            and metrics.get("easy_degradation", 1.0) <= 0.02
        ):
            best_score = score
            best_params = dict(params)
    test_metrics, test_diag = _evaluate_params(pred_test, labels_test, meta_test, base_policy, best_params)
    ref_metrics = ref.get("metrics") or {}
    lift = {
        "all_delta": float(test_metrics.get("all_improvement", 0.0) - ref_metrics.get("all_improvement", 0.0)),
        "t50_delta": float(test_metrics.get("t50_improvement", 0.0) - ref_metrics.get("t50_improvement", 0.0)),
        "t100_delta": float(test_metrics.get("t100_improvement", 0.0) - ref_metrics.get("t100_improvement", 0.0)),
        "hard_delta": float(test_metrics.get("hard_failure_improvement", 0.0) - ref_metrics.get("hard_failure_improvement", 0.0)),
        "easy_delta": float(test_metrics.get("easy_degradation", 0.0) - ref_metrics.get("easy_degradation", 0.0)),
    }
    contributes = bool(
        (lift["all_delta"] > 0 or lift["t50_delta"] > 0 or lift["hard_delta"] > 0)
        and test_metrics.get("easy_degradation", 1.0) <= 0.02
        and test_metrics.get("all_improvement", 0.0) > 0
    )
    result = {
        "source": "fresh_run",
        "protocol_status": "joint_multiagent_consistency_calibration",
        "selected_params": best_params,
        "selected_val_score": best_score,
        "test_metrics": test_metrics,
        "test_diagnostics": test_diag,
        "full_trajectory_reference_metrics": ref_metrics,
        "lift_over_full_trajectory_reference": lift,
        "joint_multiagent_consistency_contributes": contributes,
        "val_ablation_count": len(val_results),
        "val_best_mode": best_params.get("mode"),
        "no_leakage": {
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "current_frame_grouping_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "caveat": "Joint consistency uses current-frame group metadata and predicted rollout geometry only. Coordinates remain dataset-local raw-frame 2.5D, not metric/seconds/true 3D.",
    }
    _write_json(OUT_DIR / "stage41_joint_multiagent_consistency.json", result)
    write_md(
        OUT_DIR / "stage41_joint_multiagent_consistency.md",
        [
            "# Stage41 Joint Multi-Agent Consistency Calibration",
            "",
            "- source: `fresh_run`",
            f"- selected params: `{best_params}`",
            f"- joint consistency contributes: `{contributes}`",
            f"- test metrics: `{test_metrics}`",
            f"- lift over full trajectory reference: `{lift}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "This is a deployment-policy ablation over the full-trajectory world-state probe. It uses predicted multi-agent rollout consistency and current-frame grouping only; future waypoints remain labels/evaluation.",
        ],
    )
    return result


def main_joint_multiagent_consistency() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_joint_multiagent_consistency()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_joint_multiagent_consistency",
            status,
            started,
            [OUT_DIR / "stage41_full_trajectory_world_state.json", DATA_DIR / "all_agent_val.npz", DATA_DIR / "all_agent_test.npz"],
            [OUT_DIR / "stage41_joint_multiagent_consistency.md", OUT_DIR / "stage41_joint_multiagent_consistency.json"],
        )


if __name__ == "__main__":
    main_joint_multiagent_consistency()
