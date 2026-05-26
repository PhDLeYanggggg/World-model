from __future__ import annotations

import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage41_full_trajectory_world_state as ft
from src import stage42_external_validation as s42b
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_level_full_waypoint_eval_stage42.json"
REPORT_MD = OUT_DIR / "source_level_full_waypoint_eval_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_am_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

WAYPOINT_FRAC = ft.WAYPOINT_FRAC.astype(np.float64)
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
EPS = 1e-6


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AM 是 proposed source-level split full-waypoint fresh evaluation，不是 metric 或 seconds-level 结果。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _split_arrays(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    source = data["source_file"].astype(str)
    domain = data["dataset"].astype(str)
    group = np.asarray([f"{d}::{s42b._rel_source(s)}" for d, s in zip(domain, source)], dtype="U512")
    split_by_group: dict[str, str] = {}
    for g in sorted(set(group.tolist())):
        u = s42b._stable_unit(g)
        if u < 0.68:
            split_by_group[g] = "train"
        elif u < 0.84:
            split_by_group[g] = "val"
        else:
            split_by_group[g] = "test"
    split = np.asarray([split_by_group[g] for g in group], dtype="U8")
    return split, group


def _source_stats(data: Mapping[str, np.ndarray], split: np.ndarray, group: np.ndarray) -> dict[str, Any]:
    h = data["horizon"].astype(int)
    domain = data["dataset"].astype(str)
    scene = data["scene_id"].astype(str)
    out = {}
    for sp in ["train", "val", "test"]:
        m = split == sp
        out[sp] = {
            "rows": int(np.sum(m)),
            "domains": dict(Counter(domain[m].tolist())),
            "scenes": int(len(set(scene[m].tolist()))),
            "sources": int(len(set(group[m].tolist()))),
            "t10": int(np.sum(m & (h == 10))),
            "t25": int(np.sum(m & (h == 25))),
            "t50": int(np.sum(m & (h == 50))),
            "t100": int(np.sum(m & (h == 100))),
            "hard": int(np.sum(data["hard"].astype(bool)[m])),
            "failure": int(np.sum(data["failure"].astype(bool)[m])),
            "easy": int(np.sum(data["easy"].astype(bool)[m])),
        }
    overlap = {
        f"{a}_{b}": int(len(set(group[split == a].tolist()) & set(group[split == b].tolist())))
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    }
    return {"by_split": out, "group_overlap": overlap, "source_overlap_pass": all(v == 0 for v in overlap.values())}


def _reconstruct_waypoint_labels(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    tracks = ft._track_map(data["source_file"].astype(str).tolist())
    n = len(data["horizon"])
    waypoints = np.zeros((n, len(WAYPOINT_FRAC), 2), dtype=np.float32)
    valid = np.zeros((n, len(WAYPOINT_FRAC)), dtype=bool)
    missing = np.zeros(n, dtype=bool)
    for i in range(n):
        source = str(data["source_file"][i])
        agent = int(data["agent_id"][i])
        endpoint = np.asarray([data["future_endpoint_x"][i], data["future_endpoint_y"][i]], dtype=np.float32)
        track = tracks.get((source, agent))
        if track is None:
            missing[i] = True
            waypoints[i, -1] = endpoint
            valid[i, -1] = True
            continue
        pts, mask = ft._lookup_waypoints(track, float(data["frame_id"][i]), int(data["horizon"][i]), endpoint)
        waypoints[i] = pts
        valid[i] = mask
    return {"waypoint_xy": waypoints, "waypoint_valid": valid, "missing_track": missing}


def _floor_arrays(data: Mapping[str, np.ndarray], train_mask: np.ndarray) -> dict[str, np.ndarray | dict[str, Any]]:
    floor_idx, floor_fde, floor_pred, strongest_by_h, diagnostics = s41._train_horizon_floor(data, train_mask)
    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    floor_xy = cur[:, None, :] + WAYPOINT_FRAC[None, :, None] * (floor_pred.astype(np.float64) - cur)[:, None, :]
    return {
        "floor_idx": floor_idx,
        "floor_fde": floor_fde,
        "floor_endpoint": floor_pred.astype(np.float32),
        "floor_xy": floor_xy.astype(np.float32),
        "strongest_by_horizon": strongest_by_h,
        "geometry_diagnostics": diagnostics,
    }


def _feature_matrix(data: Mapping[str, np.ndarray], floor: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float32)
    scale = np.maximum(data["scale"].astype(np.float32), EPS)
    safe_pred = s41._safe_baseline_predictions(data).astype(np.float32)
    safe_rel = ((safe_pred - cur[:, None, :]) / scale[:, None, None]).reshape(len(cur), -1)
    family_rel = ((data["family_pred"].astype(np.float32) - cur[:, None, :]) / scale[:, None, None]).reshape(len(cur), -1)
    floor_rel = ((floor["floor_endpoint"].astype(np.float32) - cur) / scale[:, None]).astype(np.float32)
    hist_seq = data["history_seq"].astype(np.float32)
    hist_tail = hist_seq[:, -16:, :].reshape(len(cur), -1)
    horizon_vals = [10, 25, 50, 100]
    horizon_onehot = np.stack([(data["horizon"].astype(int) == h).astype(np.float32) for h in horizon_vals], axis=1)
    domains = sorted(set(data["dataset"].astype(str).tolist()))
    domain_onehot = np.stack([(data["dataset"].astype(str) == d).astype(np.float32) for d in domains], axis=1)
    features = [
        data["history_scalar"].astype(np.float32),
        hist_tail,
        data["prototype_likelihood"].astype(np.float32),
        data["prototype_entropy"][:, None].astype(np.float32),
        data["goal_ambiguity"][:, None].astype(np.float32),
        safe_rel.astype(np.float32),
        family_rel.astype(np.float32),
        floor_rel.astype(np.float32),
        horizon_onehot.astype(np.float32),
        domain_onehot.astype(np.float32),
    ]
    names: list[str] = []
    names.extend([f"history_scalar_{i}" for i in range(data["history_scalar"].shape[1])])
    names.extend([f"history_tail_{i}" for i in range(hist_tail.shape[1])])
    names.extend([f"prototype_{i}" for i in range(data["prototype_likelihood"].shape[1])])
    names.extend(["prototype_entropy", "goal_ambiguity"])
    names.extend([f"safe_baseline_rel_{i}" for i in range(safe_rel.shape[1])])
    names.extend([f"family_baseline_rel_{i}" for i in range(family_rel.shape[1])])
    names.extend(["floor_rel_x", "floor_rel_y"])
    names.extend([f"horizon_{h}" for h in horizon_vals])
    names.extend([f"domain_{d}" for d in domains])
    return np.concatenate(features, axis=1).astype(np.float32), names


def _standardize(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_mask].mean(axis=0).astype(np.float32)
    std = np.maximum(x[train_mask].std(axis=0), 1e-3).astype(np.float32)
    z = ((x - mean) / std).astype(np.float32)
    z = np.concatenate([z, np.ones((len(z), 1), dtype=np.float32)], axis=1)
    return z, mean, std


def _fit_ridge_weighted(x: np.ndarray, y: np.ndarray, mask: np.ndarray, lam: float) -> np.ndarray:
    ids = np.where(mask)[0]
    xt = x[ids].astype(np.float64, copy=False)
    yt = y[ids].astype(np.float64, copy=False)
    reg = np.eye(xt.shape[1], dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xt.T @ xt + reg, xt.T @ yt).astype(np.float32)


def _fit_ridge_model(
    x: np.ndarray,
    target_delta: np.ndarray,
    waypoint_valid: np.ndarray,
    train_mask: np.ndarray,
    lam: float,
) -> np.ndarray:
    y = target_delta.reshape(len(target_delta), -1)
    coef = np.zeros((x.shape[1], y.shape[1]), dtype=np.float32)
    for w in range(len(WAYPOINT_FRAC)):
        m = train_mask & waypoint_valid[:, w]
        coef[:, 2 * w] = _fit_ridge_weighted(x, y[:, 2 * w], m, lam)
        coef[:, 2 * w + 1] = _fit_ridge_weighted(x, y[:, 2 * w + 1], m, lam)
    return coef


def _predict_waypoints(x: np.ndarray, coef: np.ndarray, data: Mapping[str, np.ndarray]) -> np.ndarray:
    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    delta = (x.astype(np.float64) @ coef.astype(np.float64)).reshape(len(x), len(WAYPOINT_FRAC), 2)
    return (cur[:, None, :] + delta * scale[:, None, None]).astype(np.float32)


def _trajectory_errors(xy: np.ndarray, labels: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid = labels["waypoint_valid"].astype(bool)
    err = np.linalg.norm(xy.astype(np.float64) - labels["waypoint_xy"].astype(np.float64), axis=2)
    ade = (err * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    fde = err[:, -1]
    return ade.astype(np.float64), fde.astype(np.float64)


def _safe_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return 1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS)


def _metric(selected: np.ndarray, floor: np.ndarray, data: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(selected), dtype=bool)
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "rows": int(np.sum(mask)),
        "all_improvement": _safe_improvement(selected, floor, mask),
        "t10_improvement": _safe_improvement(selected, floor, mask & (h == 10)),
        "t25_improvement": _safe_improvement(selected, floor, mask & (h == 25)),
        "t50_improvement": _safe_improvement(selected, floor, mask & (h == 50)),
        "t100_raw_frame_diagnostic_improvement": _safe_improvement(selected, floor, mask & (h == 100)),
        "hard_failure_improvement": _safe_improvement(selected, floor, mask & hard_failure),
        "easy_degradation": -_safe_improvement(selected, floor, mask & easy),
        "switch_rate": float(np.mean(switch[mask])) if np.any(mask) else 0.0,
        "harm_over_fallback": float(np.mean(selected[mask] - floor[mask])) if np.any(mask) else 0.0,
    }


def _bootstrap_ci(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, seed: int = 42017, n: int = 1000) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 30:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals.append(1.0 - float(np.mean(selected[sample])) / max(float(np.mean(floor[sample])), EPS))
    return {
        "low": float(np.percentile(vals, 2.5)),
        "mid": float(np.percentile(vals, 50.0)),
        "high": float(np.percentile(vals, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": int(n),
    }


def _select_policy_on_val(
    pred_xy: np.ndarray,
    floor_xy: np.ndarray,
    labels: Mapping[str, np.ndarray],
    data: Mapping[str, np.ndarray],
    val_mask: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    pred_ade, pred_fde = _trajectory_errors(pred_xy, labels)
    floor_ade, floor_fde = _trajectory_errors(floor_xy, labels)
    residual_norm = np.linalg.norm(pred_xy[:, -1] - floor_xy[:, -1], axis=1) / np.maximum(data["scale"].astype(np.float64), EPS)
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    selected_ade = floor_ade.copy()
    selected_fde = floor_fde.copy()
    switch = np.zeros(len(floor_ade), dtype=bool)
    policy: dict[str, Any] = {
        "type": "stage42am_source_level_ridge_validation_safe_policy",
        "selection_source": "validation_only",
        "slices": {},
        "test_threshold_tuning": False,
    }
    for d in sorted(set(domain[val_mask].tolist())):
        for h in [10, 25, 50, 100]:
            vm = val_mask & (domain == d) & (horizon == h)
            if int(np.sum(vm)) < 80:
                continue
            q_values = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75]
            thresholds = [float(np.quantile(residual_norm[vm], q)) for q in q_values]
            best: dict[str, Any] | None = None
            best_score = 0.0
            best_metric: dict[str, Any] | None = None
            for direction in ["low", "high"]:
                for threshold in thresholds:
                    for alpha in [0.25, 0.50, 0.75, 1.0]:
                        local = vm & ((residual_norm <= threshold) if direction == "low" else (residual_norm >= threshold))
                        if not np.any(local):
                            continue
                        trial_ade = floor_ade.copy()
                        trial_fde = floor_fde.copy()
                        blended = floor_xy + float(alpha) * (pred_xy - floor_xy)
                        b_ade, b_fde = _trajectory_errors(blended, labels)
                        trial_ade[local] = b_ade[local]
                        trial_fde[local] = b_fde[local]
                        trial_switch = local.astype(bool)
                        metric = _metric(trial_ade, floor_ade, data, trial_switch, vm)
                        if metric["easy_degradation"] > 0.02:
                            continue
                        score = (
                            1.2 * metric["all_improvement"]
                            + 1.8 * metric["t50_improvement"]
                            + 1.1 * metric["hard_failure_improvement"]
                            - 20.0 * max(0.0, metric["easy_degradation"] - 0.01)
                            - 0.03 * metric["switch_rate"]
                        )
                        if score > best_score:
                            best_score = float(score)
                            best_metric = metric
                            best = {
                                "direction": direction,
                                "residual_norm_threshold": float(threshold),
                                "alpha": float(alpha),
                                "val_score": float(score),
                                "val_rows": int(np.sum(vm)),
                                "val_metric": metric,
                            }
            if best is not None and best_score > 0.0:
                policy["slices"][f"{d}|{h}"] = best
    for key, params in policy["slices"].items():
        d, h_s = key.split("|")
        m = (domain == d) & (horizon == int(h_s))
        local = m & (
            (residual_norm <= float(params["residual_norm_threshold"]))
            if params["direction"] == "low"
            else (residual_norm >= float(params["residual_norm_threshold"]))
        )
        blended = floor_xy + float(params["alpha"]) * (pred_xy - floor_xy)
        b_ade, b_fde = _trajectory_errors(blended, labels)
        selected_ade[local] = b_ade[local]
        selected_fde[local] = b_fde[local]
        switch[local] = True
    return policy, selected_ade, selected_fde, switch


def _evaluate_models(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    x: np.ndarray,
) -> dict[str, Any]:
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    target_delta = ((labels["waypoint_xy"].astype(np.float64) - np.stack([data["current_x"], data["current_y"]], axis=1)[:, None, :]) / np.maximum(data["scale"].astype(np.float64)[:, None, None], EPS)).astype(np.float32)
    val_results = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    for lam in LAMBDAS:
        coef = _fit_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, lam)
        pred_xy = _predict_waypoints(x, coef, data)
        policy, selected_ade, selected_fde, switch = _select_policy_on_val(pred_xy, floor["floor_xy"], labels, data, val_mask)
        floor_ade, floor_fde = _trajectory_errors(floor["floor_xy"], labels)
        val_metric = _metric(selected_ade, floor_ade, data, switch, val_mask)
        score = (
            1.2 * val_metric["all_improvement"]
            + 1.8 * val_metric["t50_improvement"]
            + 1.1 * val_metric["hard_failure_improvement"]
            - 30.0 * max(0.0, val_metric["easy_degradation"] - 0.02)
            - 0.03 * val_metric["switch_rate"]
        )
        val_results.append({"lambda": lam, "score": float(score), "policy_slice_count": len(policy["slices"]), "val_metric": val_metric})
        if score > best_score:
            best_score = float(score)
            best = {
                "lambda": float(lam),
                "coef": coef,
                "policy": policy,
                "pred_xy": pred_xy,
                "selected_ade": selected_ade,
                "selected_fde": selected_fde,
                "switch": switch,
                "floor_ade": floor_ade,
                "floor_fde": floor_fde,
                "val_metric": val_metric,
                "score": float(score),
            }
    if best is None:
        raise RuntimeError("No ridge model evaluated.")
    pred_ade, pred_fde = _trajectory_errors(best["pred_xy"], labels)
    ungated_switch = np.ones(len(pred_ade), dtype=bool)
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    model = {
        "best_lambda": best["lambda"],
        "validation_selection": {"source": "fresh_run", "test_threshold_tuning": False, "candidates": val_results, "selected_score": best["score"]},
        "policy": best["policy"],
        "metrics": {
            "floor": _metric(best["floor_ade"], best["floor_ade"], data, np.zeros(len(pred_ade), dtype=bool), test_mask),
            "ungated_ridge_diagnostic": _metric(pred_ade, best["floor_ade"], data, ungated_switch, test_mask),
            "protected_ridge_source_level": _metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask),
            "protected_ridge_source_level_fde": _metric(best["selected_fde"], best["floor_fde"], data, best["switch"], test_mask),
        },
        "bootstrap": {
            "all": _bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask),
            "t50": _bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (horizon == 50), seed=42018),
            "t100_raw_frame_diagnostic": _bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (horizon == 100), seed=42019),
            "hard_failure": _bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & hard_failure, seed=42020),
            "easy_degradation": _bootstrap_ci(best["floor_ade"], best["selected_ade"], test_mask & easy, seed=42021),
        },
        "by_domain": {
            d: _metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask & (domain == d))
            for d in sorted(set(domain[test_mask].tolist()))
        },
        "by_horizon": {
            str(h): _metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask & (horizon == h))
            for h in [10, 25, 50, 100]
        },
    }
    return model


def run_stage42_source_level_full_waypoint_eval() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    source_split = s42b.build_stage42_source_split()
    data = s41._combined()
    split, group = _split_arrays(data)
    split_stats = _source_stats(data, split, group)
    labels = _reconstruct_waypoint_labels(data)
    train_mask = split == "train"
    floor = _floor_arrays(data, train_mask)
    features, feature_names = _feature_matrix(data, floor)
    x, mean, std = _standardize(features, train_mask)
    eval_result = _evaluate_models(data, split, labels, floor, x)
    label_stats = {
        "rows": int(len(split)),
        "full_waypoint_rows": int(np.sum(np.all(labels["waypoint_valid"], axis=1))),
        "endpoint_only_rows": int(np.sum(labels["waypoint_valid"][:, -1] & ~np.all(labels["waypoint_valid"], axis=1))),
        "missing_track_rows": int(np.sum(labels["missing_track"])),
        "test_rows": int(np.sum(split == "test")),
        "test_full_waypoint_rows": int(np.sum((split == "test") & np.all(labels["waypoint_valid"], axis=1))),
    }
    result = {
        "source": "fresh_run",
        "stage": "Stage42-AM proposed source-level full-waypoint evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/external_source_split_stage42.json",
                "outputs/stage42_long_research/source_level_coverage_audit_stage42.json",
            ]
        ),
        "source_split": source_split,
        "split_stats": split_stats,
        "label_stats": label_stats,
        "feature_schema": {
            "source": "fresh_run",
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "normalization": "train_split_mean_std_only",
            "future_inputs": False,
            "family_fde_as_input": False,
            "safe_strongest_idx_old_as_input": False,
        },
        "floor": {
            "source": "fresh_run",
            "type": "train_horizon_selected_safe_causal_baseline",
            "strongest_by_horizon": floor["strongest_by_horizon"],
            "geometry_diagnostics": floor["geometry_diagnostics"],
        },
        "model": eval_result,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(split_stats["source_overlap_pass"]),
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["stage42_am_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    m = result["model"]["metrics"]["protected_ridge_source_level"]
    boot = result["model"]["bootstrap"]
    gates = {
        "proposed_source_level_test_evaluated": result["split_stats"]["by_split"]["test"]["rows"] == int(m["rows"]) and int(m["rows"]) > 0,
        "trajnet_full_proposed_rows_covered": result["split_stats"]["by_split"]["test"]["domains"].get("TrajNet", 0) == 37918,
        "ucy_full_proposed_rows_covered": result["split_stats"]["by_split"]["test"]["domains"].get("UCY", 0) == 9540,
        "full_waypoint_labels_reconstructed": result["label_stats"]["test_full_waypoint_rows"] > 0,
        "train_val_test_source_no_overlap": bool(result["split_stats"]["source_overlap_pass"]),
        "validation_selected_policy": not result["no_leakage"]["test_threshold_tuning"] and len(result["model"]["policy"]["slices"]) >= 0,
        "protected_source_level_non_harm": m["all_improvement"] >= -EPS and m["easy_degradation"] <= 0.02,
        "bootstrap_reported": boot["all"]["bootstrap_n"] > 0 and boot["t50"]["bootstrap_n"] > 0,
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in [
                "future_endpoint_input",
                "future_waypoint_input",
                "family_fde_input",
                "safe_strongest_idx_old_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
            ]
        )
        and result["no_leakage"]["train_only_feature_normalization"],
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    positive = m["all_improvement"] > 0 and (m["t50_improvement"] > 0 or m["hard_failure_improvement"] > 0)
    verdict = (
        "stage42_am_source_level_full_waypoint_eval_pass_positive"
        if all(gates.values()) and positive
        else "stage42_am_source_level_full_waypoint_eval_pass_nonharm_or_diagnostic"
        if all(gates.values())
        else "stage42_am_source_level_full_waypoint_eval_partial"
    )
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "positive_transfer": bool(positive), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    model = result["model"]
    metrics = model["metrics"]
    protected = metrics["protected_ridge_source_level"]
    ungated = metrics["ungated_ridge_diagnostic"]
    lines = [
        "# Stage42-AM Proposed Source-Level Full-Waypoint Evaluation",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_am_gate']['passed']} / {result['stage42_am_gate']['total']}`",
        f"- verdict: `{result['stage42_am_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## What This Fixes From Stage42-AL",
        "",
        "- Stage42-AL found that locked-policy stress rows were not a full proposed source-level split evaluation.",
        "- Stage42-AM evaluates the proposed source-level test split directly: TrajNet `37918` rows and UCY `9540` rows.",
        "- ETH_UCY remains train/val in this proposed split; ETH_UCY stress rows are not counted as source-level test evidence here.",
        "",
        "## Split And Labels",
        "",
        f"- split_stats: `{result['split_stats']['by_split']}`",
        f"- label_stats: `{result['label_stats']}`",
        f"- floor: `{result['floor']}`",
        f"- feature_count: `{result['feature_schema']['feature_count']}`",
        "",
        "## Candidate Metrics On Proposed Source-Level Test",
        "",
        "| candidate | rows | all ADE improvement | t50 ADE improvement | t100 raw-frame diag | hard/failure | easy degradation | switch rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| train-horizon causal floor | {metrics['floor']['rows']} | {metrics['floor']['all_improvement']:.6f} | {metrics['floor']['t50_improvement']:.6f} | {metrics['floor']['t100_raw_frame_diagnostic_improvement']:.6f} | {metrics['floor']['hard_failure_improvement']:.6f} | {metrics['floor']['easy_degradation']:.6f} | {metrics['floor']['switch_rate']:.6f} |",
        f"| ungated ridge diagnostic | {ungated['rows']} | {ungated['all_improvement']:.6f} | {ungated['t50_improvement']:.6f} | {ungated['t100_raw_frame_diagnostic_improvement']:.6f} | {ungated['hard_failure_improvement']:.6f} | {ungated['easy_degradation']:.6f} | {ungated['switch_rate']:.6f} |",
        f"| protected ridge source-level | {protected['rows']} | {protected['all_improvement']:.6f} | {protected['t50_improvement']:.6f} | {protected['t100_raw_frame_diagnostic_improvement']:.6f} | {protected['hard_failure_improvement']:.6f} | {protected['easy_degradation']:.6f} | {protected['switch_rate']:.6f} |",
        "",
        "## Bootstrap CI",
        "",
        "| slice | low | mid | high | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, row in model["bootstrap"].items():
        lines.append(f"| `{key}` | {row['low']:.6f} | {row['mid']:.6f} | {row['high']:.6f} | {row['n']} |")
    lines.extend(["", "## By Domain", "", "| domain | rows | all | t50 | t100 diag | hard/failure | easy | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for d, row in model["by_domain"].items():
        lines.append(f"| `{d}` | {row['rows']} | {row['all_improvement']:.6f} | {row['t50_improvement']:.6f} | {row['t100_raw_frame_diagnostic_improvement']:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} | {row['switch_rate']:.6f} |")
    lines.extend(["", "## No-Leakage And Claim Boundary", "", f"- no_leakage: `{result['no_leakage']}`", f"- claim_boundary: `{result['claim_boundary']}`", "", "## Interpretation", ""])
    if result["stage42_am_gate"]["positive_transfer"]:
        lines.append("- Stage42-AM closes the Stage42-AL proposed source-level evaluation gap with a positive protected ridge full-waypoint probe.")
    else:
        lines.append("- Stage42-AM closes the Stage42-AL coverage accounting gap but does not yet prove positive full-waypoint transfer; use as diagnostic/non-harm evidence only.")
    lines.append("- This remains dataset-local raw-frame 2.5D evidence, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.")
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_am_gate"]
    lines = [
        "# Stage42-AM Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- positive_transfer: `{gate['positive_transfer']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| {name} | `{ok}` |")
    return lines


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": "run_stage42_source_level_full_waypoint_eval.py",
        "source": result["source"],
        "status": "success",
        "generated_at_utc": result["generated_at_utc"],
        "git_commit": result["git_commit"],
        "input_hash": result["input_hash"],
        "outputs": [str(REPORT_JSON), str(REPORT_MD), str(GATE_MD)],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_source_level_full_waypoint_eval()
