from __future__ import annotations

import json
import sys
import time
import copy
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_domain_local_full_trajectory_world_state as dlft
from src import stage41_domain_local_neural_retrain as dl
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_rollout_consistency as jrc


OUT_DIR = dl.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_domain_local_full_trajectory_repair.json"
REPORT_MD = OUT_DIR / "stage41_domain_local_full_trajectory_repair.md"
BOOTSTRAP_N = 800
SEED = 41617
EPS = 1e-6

MODES = ["raw_waypoint", "endpoint_linearized", "blend_25_raw_75_endpoint"]
RIDGE_LAMBDAS = [0.1, 1.0]
MAX_GAIN_TRAIN_ROWS = 50_000
VAL_EASY_LIMIT = 0.005


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
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _mode_xy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], mode: str) -> np.ndarray:
    raw_xy = ft._pred_waypoints(pred, labels)
    if mode == "raw_waypoint":
        return raw_xy
    endpoint = raw_xy[:, -1, :]
    linear_xy = labels["current_xy"][:, None, :] + ft.WAYPOINT_FRAC[None, :, None] * (endpoint - labels["current_xy"])[:, None, :]
    if mode == "endpoint_linearized":
        return linear_xy.astype(np.float64)
    if mode == "blend_25_raw_75_endpoint":
        return (0.25 * raw_xy + 0.75 * linear_xy).astype(np.float64)
    raise ValueError(f"unknown repair trajectory mode: {mode}")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _feature_matrix(tensors: Mapping[str, Any], pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], mode: str) -> np.ndarray:
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = _mode_xy(pred, labels, mode)
    current = labels["current_xy"].astype(np.float64)
    normalizer = np.maximum(labels["normalizer"].astype(np.float64), EPS)
    floor_delta = (floor_xy[:, -1, :] - current) / normalizer[:, None]
    neural_delta = (neural_xy[:, -1, :] - current) / normalizer[:, None]
    delta_gap = (neural_xy[:, -1, :] - floor_xy[:, -1, :]) / normalizer[:, None]
    floor_seg = np.linalg.norm(np.diff(np.concatenate([current[:, None, :], floor_xy], axis=1), axis=1), axis=2) / normalizer[:, None]
    neural_seg = np.linalg.norm(np.diff(np.concatenate([current[:, None, :], neural_xy], axis=1), axis=1), axis=2) / normalizer[:, None]
    horizon = labels["horizon"].astype(np.float64)
    horizon_feats = np.column_stack([(horizon == 10), (horizon == 25), (horizon == 50), (horizon == 100)]).astype(np.float64)
    static = _to_numpy(tensors["static"]).astype(np.float64)
    core = np.column_stack(
        [
            pred["traj_risk"].astype(np.float64),
            pred["physical"].astype(np.float64),
            pred["interaction"].astype(np.float64),
            pred["occupancy"].astype(np.float64),
            np.linalg.norm(floor_delta, axis=1),
            np.linalg.norm(neural_delta, axis=1),
            np.linalg.norm(delta_gap, axis=1),
            floor_seg.sum(axis=1),
            neural_seg.sum(axis=1),
            np.max(neural_seg, axis=1),
            np.std(neural_seg, axis=1),
            horizon / 100.0,
        ]
    )
    x = np.concatenate([core, horizon_feats, static], axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> dict[str, Any]:
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    xn = (x - mean) / std
    xb = np.concatenate([np.ones((len(xn), 1), dtype=np.float64), xn], axis=1)
    reg = lam * np.eye(xb.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    w = np.linalg.solve(xb.T @ xb + reg, xb.T @ y.astype(np.float64))
    return {"mean": mean, "std": std, "weights": w, "lambda": float(lam)}


def _ridge_predict(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    w = np.asarray(model["weights"], dtype=np.float64)
    xn = (x - mean) / std
    xb = np.concatenate([np.ones((len(xn), 1), dtype=np.float64), xn], axis=1)
    return xb @ w


def _metrics(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    return s41._metrics(selected, floor, dlft._metric_ds(labels), switch)


def _bootstrap(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, seed: int) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 20:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(BOOTSTRAP_N):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals.append(1.0 - float(selected[sample].mean()) / max(float(floor[sample].mean()), EPS))
    return {"low": float(np.percentile(vals, 2.5)), "mid": float(np.percentile(vals, 50)), "high": float(np.percentile(vals, 97.5)), "n": int(len(ids)), "bootstrap_n": BOOTSTRAP_N}


def _switch_from_params(
    pred: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    mask: np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    # Inference-time policy uses only model outputs and split/horizon metadata.
    # Hard/easy/failure labels remain validation/evaluation labels only.
    local = (
        mask
        & (pred["traj_risk"] <= float(params.get("traj_risk_max", np.inf)))
        & (pred["physical"] >= float(params.get("physical_prob_min", 0.0)))
        & (pred["interaction"] >= float(params.get("interaction_prob_min", 0.0)))
        & (pred["occupancy"] >= float(params.get("occupancy_prob_min", 0.0)))
    )
    max_switch = float(params.get("max_switch", 1.0))
    if max_switch <= 0.0:
        return np.zeros(len(mask), dtype=bool)
    if max_switch < 1.0 and np.any(local):
        ids = np.where(local)[0]
        keep_n = max(1, int(max_switch * int(np.sum(mask))))
        keep = np.zeros(len(mask), dtype=bool)
        keep[ids[np.argsort(pred["traj_risk"][ids])[:keep_n]]] = True
        local &= keep
    return local


def _switch_from_gain_params(
    gain_score: np.ndarray,
    pred: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    mask: np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    local = (
        mask
        & (gain_score >= float(params.get("gain_threshold", np.inf)))
        & (pred["physical"] >= float(params.get("physical_prob_min", 0.0)))
        & (pred["traj_risk"] <= float(params.get("traj_risk_max", np.inf)))
    )
    max_switch = float(params.get("max_switch", 1.0))
    if max_switch <= 0.0:
        return np.zeros(len(mask), dtype=bool)
    if max_switch < 1.0 and np.any(local):
        ids = np.where(local)[0]
        keep_n = max(1, int(max_switch * int(np.sum(mask))))
        keep = np.zeros(len(mask), dtype=bool)
        # Higher predicted gain is safer for intervention.
        keep[ids[np.argsort(gain_score[ids])[::-1][:keep_n]]] = True
        local &= keep
    return local


def _fit_policy_for_mode(
    pred: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = _mode_xy(pred, labels, mode)
    floor_ade, _floor_fde = ft._trajectory_errors(floor_xy, labels)
    neural_ade, _neural_fde = ft._trajectory_errors(neural_xy, labels)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    selected_all = floor_ade.copy()
    switch_all = np.zeros(len(floor_ade), dtype=bool)
    policy = {"type": "domain_local_full_trajectory_repair_policy", "trajectory_mode": mode, "slices": {}}
    diagnostics: dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        for h in [10, 25, 50, 100]:
            mask = (domain == d) & (horizon == h)
            rows = int(np.sum(mask))
            if rows < 80:
                diagnostics[f"{d}|{h}"] = {"selected": False, "rows": rows, "reason": "too_few_rows"}
                continue
            risk = pred["traj_risk"][mask]
            risk_grid = [float(v) for v in np.quantile(risk, [0.05, 0.10, 0.20, 0.35, 0.55, 0.75])]
            best_score = -1e18
            best_params: dict[str, Any] | None = None
            best_metrics: dict[str, Any] | None = None
            for traj_risk_max in risk_grid:
                for physical_prob_min in [0.0, 0.35, 0.55, 0.70]:
                    for interaction_prob_min in [0.0, 0.20, 0.40]:
                        for occupancy_prob_min in [0.0, 0.20]:
                            for max_switch in [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70]:
                                params = {
                                    "traj_risk_max": traj_risk_max,
                                    "physical_prob_min": physical_prob_min,
                                    "interaction_prob_min": interaction_prob_min,
                                    "occupancy_prob_min": occupancy_prob_min,
                                    "max_switch": max_switch,
                                }
                                switch = _switch_from_params(pred, labels, mask, params)
                                selected = floor_ade.copy()
                                selected[switch] = neural_ade[switch]
                                m = _metrics(selected[mask], floor_ade[mask], {k: v[mask] for k, v in labels.items()}, switch[mask])
                                if m.get("all_improvement", 0.0) <= 0.0 or m.get("easy_degradation", 1.0) > VAL_EASY_LIMIT:
                                    continue
                                # Horizon-specific score. The h=50/h=100 slices are no
                                # longer drowned by all-test rows, which was a Stage41
                                # failure mode on ETH_UCY.
                                score = (
                                    m.get("all_improvement", 0.0)
                                    + 2.2 * m.get("t50_improvement", 0.0)
                                    + 1.6 * m.get("t100_improvement", 0.0)
                                    + 1.4 * m.get("hard_failure_improvement", 0.0)
                                - 30.0 * m.get("easy_degradation", 0.0)
                                )
                                if score > best_score:
                                    best_score = score
                                    best_params = params
                                    best_metrics = m
            if best_params is not None:
                switch = _switch_from_params(pred, labels, mask, best_params)
                selected_all[mask] = np.where(switch[mask], neural_ade[mask], floor_ade[mask])
                switch_all |= switch
                policy["slices"][f"{d}|{h}"] = best_params
            diagnostics[f"{d}|{h}"] = {
                "selected": bool(best_params is not None),
                "rows": rows,
                "val_score": float(best_score if best_params is not None else 0.0),
                "val_metrics": best_metrics or {"rows": rows, "all_improvement": 0.0},
            }
    val_metrics = _metrics(selected_all, floor_ade, labels, switch_all)
    val_metrics["slice_diagnostics"] = diagnostics
    val_metrics["trajectory_mode"] = mode
    return policy, val_metrics


def _apply_policy_with_mode(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(policy.get("trajectory_mode", "raw_waypoint"))
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = _mode_xy(pred, labels, mode)
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels)
    neural_ade, neural_fde = ft._trajectory_errors(neural_xy, labels)
    selected_xy = floor_xy.copy()
    selected_ade = floor_ade.copy()
    selected_fde = floor_fde.copy()
    switch = np.zeros(len(floor_ade), dtype=bool)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    for key, params in policy.get("slices", {}).items():
        d, h_s = key.split("|")
        mask = (domain == d) & (horizon == int(h_s))
        local = _switch_from_params(pred, labels, mask, params)
        selected_xy[local] = neural_xy[local]
        selected_ade[local] = neural_ade[local]
        selected_fde[local] = neural_fde[local]
        switch |= local
    return {
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_xy": floor_xy,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "neural_xy": neural_xy,
        "neural_ade": neural_ade,
        "neural_fde": neural_fde,
        "switch": switch,
    }


def _fit_gain_policy_for_mode(
    train_tensors: Mapping[str, Any],
    pred_train: Mapping[str, np.ndarray],
    labels_train: Mapping[str, np.ndarray],
    val_tensors: Mapping[str, Any],
    pred_val: Mapping[str, np.ndarray],
    labels_val: Mapping[str, np.ndarray],
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_floor_xy = ft._floor_waypoints(labels_train)
    train_neural_xy = _mode_xy(pred_train, labels_train, mode)
    train_floor_ade, _ = ft._trajectory_errors(train_floor_xy, labels_train)
    train_neural_ade, _ = ft._trajectory_errors(train_neural_xy, labels_train)
    x_train = _feature_matrix(train_tensors, pred_train, labels_train, mode)
    y_train = (train_floor_ade - train_neural_ade).astype(np.float64)
    if len(x_train) > MAX_GAIN_TRAIN_ROWS:
        rng = np.random.default_rng(SEED + len(x_train))
        keep = np.sort(rng.choice(np.arange(len(x_train)), size=MAX_GAIN_TRAIN_ROWS, replace=False))
        x_train = x_train[keep]
        y_train = y_train[keep]
    val_floor_xy = ft._floor_waypoints(labels_val)
    val_neural_xy = _mode_xy(pred_val, labels_val, mode)
    val_floor_ade, _ = ft._trajectory_errors(val_floor_xy, labels_val)
    val_neural_ade, _ = ft._trajectory_errors(val_neural_xy, labels_val)
    x_val = _feature_matrix(val_tensors, pred_val, labels_val, mode)
    domain = labels_val["domain"].astype(str)
    horizon = labels_val["horizon"].astype(int)
    best_model: dict[str, Any] | None = None
    best_policy: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = -1e18
    model_diagnostics: dict[str, Any] = {}
    for lam in RIDGE_LAMBDAS:
        model = _fit_ridge(x_train, y_train, lam)
        gain_val = _ridge_predict(model, x_val)
        selected_all = val_floor_ade.copy()
        switch_all = np.zeros(len(val_floor_ade), dtype=bool)
        policy = {
            "type": "domain_local_full_trajectory_gain_calibrated_policy",
            "trajectory_mode": mode,
            "gain_model": {
                "mean": model["mean"],
                "std": model["std"],
                "weights": model["weights"],
                "lambda": float(lam),
                "target": "floor_ade_minus_neural_ade",
                "feature_source": "past_only_static_plus_neural_outputs_and_rollout_diagnostics",
            },
            "slices": {},
        }
        slice_diag: dict[str, Any] = {}
        for d in sorted(set(domain.tolist())):
            for h in [10, 25, 50, 100]:
                mask = (domain == d) & (horizon == h)
                rows = int(np.sum(mask))
                if rows < 80:
                    slice_diag[f"{d}|{h}"] = {"selected": False, "rows": rows, "reason": "too_few_rows"}
                    continue
                gains = gain_val[mask]
                risk = pred_val["traj_risk"][mask]
                threshold_grid = sorted(set([0.0, *[float(v) for v in np.quantile(gains, [0.65, 0.80, 0.92])]]))
                risk_grid = [float(v) for v in np.quantile(risk, [0.55, 0.85])]
                best_params: dict[str, Any] | None = None
                best_slice_metrics: dict[str, Any] | None = None
                best_slice_score = -1e18
                for gain_threshold in threshold_grid:
                    for traj_risk_max in risk_grid:
                        for physical_prob_min in [0.0, 0.55]:
                            for max_switch in [0.0, 0.02, 0.05, 0.10, 0.20]:
                                params = {
                                    "gain_threshold": gain_threshold,
                                    "traj_risk_max": traj_risk_max,
                                    "physical_prob_min": physical_prob_min,
                                    "max_switch": max_switch,
                                }
                                switch = _switch_from_gain_params(gain_val, pred_val, labels_val, mask, params)
                                selected = val_floor_ade.copy()
                                selected[switch] = val_neural_ade[switch]
                                m = _metrics(selected[mask], val_floor_ade[mask], {k: v[mask] for k, v in labels_val.items()}, switch[mask])
                                if m.get("all_improvement", 0.0) <= 0.0 or m.get("easy_degradation", 1.0) > VAL_EASY_LIMIT:
                                    continue
                                score = (
                                    m.get("all_improvement", 0.0)
                                    + 2.2 * m.get("t50_improvement", 0.0)
                                    + 1.6 * m.get("t100_improvement", 0.0)
                                    + 1.4 * m.get("hard_failure_improvement", 0.0)
                                    - 30.0 * m.get("easy_degradation", 0.0)
                                )
                                if score > best_slice_score:
                                    best_slice_score = score
                                    best_params = params
                                    best_slice_metrics = m
                if best_params is not None:
                    switch = _switch_from_gain_params(gain_val, pred_val, labels_val, mask, best_params)
                    selected_all[mask] = np.where(switch[mask], val_neural_ade[mask], val_floor_ade[mask])
                    switch_all |= switch
                    policy["slices"][f"{d}|{h}"] = best_params
                slice_diag[f"{d}|{h}"] = {
                    "selected": bool(best_params is not None),
                    "rows": rows,
                    "val_score": float(best_slice_score if best_params is not None else 0.0),
                    "val_metrics": best_slice_metrics or {"rows": rows, "all_improvement": 0.0},
                }
        m_all = _metrics(selected_all, val_floor_ade, labels_val, switch_all)
        m_all["slice_diagnostics"] = slice_diag
        m_all["trajectory_mode"] = mode
        m_all["lambda"] = float(lam)
        score_all = (
            m_all.get("all_improvement", 0.0)
            + 2.0 * m_all.get("t50_improvement", 0.0)
            + m_all.get("t100_improvement", 0.0)
            + 1.2 * m_all.get("hard_failure_improvement", 0.0)
            - 15.0 * max(0.0, m_all.get("easy_degradation", 0.0) - 0.02)
        )
        model_diagnostics[str(lam)] = {
            "score": float(score_all),
            "metrics": {
                "rows": m_all.get("rows", 0),
                "all_improvement": m_all.get("all_improvement", 0.0),
                "t50_improvement": m_all.get("t50_improvement", 0.0),
                "t100_improvement": m_all.get("t100_improvement", 0.0),
                "hard_failure_improvement": m_all.get("hard_failure_improvement", 0.0),
                "easy_degradation": m_all.get("easy_degradation", 0.0),
                "switch_rate": m_all.get("switch_rate", 0.0),
            },
        }
        if score_all > best_score:
            best_score = score_all
            best_model = model
            best_policy = policy
            best_metrics = m_all
    assert best_model is not None and best_policy is not None and best_metrics is not None
    best_policy["gain_model"]["mean"] = best_model["mean"]
    best_policy["gain_model"]["std"] = best_model["std"]
    best_policy["gain_model"]["weights"] = best_model["weights"]
    best_metrics["model_diagnostics"] = model_diagnostics
    best_metrics["policy_family"] = "gain_calibrated"
    return best_policy, best_metrics


def _apply_gain_policy_with_mode(
    tensors: Mapping[str, Any],
    pred: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    mode = str(policy.get("trajectory_mode", "raw_waypoint"))
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = _mode_xy(pred, labels, mode)
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels)
    neural_ade, neural_fde = ft._trajectory_errors(neural_xy, labels)
    model = policy["gain_model"]
    gain_score = _ridge_predict(model, _feature_matrix(tensors, pred, labels, mode))
    selected_xy = floor_xy.copy()
    selected_ade = floor_ade.copy()
    selected_fde = floor_fde.copy()
    switch = np.zeros(len(floor_ade), dtype=bool)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    for key, params in policy.get("slices", {}).items():
        d, h_s = key.split("|")
        mask = (domain == d) & (horizon == int(h_s))
        local = _switch_from_gain_params(gain_score, pred, labels, mask, params)
        selected_xy[local] = neural_xy[local]
        selected_ade[local] = neural_ade[local]
        selected_fde[local] = neural_fde[local]
        switch |= local
    return {
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_xy": floor_xy,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "neural_xy": neural_xy,
        "neural_ade": neural_ade,
        "neural_fde": neural_fde,
        "switch": switch,
        "gain_score": gain_score,
    }


def _choose_guard(
    bundle: Mapping[str, Any],
    labels: Mapping[str, np.ndarray],
    keys: np.ndarray,
) -> dict[str, Any]:
    floor_min = jmc._min_group_distance(bundle["floor_xy"], keys, labels["normalizer"].astype(np.float64))
    selected_min_base = jmc._min_group_distance(bundle["selected_xy"], keys, labels["normalizer"].astype(np.float64))
    best: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    for min_sep in [0.0, 0.02, 0.04, 0.05, 0.08, 0.12, 0.18]:
        switch = bundle["switch"].copy()
        if min_sep > 0:
            guard = switch & np.isfinite(selected_min_base) & (selected_min_base < min_sep) & (selected_min_base < floor_min)
            switch[guard] = False
        else:
            guard = np.zeros(len(switch), dtype=bool)
        selected_xy = bundle["floor_xy"].copy()
        selected_xy[switch] = bundle["neural_xy"][switch]
        selected_ade, _selected_fde = ft._trajectory_errors(selected_xy, labels)
        m = _metrics(selected_ade, bundle["floor_ade"], labels, switch)
        # Validation guard is deliberately approximate for speed: rows that are
        # guarded off inherit the floor min-distance. The final test evaluation
        # still recomputes exact same-frame proximity from selected_xy.
        selected_min = selected_min_base.copy()
        selected_min[guard] = floor_min[guard]
        finite_floor = np.isfinite(floor_min)
        finite_sel = np.isfinite(selected_min)
        floor_nc = float(np.mean(floor_min[finite_floor] < 0.05)) if np.any(finite_floor) else 0.0
        selected_nc = float(np.mean(selected_min[finite_sel] < 0.05)) if np.any(finite_sel) else 0.0
        collision_delta = selected_nc - floor_nc
        eligible = bool(
            m.get("all_improvement", 0.0) > 0.0
            and m.get("hard_failure_improvement", 0.0) > 0.0
            and m.get("easy_degradation", 1.0) <= VAL_EASY_LIMIT
            and collision_delta <= 0.01
        )
        score = (
            m.get("all_improvement", 0.0)
            + 2.0 * m.get("t50_improvement", 0.0)
            + m.get("t100_improvement", 0.0)
            + 1.2 * m.get("hard_failure_improvement", 0.0)
            - 30.0 * max(0.0, m.get("easy_degradation", 0.0) - VAL_EASY_LIMIT)
            - 3.0 * max(0.0, collision_delta - 0.01)
        )
        row = {
            "min_sep": float(min_sep),
            "guarded_off": int(np.sum(guard)),
            "eligible": eligible,
            "score": float(score),
            "val_metrics": m,
            "val_collision_delta_005": float(collision_delta),
        }
        candidates.append(row)
        if best is None or (eligible, score) > (bool(best["eligible"]), float(best["score"])):
            best = row
    assert best is not None
    return {"selected": best, "candidates": candidates}


def _score_from_metrics(m: Mapping[str, Any]) -> float:
    return float(
        m.get("all_improvement", 0.0)
        + 2.0 * m.get("t50_improvement", 0.0)
        + m.get("t100_improvement", 0.0)
        + 1.2 * m.get("hard_failure_improvement", 0.0)
        - 30.0 * max(0.0, m.get("easy_degradation", 0.0) - VAL_EASY_LIMIT)
    )


def _apply_guard(bundle: Mapping[str, Any], labels: Mapping[str, np.ndarray], keys: np.ndarray, min_sep: float) -> tuple[np.ndarray, np.ndarray, int]:
    switch = bundle["switch"].copy()
    if min_sep > 0:
        floor_min = jmc._min_group_distance(bundle["floor_xy"], keys, labels["normalizer"].astype(np.float64))
        selected_min = jmc._min_group_distance(bundle["selected_xy"], keys, labels["normalizer"].astype(np.float64))
        guard = switch & np.isfinite(selected_min) & (selected_min < min_sep) & (selected_min < floor_min)
        switch[guard] = False
    else:
        guard = np.zeros(len(switch), dtype=bool)
    selected_xy = bundle["floor_xy"].copy()
    selected_xy[switch] = bundle["neural_xy"][switch]
    return selected_xy, switch, int(np.sum(guard))


def _evaluate(selected_xy: np.ndarray, switch: np.ndarray, bundle: Mapping[str, Any], labels: Mapping[str, np.ndarray], keys: np.ndarray) -> dict[str, Any]:
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, labels)
    neural_ade, neural_fde = ft._trajectory_errors(bundle["neural_xy"], labels)
    group_counts = np.asarray([Counter(map(str, keys.tolist()))[str(k)] for k in keys], dtype=np.int32)
    multi = group_counts >= 2
    floor_stats = jrc._joint_stats("floor", bundle["floor_xy"], labels, keys, np.zeros(len(switch), dtype=bool))
    selected_stats = jrc._joint_stats("domain_local_full_trajectory_repair_selected", selected_xy, labels, keys, switch)
    neural_stats = jrc._joint_stats("domain_local_full_trajectory_repair_neural_without_fallback", bundle["neural_xy"], labels, keys, np.ones(len(switch), dtype=bool))
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    return {
        "ade_metrics_vs_floor": _metrics(selected_ade, bundle["floor_ade"], labels, switch),
        "fde_metrics_vs_floor": _metrics(selected_fde, bundle["floor_fde"], labels, switch),
        "multi_agent_ade_metrics": dlft._safe_metrics(selected_ade, bundle["floor_ade"], labels, switch, multi),
        "neural_without_fallback_ade": _metrics(neural_ade, bundle["floor_ade"], labels, np.ones(len(switch), dtype=bool)),
        "neural_without_fallback_fde": _metrics(neural_fde, bundle["floor_fde"], labels, np.ones(len(switch), dtype=bool)),
        "rollout_stats": {"floor": floor_stats, "selected": selected_stats, "neural_without_fallback": neural_stats},
        "collision_delta_vs_floor_005": float(selected_stats["near_collision_rate_005"] - floor_stats["near_collision_rate_005"]),
        "smoothness_jagged_delta": float(selected_stats["smoothness"]["jagged_rate"] - floor_stats["smoothness"]["jagged_rate"]),
        "bootstrap_ade": {
            "all": _bootstrap(selected_ade, bundle["floor_ade"], np.ones(len(switch), dtype=bool), SEED),
            "t50": _bootstrap(selected_ade, bundle["floor_ade"], horizon == 50, SEED + 1),
            "t100": _bootstrap(selected_ade, bundle["floor_ade"], horizon == 100, SEED + 2),
            "hard_failure": _bootstrap(selected_ade, bundle["floor_ade"], hard, SEED + 3),
            "multi_agent": _bootstrap(selected_ade, bundle["floor_ade"], multi, SEED + 4),
        },
    }


def _load_previous() -> dict[str, Any]:
    if not dlft.REPORT_JSON.exists():
        dlft.run_domain_local_full_trajectory_world_state()
    return json.loads(dlft.REPORT_JSON.read_text(encoding="utf-8"))


def _domain_candidate(domain: str, row: Mapping[str, Any], train_all: Mapping[str, Any], val_all: Mapping[str, Any], test_all: Mapping[str, Any]) -> dict[str, Any]:
    train = dlft._subset_tensors(train_all, dlft._domain_mask(train_all, domain))
    val = dlft._subset_tensors(val_all, dlft._domain_mask(val_all, domain))
    test = dlft._subset_tensors(test_all, dlft._domain_mask(test_all, domain))
    trial_rows = row.get("trials", {})
    if not trial_rows:
        return {"domain": domain, "status": "not_run", "reason": "no cached trained full-waypoint domain-local trial available"}
    best: dict[str, Any] | None = None
    candidates: dict[str, Any] = {}
    preferred_ckpt = str(row.get("best_trial") or "")
    selected_trial_rows = {
        name: trial
        for name, trial in trial_rows.items()
        if not preferred_ckpt or str((trial.get("training") or {}).get("checkpoint")) == preferred_ckpt
    }
    if not selected_trial_rows:
        selected_trial_rows = dict(trial_rows)
    for trial_name, trial in selected_trial_rows.items():
        ckpt = (trial.get("training") or {}).get("checkpoint")
        if not ckpt or not Path(ckpt).exists():
            continue
        pred_train, labels_train = dlft._predict(ckpt, train)
        pred_val, labels_val = dlft._predict(ckpt, val)
        for mode in MODES:
            for family in ["risk_only", "gain_calibrated"]:
                if family == "risk_only":
                    policy, val_metrics = _fit_policy_for_mode(pred_val, labels_val, mode)
                else:
                    policy, val_metrics = _fit_gain_policy_for_mode(train, pred_train, labels_train, val, pred_val, labels_val, mode)
                score = _score_from_metrics(val_metrics)
                key = f"{trial_name}|{mode}|{family}"
                candidates[key] = {
                    "checkpoint": ckpt,
                    "trial": trial_name,
                    "mode": mode,
                    "policy_family": family,
                    "policy": policy,
                    "val_metrics": val_metrics,
                    "proximity_guard_selection": {"status": "not_run_before_top_candidate_filter"},
                    "val_score": score,
                }
    top_base = sorted(candidates.values(), key=lambda c: float(c["val_score"]), reverse=True)[:2]
    top = []
    for candidate in top_base:
        top.append(candidate)
        if candidate.get("policy_family") == "gain_calibrated":
            for variant, suffixes in {
                "t50_only": ("|50",),
                "long_horizon_only": ("|50", "|100"),
            }.items():
                cloned = copy.deepcopy(candidate)
                cloned["deployment_variant"] = variant
                cloned["policy"] = copy.deepcopy(candidate["policy"])
                cloned["policy"]["deployment_variant"] = variant
                cloned["policy"]["slices"] = {
                    key: params for key, params in candidate["policy"].get("slices", {}).items() if key.endswith(suffixes)
                }
                if cloned["policy"]["slices"]:
                    top.append(cloned)
    for candidate in top:
        pred_val, labels_val = dlft._predict(candidate["checkpoint"], val)
        if candidate.get("policy_family") == "gain_calibrated":
            bundle_val = _apply_gain_policy_with_mode(val, pred_val, labels_val, candidate["policy"])
        else:
            bundle_val = _apply_policy_with_mode(pred_val, labels_val, candidate["policy"])
        guard = _choose_guard(bundle_val, labels_val, dlft._keys(labels_val))
        candidate["proximity_guard_selection"] = guard
        candidate["val_score"] = float(guard["selected"]["score"])
        selected = guard["selected"]
        val_m = selected["val_metrics"]
        rank = (
            bool(selected["eligible"]),
            bool(val_m.get("t50_improvement", 0.0) > 0.0),
            bool(candidate.get("deployment_variant", "all_horizons") != "all_horizons"),
            float(candidate["val_score"]),
        )
        candidate["selection_rank"] = rank
        if best is None:
            best = candidate
            continue
        best_selected = best["proximity_guard_selection"]["selected"]
        best_m = best_selected["val_metrics"]
        best_rank = (
            bool(best_selected["eligible"]),
            bool(best_m.get("t50_improvement", 0.0) > 0.0),
            bool(best.get("deployment_variant", "all_horizons") != "all_horizons"),
            float(best["val_score"]),
        )
        if rank > best_rank:
            best = candidate
    if best is None:
        return {"domain": domain, "status": "not_run", "reason": "no usable cached checkpoint for repair evaluation"}
    pred_test, labels_test = dlft._predict(best["checkpoint"], test)
    if best.get("policy_family") == "gain_calibrated":
        bundle_test = _apply_gain_policy_with_mode(test, pred_test, labels_test, best["policy"])
    else:
        bundle_test = _apply_policy_with_mode(pred_test, labels_test, best["policy"])
    keys_test = dlft._keys(labels_test)
    selected_xy, switch, guarded_off = _apply_guard(
        bundle_test,
        labels_test,
        keys_test,
        float(best["proximity_guard_selection"]["selected"]["min_sep"]),
    )
    ev = _evaluate(selected_xy, switch, bundle_test, labels_test, keys_test)
    m = ev["ade_metrics_vs_floor"]
    mm = ev["multi_agent_ade_metrics"]
    pass_gate = bool(
        m.get("all_improvement", 0.0) > 0.0
        and m.get("t50_improvement", 0.0) > 0.0
        and m.get("hard_failure_improvement", 0.0) > 0.0
        and m.get("easy_degradation", 1.0) <= 0.02
        and mm.get("all_improvement", 0.0) > 0.0
        and ev["collision_delta_vs_floor_005"] <= 0.01
        and ev["smoothness_jagged_delta"] <= 0.01
        and (ev["bootstrap_ade"]["all"]["low"] > 0 or ev["bootstrap_ade"]["t50"]["low"] > 0 or ev["bootstrap_ade"]["hard_failure"]["low"] > 0)
    )
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "best_trial": best["trial"],
        "best_checkpoint": best["checkpoint"],
        "trajectory_mode": best["mode"],
        "policy_family": best.get("policy_family", "risk_only"),
        "deployment_variant": best.get("deployment_variant", "all_horizons"),
        "policy": best["policy"],
        "val_score": best["val_score"],
        "val_metrics": best["val_metrics"],
        "proximity_guard_selection": best["proximity_guard_selection"],
        "test_guarded_off": guarded_off,
        **ev,
        "repair_gate": pass_gate,
        "candidates": candidates,
    }


def run_domain_local_full_trajectory_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    previous = _load_previous()
    ft.build_full_trajectory_labels()
    train_all = ft._load_tensors("train")
    val_all = ft._load_tensors("val")
    test_all = ft._load_tensors("test")
    results: dict[str, Any] = {}
    for domain, row in previous.get("domain_results", {}).items():
        if row.get("status") == "ok":
            results[domain] = _domain_candidate(domain, row, train_all, val_all, test_all)
    positive_domains = [d for d, row in results.items() if row.get("repair_gate")]
    failure_taxonomy: dict[str, Any] = {}
    for domain, row in results.items():
        if row.get("status") != "ok" or row.get("repair_gate"):
            continue
        m = row.get("ade_metrics_vs_floor", {})
        f = row.get("fde_metrics_vs_floor", {})
        reasons = []
        if m.get("all_improvement", 0.0) <= 0:
            reasons.append("all_ade_not_positive")
        if m.get("t50_improvement", 0.0) <= 0:
            reasons.append("t50_ade_not_positive")
        if m.get("hard_failure_improvement", 0.0) <= 0:
            reasons.append("hard_failure_ade_not_positive")
        if m.get("easy_degradation", 1.0) > 0.02:
            reasons.append("easy_degradation_over_2pct")
        if row.get("collision_delta_vs_floor_005", 1.0) > 0.01:
            reasons.append("same_frame_proximity_delta_unsafe")
        if f.get("all_improvement", 0.0) > 0 and m.get("all_improvement", 0.0) <= 0:
            reasons.append("endpoint_fde_positive_but_waypoint_ade_negative")
        if row.get("trajectory_mode") == "endpoint_linearized" and m.get("all_improvement", 0.0) <= 0:
            reasons.append("endpoint_linearization_insufficient")
        failure_taxonomy[domain] = {
            "reasons": reasons,
            "trajectory_mode": row.get("trajectory_mode"),
            "policy_family": row.get("policy_family"),
            "deployment_variant": row.get("deployment_variant"),
            "ade_all": m.get("all_improvement"),
            "ade_t50": m.get("t50_improvement"),
            "ade_t100": m.get("t100_improvement"),
            "fde_all": f.get("all_improvement"),
            "fde_t50": f.get("t50_improvement"),
            "collision_delta_vs_floor_005": row.get("collision_delta_vs_floor_005"),
        }
    result = {
        "source": "fresh_run",
        "protocol": "domain_local_full_trajectory_repair_endpoint_linearized_gain_calibrated_guarded",
        "stage5c_executed": False,
        "smc_enabled": False,
        "metric_or_seconds_claim": False,
        "previous_report": str(dlft.REPORT_JSON),
        "trajectory_modes": MODES,
        "positive_domains": positive_domains,
        "positive_domain_count": int(len(positive_domains)),
        "two_domain_repair_gate": bool(len(positive_domains) >= 2),
        "domain_results": results,
        "failure_taxonomy": failure_taxonomy,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "val_selected_policy": True,
            "inference_switch_uses_hard_easy_labels": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "learned_full_waypoint_neural_dynamics": True,
            "endpoint_linearized_repair_if_selected": True,
            "all_active_agent_world_state_audit": True,
            "latent_generative_rollout": False,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
        },
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Domain-Local Full-Trajectory Repair",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- trajectory modes tested: `{MODES}`",
        f"- positive domains: `{positive_domains}`",
        f"- two-domain repair gate: `{result['two_domain_repair_gate']}`",
        f"- failure taxonomy: `{failure_taxonomy}`",
        "",
        "| domain | family | variant | mode | all ADE | t50 ADE | t100 ADE | hard ADE | easy | multi all | collision d005 | guard off | pass |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain, row in results.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | `{row.get('reason')}` | `not_run` | `not_run` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `False` |")
            continue
        m = row["ade_metrics_vs_floor"]
        mm = row["multi_agent_ade_metrics"]
        lines.append(
            f"| `{domain}` | `{row.get('policy_family')}` | `{row.get('deployment_variant')}` | `{row.get('trajectory_mode')}` | "
            f"{float(m.get('all_improvement', 0.0)):.4f} | {float(m.get('t50_improvement', 0.0)):.4f} | "
            f"{float(m.get('t100_improvement', 0.0)):.4f} | {float(m.get('hard_failure_improvement', 0.0)):.4f} | "
            f"{float(m.get('easy_degradation', 0.0)):.4f} | {float(mm.get('all_improvement', 0.0)):.4f} | "
            f"{float(row.get('collision_delta_vs_floor_005', 0.0)):.4f} | {int(row.get('test_guarded_off', 0))} | `{row.get('repair_gate')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This repair uses cached trained neural full-waypoint checkpoints and performs a fresh validation-selected policy/evaluation pass.",
            "- Endpoint-linearized mode is explicitly labeled when selected: it tests whether the neural model learned endpoint dynamics while failing intermediate waypoint shape. It is not claimed as learned full-shape dynamics.",
            "- Gain-calibrated mode trains a ridge switch head on train split labels only, selects thresholds on val, and evaluates test once.",
            "- Inference-time switching does not consume hard/easy/failure labels; those remain validation/evaluation labels.",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_domain_local_full_trajectory_repair() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_domain_local_full_trajectory_repair()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_domain_local_full_trajectory_repair",
            status,
            started,
            [dlft.REPORT_JSON, ft.DATA_DIR / "all_agent_val.npz", ft.DATA_DIR / "all_agent_test.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_domain_local_full_trajectory_repair()
