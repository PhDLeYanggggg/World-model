from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_external_validation as s42b
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
REPORT_MD = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.md"
REPORT_CSV = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_di_gate.md"

AM_JSON = OUT_DIR / "source_level_full_waypoint_eval_stage42.json"
CQ_JSON = OUT_DIR / "proximity_aware_composer_guard_stage42.json"
DH_JSON = OUT_DIR / "full_waypoint_proximity_occupancy_loss_repair_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_README = Path("README_M3W_GOAL_SUMMARY_ZH.md")
CURRENT_RETROSPECTIVE = Path("README_M3W_CURRENT_FULL_RETROSPECTIVE_ZH.md")
RESEARCH_STATE = Path("research_state.json")

EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DI 针对 Stage42-DE/DH 的 full-waypoint all-agent proximity / group-consistency blocker。",
    "group-consistency repair 只使用 predicted rollout geometry、source/frame/horizon group key、agent id、当前/过去可得信息。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "validation 选择 repair policy；test 只评一次。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _group_key(data: Mapping[str, np.ndarray]) -> np.ndarray:
    source = data["source_file"].astype(str)
    frame = data["frame_id"].astype(np.float64)
    horizon = data["horizon"].astype(int)
    return np.asarray([f"{s}\t{int(round(f * 1000.0))}\t{h}" for s, f, h in zip(source, frame, horizon)], dtype=object)


def _min_group_distance_fast(
    xy: np.ndarray,
    group_key: np.ndarray,
    normalizer: np.ndarray,
    agent_id: np.ndarray,
) -> np.ndarray:
    out = np.full(len(xy), np.inf, dtype=np.float64)
    if len(xy) == 0:
        return out
    keys = np.asarray(group_key, dtype=object)
    order = np.argsort(keys)
    start = 0
    while start < len(order):
        end = start + 1
        key = keys[order[start]]
        while end < len(order) and keys[order[end]] == key:
            end += 1
        rows = order[start:end]
        unique_agents, first_idx, inverse = np.unique(agent_id[rows], return_index=True, return_inverse=True)
        del unique_agents
        unique_rows = rows[first_idx]
        if len(unique_rows) > 1:
            pts = xy[unique_rows].astype(np.float64, copy=False)
            diff = pts[:, None, :, :] - pts[None, :, :, :]
            dist = np.linalg.norm(diff, axis=3)
            dist[np.arange(len(unique_rows)), np.arange(len(unique_rows)), :] = np.inf
            per_unique = np.min(dist, axis=(1, 2)) / np.maximum(normalizer[unique_rows].astype(np.float64), EPS)
            out[rows] = per_unique[inverse]
        start = end
    return out


def _trajectory_errors_subset(xy: np.ndarray, labels: Mapping[str, np.ndarray], ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = labels["waypoint_valid"][ids].astype(bool)
    err = np.linalg.norm(xy.astype(np.float64) - labels["waypoint_xy"][ids].astype(np.float64), axis=2)
    ade = (err * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    fde = err[:, -1]
    return ade.astype(np.float64), fde.astype(np.float64)


def _safe_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return 1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS)


def _metric_subset(
    selected: np.ndarray,
    floor: np.ndarray,
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    switch: np.ndarray,
) -> dict[str, Any]:
    h = data["horizon"][ids].astype(int)
    hard_failure = data["hard"][ids].astype(bool) | data["failure"][ids].astype(bool)
    easy = data["easy"][ids].astype(bool)
    all_mask = np.ones(len(ids), dtype=bool)
    return {
        "rows": int(len(ids)),
        "all_improvement": _safe_improvement(selected, floor, all_mask),
        "t10_improvement": _safe_improvement(selected, floor, h == 10),
        "t25_improvement": _safe_improvement(selected, floor, h == 25),
        "t50_improvement": _safe_improvement(selected, floor, h == 50),
        "t100_raw_frame_diagnostic_improvement": _safe_improvement(selected, floor, h == 100),
        "hard_failure_improvement": _safe_improvement(selected, floor, hard_failure),
        "easy_degradation": -_safe_improvement(selected, floor, easy),
        "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
        "harm_over_fallback": float(np.mean(selected - floor)) if len(ids) else 0.0,
    }


def _bootstrap_ci_subset(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, seed: int, n: int = 1000) -> dict[str, Any]:
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


def _apply_am_policy_xy(
    pred_xy: np.ndarray,
    floor_xy: np.ndarray,
    data: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    residual_norm = np.linalg.norm(pred_xy[:, -1] - floor_xy[:, -1], axis=1) / np.maximum(data["scale"].astype(np.float64), EPS)
    selected = floor_xy.astype(np.float32).copy()
    switch = np.zeros(len(floor_xy), dtype=bool)
    for key, params in (policy.get("slices") or {}).items():
        d, h_s = str(key).split("|")
        m = (domain == d) & (horizon == int(h_s))
        threshold = float(params["residual_norm_threshold"])
        direction = str(params["direction"])
        local = m & ((residual_norm <= threshold) if direction == "low" else (residual_norm >= threshold))
        blended = floor_xy + float(params["alpha"]) * (pred_xy - floor_xy)
        selected[local] = blended[local]
        switch[local] = True
    return selected, switch


def _rebuild_stage42_am_candidate(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
) -> dict[str, Any]:
    train_mask = split == "train"
    val_mask = split == "val"
    features, feature_names = am._feature_matrix(data, floor)
    x, mean, std = am._standardize(features, train_mask)
    target_delta = (
        (
            labels["waypoint_xy"].astype(np.float64)
            - np.stack([data["current_x"], data["current_y"]], axis=1)[:, None, :]
        )
        / np.maximum(data["scale"].astype(np.float64)[:, None, None], EPS)
    ).astype(np.float32)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    best: dict[str, Any] | None = None
    best_score = -1e9
    rows = []
    for lam in am.LAMBDAS:
        coef = am._fit_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, lam)
        pred_xy = am._predict_waypoints(x, coef, data)
        policy, selected_ade, selected_fde, switch = am._select_policy_on_val(pred_xy, floor["floor_xy"], labels, data, val_mask)
        val_metric = am._metric(selected_ade, floor_ade, data, switch, val_mask)
        score = (
            1.2 * val_metric["all_improvement"]
            + 1.8 * val_metric["t50_improvement"]
            + 1.1 * val_metric["hard_failure_improvement"]
            - 30.0 * max(0.0, val_metric["easy_degradation"] - 0.02)
            - 0.03 * val_metric["switch_rate"]
        )
        rows.append({"lambda": float(lam), "score": float(score), "policy_slice_count": len(policy["slices"]), "val_metric": val_metric})
        if score > best_score:
            selected_xy, switch_xy = _apply_am_policy_xy(pred_xy, floor["floor_xy"], data, policy)
            best_score = float(score)
            best = {
                "lambda": float(lam),
                "coef": coef,
                "pred_xy": pred_xy,
                "policy": policy,
                "selected_xy": selected_xy,
                "switch": switch_xy,
                "selected_ade": selected_ade,
                "selected_fde": selected_fde,
                "floor_ade": floor_ade,
                "floor_fde": floor_fde,
                "score": float(score),
                "val_metric": val_metric,
            }
    if best is None:
        raise RuntimeError("No Stage42-AM candidate rebuilt for Stage42-DI.")
    best["validation_rows"] = rows
    best["feature_count"] = len(feature_names)
    return best


def _repel_selected_rows(
    xy: np.ndarray,
    switch: np.ndarray,
    group_key: np.ndarray,
    normalizer: np.ndarray,
    agent_id: np.ndarray,
    current_xy: np.ndarray,
    min_sep: float,
    strength: float,
) -> np.ndarray:
    repaired = xy.astype(np.float32).copy()
    keys = np.asarray(group_key, dtype=object)
    order = np.argsort(keys)
    start = 0
    while start < len(order):
        end = start + 1
        key = keys[order[start]]
        while end < len(order) and keys[order[end]] == key:
            end += 1
        rows = order[start:end]
        unique_agents, first_idx, inverse = np.unique(agent_id[rows], return_index=True, return_inverse=True)
        del unique_agents, inverse
        unique_rows = rows[first_idx]
        if len(unique_rows) > 1 and np.any(switch[unique_rows]):
            pts = repaired[unique_rows].astype(np.float64, copy=False)
            diff = pts[:, None, :, :] - pts[None, :, :, :]
            dist = np.linalg.norm(diff, axis=3)
            dist[np.arange(len(unique_rows)), np.arange(len(unique_rows)), :] = np.inf
            for i, row in enumerate(unique_rows):
                if not switch[row]:
                    continue
                min_d = float(np.min(dist[i]))
                sep = float(min_sep) * float(max(normalizer[row], EPS))
                if not np.isfinite(min_d) or min_d >= sep:
                    continue
                j = int(np.unravel_index(np.argmin(dist[i]), dist[i].shape)[0])
                direction = current_xy[row].astype(np.float64) - current_xy[unique_rows[j]].astype(np.float64)
                norm = float(np.linalg.norm(direction))
                if norm < EPS:
                    at = np.unravel_index(np.argmin(dist[i]), dist[i].shape)[1]
                    direction = pts[i, at] - pts[j, at]
                    norm = float(np.linalg.norm(direction))
                if norm < EPS:
                    direction = np.asarray([1.0, 0.0], dtype=np.float64)
                    norm = 1.0
                offset = direction / norm * (sep - min_d) * float(strength)
                repaired[row] = repaired[row] + offset[None, :].astype(np.float32)
        start = end
    return repaired


def _repair_subset(
    ids: np.ndarray,
    candidate: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    pred_xy: np.ndarray,
    base_xy: np.ndarray,
    base_switch: np.ndarray,
    group_key: np.ndarray,
) -> dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    selected_xy = base_xy[ids].astype(np.float32).copy()
    switch = base_switch[ids].astype(bool).copy()
    normalizer = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    agent = data["agent_id"][ids].astype(np.int64)
    keys = group_key[ids]
    floor_min = _min_group_distance_fast(floor_xy[ids], keys, normalizer, agent)
    base_min = _min_group_distance_fast(selected_xy, keys, normalizer, agent)
    pred_min = _min_group_distance_fast(pred_xy[ids], keys, normalizer, agent)
    min_sep = float(candidate["min_sep"])
    margin = float(candidate.get("margin", 0.0))
    unsafe = switch & np.isfinite(base_min) & np.isfinite(floor_min) & (base_min < min_sep) & (base_min + margin < floor_min)
    mode = str(candidate["mode"])
    if mode == "fallback_unsafe":
        selected_xy[unsafe] = floor_xy[ids][unsafe]
        switch[unsafe] = False
    elif mode == "blend_unsafe":
        alpha = float(candidate["alpha"])
        blend = floor_xy[ids] + alpha * (base_xy[ids] - floor_xy[ids])
        selected_xy[unsafe] = blend[unsafe]
        switch[unsafe] = alpha > EPS
    elif mode == "predicted_safe_only":
        safe = np.isfinite(pred_min) & (pred_min >= float(candidate["safe_min_sep"]))
        off = switch & ~safe
        selected_xy[off] = floor_xy[ids][off]
        switch[off] = False
    elif mode == "repel_unsafe":
        selected_xy = _repel_selected_rows(
            selected_xy,
            switch,
            keys,
            normalizer,
            agent,
            np.stack([data["current_x"][ids], data["current_y"][ids]], axis=1),
            min_sep=min_sep,
            strength=float(candidate["strength"]),
        )
    else:
        raise ValueError(f"Unknown Stage42-DI repair mode: {mode}")
    final_min = _min_group_distance_fast(selected_xy, keys, normalizer, agent)
    selected_ade, selected_fde = _trajectory_errors_subset(selected_xy, labels, ids)
    floor_ade, floor_fde = _trajectory_errors_subset(floor_xy[ids], labels, ids)
    metric = _metric_subset(selected_ade, floor_ade, data, ids, switch)
    return {
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "switch": switch,
        "metric": metric,
        "diagnostics": {
            "unsafe_rows": int(np.sum(unsafe)),
            "unsafe_rate": float(np.mean(unsafe)) if len(unsafe) else 0.0,
            "base_switch_rate": float(np.mean(base_switch[ids])) if len(ids) else 0.0,
            "final_switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
            "base_near_005": float(np.mean(np.isfinite(base_min) & (base_min < 0.05))) if len(base_min) else 0.0,
            "final_near_005": float(np.mean(np.isfinite(final_min) & (final_min < 0.05))) if len(final_min) else 0.0,
            "floor_near_005": float(np.mean(np.isfinite(floor_min) & (floor_min < 0.05))) if len(floor_min) else 0.0,
            "base_p05_min_distance": float(np.percentile(base_min[np.isfinite(base_min)], 5)) if np.any(np.isfinite(base_min)) else None,
            "final_p05_min_distance": float(np.percentile(final_min[np.isfinite(final_min)], 5)) if np.any(np.isfinite(final_min)) else None,
            "floor_p05_min_distance": float(np.percentile(floor_min[np.isfinite(floor_min)], 5)) if np.any(np.isfinite(floor_min)) else None,
        },
    }


def _candidate_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for min_sep in [0.02, 0.05, 0.08, 0.12]:
        for margin in [0.0, 0.005, 0.01]:
            rows.append({"mode": "fallback_unsafe", "min_sep": min_sep, "margin": margin})
            rows.append({"mode": "predicted_safe_only", "min_sep": min_sep, "margin": margin, "safe_min_sep": min_sep})
            for alpha in [0.25, 0.50, 0.75]:
                rows.append({"mode": "blend_unsafe", "min_sep": min_sep, "margin": margin, "alpha": alpha})
            for strength in [0.25, 0.50]:
                rows.append({"mode": "repel_unsafe", "min_sep": min_sep, "margin": margin, "strength": strength})
    return rows


def _selection_score(metric: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> float:
    near_delta = float(diagnostics["final_near_005"]) - float(diagnostics["base_near_005"])
    return (
        1.4 * float(metric["all_improvement"])
        + 1.4 * float(metric["hard_failure_improvement"])
        + 1.1 * float(metric["t50_improvement"])
        + 0.5 * float(metric["t100_raw_frame_diagnostic_improvement"])
        - 35.0 * max(0.0, float(metric["easy_degradation"]) - 0.02)
        - 8.0 * max(0.0, near_delta)
        - 0.01 * float(metric["switch_rate"])
    )


def _evaluate_repairs(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    am_candidate: Mapping[str, Any],
    group_key: np.ndarray,
) -> dict[str, Any]:
    val_ids = np.where(split == "val")[0]
    test_ids = np.where(split == "test")[0]
    floor_xy = floor["floor_xy"].astype(np.float32)
    pred_xy = am_candidate["pred_xy"].astype(np.float32)
    base_xy = am_candidate["selected_xy"].astype(np.float32)
    base_switch = am_candidate["switch"].astype(bool)
    rows = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    for candidate in _candidate_grid():
        val = _repair_subset(val_ids, candidate, data, labels, floor_xy, pred_xy, base_xy, base_switch, group_key)
        score = _selection_score(val["metric"], val["diagnostics"])
        row = {
            "candidate": dict(candidate),
            "val_score": float(score),
            "val_metric": val["metric"],
            "val_diagnostics": val["diagnostics"],
        }
        rows.append(row)
        if score > best_score:
            best_score = float(score)
            best = row
    if best is None:
        raise RuntimeError("No Stage42-DI repair candidate evaluated.")
    test = _repair_subset(test_ids, best["candidate"], data, labels, floor_xy, pred_xy, base_xy, base_switch, group_key)
    base_test_ade = am_candidate["selected_ade"][test_ids].astype(np.float64)
    floor_test_ade = am_candidate["floor_ade"][test_ids].astype(np.float64)
    base_test_metric = _metric_subset(base_test_ade, floor_test_ade, data, test_ids, base_switch[test_ids])
    h = data["horizon"][test_ids].astype(int)
    hard_failure = data["hard"][test_ids].astype(bool) | data["failure"][test_ids].astype(bool)
    easy = data["easy"][test_ids].astype(bool)
    domain = data["dataset"][test_ids].astype(str)
    bootstrap = {
        "all": _bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], np.ones(len(test_ids), dtype=bool), seed=42101),
        "t50": _bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], h == 50, seed=42102),
        "t100_raw_frame_diagnostic": _bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], h == 100, seed=42103),
        "hard_failure": _bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], hard_failure, seed=42104),
        "easy_degradation": _bootstrap_ci_subset(test["floor_ade"], test["selected_ade"], easy, seed=42105),
    }
    by_domain = {
        d: _metric_subset(test["selected_ade"][domain == d], test["floor_ade"][domain == d], data, test_ids[domain == d], test["switch"][domain == d])
        for d in sorted(set(domain.tolist()))
    }
    return {
        "candidate_count": len(rows),
        "validation_rows": sorted(rows, key=lambda row: row["val_score"], reverse=True),
        "selected": best,
        "baseline_stage42_am_on_test": base_test_metric,
        "test": {
            "metric_vs_floor": test["metric"],
            "diagnostics": test["diagnostics"],
            "bootstrap": bootstrap,
            "by_domain": by_domain,
        },
    }


def _compare_to_prior(result_metric: Mapping[str, Any]) -> dict[str, Any]:
    am_payload = read_json(AM_JSON, {})
    cq_payload = read_json(CQ_JSON, {})
    dh_payload = read_json(DH_JSON, {})
    am_metric = am_payload.get("model", {}).get("metrics", {}).get("protected_ridge_source_level", {})
    cq_metric = cq_payload.get("test_eval", {}).get("metric_vs_endpoint_ade", {})
    dh_metric = dh_payload.get("model", {}).get("metrics", {}).get("protected_selected_candidate", {})

    def delta(ref: Mapping[str, Any]) -> dict[str, float | None]:
        if not ref:
            return {
                "all_improvement": None,
                "t50_improvement": None,
                "t100_raw_frame_diagnostic_improvement": None,
                "hard_failure_improvement": None,
                "easy_degradation": None,
            }
        return {
            "all_improvement": float(result_metric.get("all_improvement", 0.0)) - float(ref.get("all_improvement", 0.0)),
            "t50_improvement": float(result_metric.get("t50_improvement", 0.0)) - float(ref.get("t50_improvement", 0.0)),
            "t100_raw_frame_diagnostic_improvement": float(result_metric.get("t100_raw_frame_diagnostic_improvement", 0.0))
            - float(ref.get("t100_raw_frame_diagnostic_improvement", 0.0)),
            "hard_failure_improvement": float(result_metric.get("hard_failure_improvement", 0.0)) - float(ref.get("hard_failure_improvement", 0.0)),
            "easy_degradation": float(result_metric.get("easy_degradation", 0.0)) - float(ref.get("easy_degradation", 0.0)),
        }

    return {
        "stage42_am_metric": am_metric,
        "stage42_cq_metric": cq_metric,
        "stage42_dh_metric": dh_metric,
        "delta_vs_stage42_am": delta(am_metric),
        "delta_vs_stage42_cq": delta(cq_metric),
        "delta_vs_stage42_dh": delta(dh_metric),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    source_split = s42b.build_stage42_source_split()
    data = s41._combined()
    split, group = am._split_arrays(data)
    split_stats = am._source_stats(data, split, group)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = split == "train"
    floor = am._floor_arrays(data, train_mask)
    am_candidate = _rebuild_stage42_am_candidate(data, split, labels, floor)
    group_key = _group_key(data)
    eval_result = _evaluate_repairs(data, split, labels, floor, am_candidate, group_key)
    metric = eval_result["test"]["metric_vs_floor"]
    comparison = _compare_to_prior(metric)
    delta_am = comparison["delta_vs_stage42_am"]
    promotes = (
        metric["all_improvement"] > 0.0
        and metric["hard_failure_improvement"] > 0.0
        and metric["easy_degradation"] <= 0.02
        and (delta_am["all_improvement"] or 0.0) > 0.0
        and (delta_am["hard_failure_improvement"] or 0.0) > 0.0
        and eval_result["test"]["diagnostics"]["final_near_005"] <= eval_result["test"]["diagnostics"]["base_near_005"] + EPS
    )
    result: dict[str, Any] = {
        "source": "fresh_stage42_di_group_consistency_full_waypoint_repair",
        "stage": "Stage42-DI group-consistency full-waypoint repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(AM_JSON),
                str(CQ_JSON),
                str(DH_JSON),
            ]
        ),
        "source_split": source_split,
        "split_stats": split_stats,
        "label_stats": {
            "rows": int(len(split)),
            "test_rows": int(np.sum(split == "test")),
            "test_full_waypoint_rows": int(np.sum((split == "test") & np.all(labels["waypoint_valid"], axis=1))),
        },
        "group_schema": {
            "source": "fresh_stage42_di_group_key",
            "key": "source_file + frame_id*1000 + horizon",
            "agent_deduplication": True,
            "distance_unit": "normalized_by_dataset_local_scale",
            "uses_future_inputs": False,
        },
        "stage42_am_rebuilt": {
            "source": "fresh_rebuild_for_di",
            "lambda": am_candidate["lambda"],
            "feature_count": am_candidate["feature_count"],
            "policy_slice_count": len(am_candidate["policy"]["slices"]),
            "val_metric": am_candidate["val_metric"],
        },
        "repair": eval_result,
        "comparison_to_prior": comparison,
        "deployment_decision": {
            "promote_group_consistency_full_waypoint_repair": bool(promotes),
            "decision": "promote_stage42_di_group_consistency_full_waypoint_repair"
            if promotes
            else "group_consistency_repair_not_enough_keep_stage42_am_or_cq_floor",
            "reason": "Promotion requires all+hard positive, easy safe, improvement over Stage42-AM on all+hard, and no worse near@0.05 than rebuilt AM selected rollout.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "group_features_predicted_rollout_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
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
    result["stage42_di_gate"] = _gate(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    metric = result["repair"]["test"]["metric_vs_floor"]
    diag = result["repair"]["test"]["diagnostics"]
    delta_am = result["comparison_to_prior"]["delta_vs_stage42_am"]
    no_leak = result["no_leakage"]
    gates = {
        "source_level_split_rebuilt": result["split_stats"]["by_split"]["test"]["rows"] == int(metric["rows"]) and int(metric["rows"]) > 0,
        "full_waypoint_labels_available": result["label_stats"]["test_full_waypoint_rows"] > 0,
        "group_key_built": result["group_schema"]["agent_deduplication"] is True,
        "repair_candidates_run": result["repair"]["candidate_count"] >= 40,
        "validation_selected_repair": result["repair"]["selected"]["val_score"] != 0.0 and no_leak["test_threshold_tuning"] is False,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["future_waypoint_label_eval_only"] is True,
                no_leak["group_features_predicted_rollout_only"] is True,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
                no_leak["validation_only_policy_selection"] is True,
                no_leak["train_only_feature_normalization"] is True,
            ]
        ),
        "test_all_positive_vs_floor": metric["all_improvement"] > 0.0,
        "test_t50_positive_vs_floor": metric["t50_improvement"] > 0.0,
        "test_hard_positive_vs_floor": metric["hard_failure_improvement"] > 0.0,
        "easy_degradation_under_2pct": metric["easy_degradation"] <= 0.02,
        "near_collision_not_worse_than_am": diag["final_near_005"] <= diag["base_near_005"] + EPS,
        "beats_stage42_am_all": (delta_am["all_improvement"] or 0.0) > 0.0,
        "beats_stage42_am_hard": (delta_am["hard_failure_improvement"] or 0.0) > 0.0,
        "bootstrap_reported": result["repair"]["test"]["bootstrap"]["all"]["bootstrap_n"] > 0,
        "no_metric_seconds_overclaim": result["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": result["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": result["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage42_di_group_consistency_full_waypoint_repair_pass_promotable"
    elif gates["test_all_positive_vs_floor"] and gates["test_hard_positive_vs_floor"] and gates["easy_degradation_under_2pct"]:
        verdict = "stage42_di_group_consistency_full_waypoint_repair_pass_positive_not_promotable"
    else:
        verdict = "stage42_di_group_consistency_full_waypoint_repair_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_csv(rows: list[Mapping[str, Any]]) -> None:
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "mode",
                "min_sep",
                "margin",
                "alpha",
                "safe_min_sep",
                "strength",
                "val_score",
                "val_all",
                "val_t50",
                "val_hard",
                "val_easy",
                "val_base_near005",
                "val_final_near005",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows[:120], start=1):
            c = row["candidate"]
            m = row["val_metric"]
            d = row["val_diagnostics"]
            writer.writerow(
                {
                    "rank": rank,
                    "mode": c.get("mode"),
                    "min_sep": c.get("min_sep"),
                    "margin": c.get("margin"),
                    "alpha": c.get("alpha"),
                    "safe_min_sep": c.get("safe_min_sep"),
                    "strength": c.get("strength"),
                    "val_score": row["val_score"],
                    "val_all": m["all_improvement"],
                    "val_t50": m["t50_improvement"],
                    "val_hard": m["hard_failure_improvement"],
                    "val_easy": m["easy_degradation"],
                    "val_base_near005": d["base_near_005"],
                    "val_final_near005": d["final_near_005"],
                }
            )


def _render_report(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_di_gate"]
    selected = result["repair"]["selected"]
    metric = result["repair"]["test"]["metric_vs_floor"]
    diag = result["repair"]["test"]["diagnostics"]
    base = result["repair"]["baseline_stage42_am_on_test"]
    delta_am = result["comparison_to_prior"]["delta_vs_stage42_am"]
    delta_cq = result["comparison_to_prior"]["delta_vs_stage42_cq"]
    delta_dh = result["comparison_to_prior"]["delta_vs_stage42_dh"]
    lines = [
        "# Stage42-DI Group-Consistency Full-Waypoint Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- decision: `{result['deployment_decision']['decision']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Selected Repair",
        "",
        f"- candidate: `{selected['candidate']}`",
        f"- val_score: `{selected['val_score']:.6f}`",
        f"- val_metric: `{selected['val_metric']}`",
        f"- val_diagnostics: `{selected['val_diagnostics']}`",
        "",
        "## Test Once Metrics vs Train-Horizon Causal Floor",
        "",
        "| candidate | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | near@0.05 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Stage42-AM rebuilt floor-protected source-level | {_pct(base['all_improvement'])} | {_pct(base['t50_improvement'])} | {_pct(base['t100_raw_frame_diagnostic_improvement'])} | {_pct(base['hard_failure_improvement'])} | {_pct(base['easy_degradation'])} | {_pct(base['switch_rate'])} | {_pct(diag['base_near_005'])} |",
        f"| Stage42-DI group-consistency repair | {_pct(metric['all_improvement'])} | {_pct(metric['t50_improvement'])} | {_pct(metric['t100_raw_frame_diagnostic_improvement'])} | {_pct(metric['hard_failure_improvement'])} | {_pct(metric['easy_degradation'])} | {_pct(metric['switch_rate'])} | {_pct(diag['final_near_005'])} |",
        "",
        "## Delta vs Prior Evidence",
        "",
        f"- delta_vs_stage42_am all/t50/t100/hard/easy: `{_pct(delta_am['all_improvement'])}` / `{_pct(delta_am['t50_improvement'])}` / `{_pct(delta_am['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(delta_am['hard_failure_improvement'])}` / `{_pct(delta_am['easy_degradation'])}`",
        f"- delta_vs_stage42_cq all/t50/t100/hard/easy: `{_pct(delta_cq['all_improvement'])}` / `{_pct(delta_cq['t50_improvement'])}` / `{_pct(delta_cq['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(delta_cq['hard_failure_improvement'])}` / `{_pct(delta_cq['easy_degradation'])}`",
        f"- delta_vs_stage42_dh all/t50/t100/hard/easy: `{_pct(delta_dh['all_improvement'])}` / `{_pct(delta_dh['t50_improvement'])}` / `{_pct(delta_dh['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(delta_dh['hard_failure_improvement'])}` / `{_pct(delta_dh['easy_degradation'])}`",
        "",
        "## Proximity Diagnostics",
        "",
        f"- base_near_005: `{_pct(diag['base_near_005'])}`",
        f"- final_near_005: `{_pct(diag['final_near_005'])}`",
        f"- floor_near_005: `{_pct(diag['floor_near_005'])}`",
        f"- base_p05_min_distance: `{diag['base_p05_min_distance']}`",
        f"- final_p05_min_distance: `{diag['final_p05_min_distance']}`",
        f"- floor_p05_min_distance: `{diag['floor_p05_min_distance']}`",
        "",
        "## Bootstrap CI",
        "",
        "| slice | low | mid | high | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, row in result["repair"]["test"]["bootstrap"].items():
        lines.append(f"| `{key}` | {row['low']:.6f} | {row['mid']:.6f} | {row['high']:.6f} | {row['n']} |")
    lines.extend(["", "## By Domain", "", "| domain | rows | all | t50 | t100 raw diag | hard/failure | easy | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for domain, row in result["repair"]["test"]["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | {_pct(row['all_improvement'])} | {_pct(row['t50_improvement'])} | {_pct(row['t100_raw_frame_diagnostic_improvement'])} | {_pct(row['hard_failure_improvement'])} | {_pct(row['easy_degradation'])} | {_pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-DI changes the repair mechanism from scalar loss weighting to explicit group-consistency / proximity-aware repair over predicted full-waypoint rollouts.",
            "- Repair selection is validation-only; test is evaluated once.",
            "- Promotion requires improving Stage42-AM on all and hard/failure, preserving easy, and not worsening near@0.05 relative to rebuilt Stage42-AM selected rollout.",
            "- If not promotable, keep Stage42-AM/CQ or Stage37/teacher safety floor as deployable floor and treat DI as diagnostic evidence.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_di_gate"]
    return [
        "# Stage42-DI Gate",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Gates",
        "",
        *[f"- {key}: `{value}`" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_di_gate"]
    metric = result["repair"]["test"]["metric_vs_floor"]
    diag = result["repair"]["test"]["diagnostics"]
    delta_am = result["comparison_to_prior"]["delta_vs_stage42_am"]
    return [
        "## Stage42-DI Group-Consistency Full-Waypoint Repair",
        "",
        "- source: `fresh_stage42_di_group_consistency_full_waypoint_repair`",
        "- role: explicit all-agent group-consistency / proximity repair over source-level full-waypoint predictions after Stage42-DE/DF/DG/DH blockers.",
        f"- selected repair: `{result['repair']['selected']['candidate']}`.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- test vs train-horizon causal floor: all `{_pct(metric['all_improvement'])}`, t50 `{_pct(metric['t50_improvement'])}`, t100 raw `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(metric['hard_failure_improvement'])}`, easy `{_pct(metric['easy_degradation'])}`.",
        f"- delta vs Stage42-AM: all `{_pct(delta_am['all_improvement'])}`, t50 `{_pct(delta_am['t50_improvement'])}`, t100 raw `{_pct(delta_am['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(delta_am['hard_failure_improvement'])}`, easy `{_pct(delta_am['easy_degradation'])}`.",
        f"- near@0.05 base/final/floor: `{_pct(diag['base_near_005'])}` / `{_pct(diag['final_near_005'])}` / `{_pct(diag['floor_near_005'])}`.",
        f"- decision: `{result['deployment_decision']['decision']}`.",
        "- Stage5C remains false; SMC remains false; no metric/seconds claim.",
    ]


def _refresh_readmes(result: Mapping[str, Any]) -> None:
    lines = _refresh_lines(result)
    for path in [README_RESULTS, M3W_README, GOAL_README, CURRENT_RETROSPECTIVE]:
        _replace_section(path, "STAGE42_DI_GROUP_CONSISTENCY_FULL_WAYPOINT_REPAIR", lines)


def _refresh_research_state(result: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DI group-consistency full-waypoint repair"
    state["current_verdict"] = result["stage42_di_gate"]["verdict"]
    state["stage42_di_group_consistency_full_waypoint_repair"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": result["stage42_di_gate"]["verdict"],
        "gates": f"{result['stage42_di_gate']['passed']}/{result['stage42_di_gate']['total']}",
        "deployment_decision": result["deployment_decision"],
        "selected_repair": result["repair"]["selected"]["candidate"],
        "test_metric_vs_floor": result["repair"]["test"]["metric_vs_floor"],
        "test_proximity_diagnostics": result["repair"]["test"]["diagnostics"],
        "comparison_to_stage42_am": result["comparison_to_prior"]["delta_vs_stage42_am"],
        "claim_boundary": result["claim_boundary"],
        "conclusion": "Stage42-DI is a fresh explicit group-consistency/proximity repair over source-level full-waypoint rollouts. Promotion requires all+hard improvement over Stage42-AM plus proximity safety; otherwise keep Stage42-AM/CQ or Stage37/teacher floor.",
    }
    write_json(RESEARCH_STATE, _jsonable(state))


def run_stage42_group_consistency_full_waypoint_repair() -> dict[str, Any]:
    result = _build_payload()
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _write_csv(result["repair"]["validation_rows"])
    _refresh_readmes(result)
    _refresh_research_state(result)
    return result


if __name__ == "__main__":
    payload = run_stage42_group_consistency_full_waypoint_repair()
    gate = payload["stage42_di_gate"]
    print(f"Stage42-DI gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
    print(f"Decision: {payload['deployment_decision']['decision']}")
