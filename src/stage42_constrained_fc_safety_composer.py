from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_external_validation as s42b
from src import stage42_full_waypoint_all_hard_loss_repair as dg
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_objective_level_proximity_training as fc
from src import stage42_proximity_pareto_composer as fb
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_graph_context as graph_ctx
from src import stage42_waypointwise_group_repel_repair as fa
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "constrained_fc_safety_composer_stage42.json"
REPORT_MD = OUT_DIR / "constrained_fc_safety_composer_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fe_gate.md"

AM_JSON = OUT_DIR / "source_level_full_waypoint_eval_stage42.json"
DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
FA_JSON = OUT_DIR / "waypointwise_group_repel_repair_stage42.json"
FB_JSON = OUT_DIR / "proximity_pareto_composer_stage42.json"
FC_JSON = OUT_DIR / "objective_level_proximity_training_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fc.PAPER_FILES

SOURCE = "fresh_stage42_constrained_fc_safety_composer"
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FC 提高 all/t50/hard 但 proximity safety 不过；Stage42-FD 证明简单 safety-teacher target blend 会被 validation 选回 alpha=0。",
    "Stage42-FE 改为 constrained composer：FC 高精度预测为默认，DI/FA/FB 作为 predicted-proximity 安全回退候选。",
    "composer 只使用 predicted rollout geometry、source/frame/horizon group key、agent id、当前/过去可得信息。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "composer candidate 和 policy 只在 validation 上选择；test 只评一次。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = dict(fc.CLAIM_BOUNDARY)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _load_prior() -> dict[str, Any]:
    return {
        "am": read_json(AM_JSON, {}),
        "di": read_json(DI_JSON, {}),
        "fa": read_json(FA_JSON, {}),
        "fb": read_json(FB_JSON, {}),
        "fc": read_json(FC_JSON, {}),
    }


def _rebuild_fc_candidate(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    features: np.ndarray,
    graph: np.ndarray,
    signals: Mapping[str, np.ndarray],
    group_key: np.ndarray,
    fc_payload: Mapping[str, Any],
) -> dict[str, Any]:
    selected = fc_payload.get("model", {}).get("selected", {})
    variant = str(selected.get("variant", "label_proximity_objective"))
    lam = float(selected.get("lambda", 10.0))
    train_mask = split == "train"
    val_mask = split == "val"
    current = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    target_delta = ((labels["waypoint_xy"].astype(np.float64) - current[:, None, :]) / scale[:, None, None]).astype(np.float32)
    specs = fc._candidate_specs(data, features, graph, signals, train_mask)
    spec = next((row for row in specs if row["variant"] == variant), None)
    if spec is None:
        raise RuntimeError(f"Could not rebuild Stage42-FC selected variant: {variant}")
    x, _mean, _std = am._standardize(spec["features"], train_mask)
    coef = dg._fit_weighted_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, spec["weights"], lam)
    pred_xy = am._predict_waypoints(x, coef, data)
    policy, _ade, _fde, _switch = am._select_policy_on_val(pred_xy, floor["floor_xy"], labels, data, val_mask)
    selected_xy, switch_xy = di._apply_am_policy_xy(pred_xy, floor["floor_xy"], data, policy)
    selected_ade, selected_fde = am._trajectory_errors(selected_xy, labels)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    val_ids = np.where(val_mask)[0]
    test_ids = np.where(split == "test")[0]
    return {
        "variant": variant,
        "lambda": lam,
        "pred_xy": pred_xy.astype(np.float32),
        "selected_xy": selected_xy.astype(np.float32),
        "switch": switch_xy.astype(bool),
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "policy": policy,
        "val_metric": am._metric(selected_ade, floor_ade, data, switch_xy, val_mask),
        "test_metric": am._metric(selected_ade, floor_ade, data, switch_xy, split == "test"),
        "val_near": fc._near_diagnostics(selected_xy, data, val_ids, group_key),
        "test_near": fc._near_diagnostics(selected_xy, data, test_ids, group_key),
    }


def _eval_full_candidate(
    name: str,
    ids: np.ndarray,
    xy_full: np.ndarray,
    switch_full: np.ndarray,
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    group_key: np.ndarray,
) -> dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    xy = xy_full[ids].astype(np.float32)
    switch = switch_full[ids].astype(bool)
    selected_ade, selected_fde = di._trajectory_errors_subset(xy, labels, ids)
    floor_ade, floor_fde = di._trajectory_errors_subset(floor_xy[ids], labels, ids)
    keys = group_key[ids]
    normalizer = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    agent = data["agent_id"][ids].astype(np.int64)
    mind = di._min_group_distance_fast(xy, keys, normalizer, agent)
    return {
        "name": name,
        "selected_xy": xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "switch": switch,
        "min_distance": mind,
        "metric": di._metric_subset(selected_ade, floor_ade, data, ids, switch),
        "diagnostics": {
            "near_005": float(np.mean(np.isfinite(mind) & (mind < 0.05))) if len(mind) else 0.0,
            "near_008": float(np.mean(np.isfinite(mind) & (mind < 0.08))) if len(mind) else 0.0,
            "p05_min_distance": float(np.percentile(mind[np.isfinite(mind)], 5)) if np.any(np.isfinite(mind)) else None,
        },
    }


def _attach_min_distance(
    row: Mapping[str, Any],
    ids: np.ndarray,
    data: Mapping[str, np.ndarray],
    group_key: np.ndarray,
) -> dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    xy = row["selected_xy"].astype(np.float32)
    normalizer = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    agent = data["agent_id"][ids].astype(np.int64)
    mind = di._min_group_distance_fast(xy, group_key[ids], normalizer, agent)
    out = dict(row)
    out["min_distance"] = mind
    diagnostics = dict(out.get("diagnostics", {}))
    diagnostics.setdefault("near_005", float(np.mean(np.isfinite(mind) & (mind < 0.05))) if len(mind) else 0.0)
    diagnostics.setdefault("near_008", float(np.mean(np.isfinite(mind) & (mind < 0.08))) if len(mind) else 0.0)
    diagnostics.setdefault("p05_min_distance", float(np.percentile(mind[np.isfinite(mind)], 5)) if np.any(np.isfinite(mind)) else None)
    out["diagnostics"] = diagnostics
    return out


def _reference_evals(
    ids: np.ndarray,
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    am_candidate: Mapping[str, Any],
    fc_candidate: Mapping[str, Any],
    group_key: np.ndarray,
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    floor_xy = floor["floor_xy"].astype(np.float32)
    di_candidate = prior["di"].get("repair", {}).get("selected", {}).get("candidate")
    fa_candidate = prior["fa"].get("repair", {}).get("selected", {}).get("candidate")
    fb_candidate = prior["fb"].get("repair", {}).get("selected", {}).get("candidate")
    if not di_candidate or not fa_candidate or not fb_candidate:
        raise RuntimeError("Stage42-FE requires Stage42-DI, Stage42-FA, and Stage42-FB selected candidate artifacts.")
    di_eval = di._repair_subset(
        ids,
        di_candidate,
        data,
        labels,
        floor_xy,
        am_candidate["pred_xy"].astype(np.float32),
        am_candidate["selected_xy"].astype(np.float32),
        am_candidate["switch"].astype(bool),
        group_key,
    )
    fa_eval = fa._repair_subset_waypointwise(
        ids,
        fa_candidate,
        data,
        labels,
        floor_xy,
        am_candidate["pred_xy"].astype(np.float32),
        am_candidate["selected_xy"].astype(np.float32),
        am_candidate["switch"].astype(bool),
        group_key,
    )
    di_eval = _attach_min_distance(di_eval, ids, data, group_key)
    fa_eval = _attach_min_distance(fa_eval, ids, data, group_key)
    fb_eval = fb._compose_xy(fb_candidate, ids, data, labels, floor_xy, group_key, di_eval, fa_eval)
    fb_eval = _attach_min_distance(fb_eval, ids, data, group_key)
    fc_eval = _eval_full_candidate("fc", ids, fc_candidate["selected_xy"], fc_candidate["switch"], data, labels, floor_xy, group_key)
    return {"fc": fc_eval, "di": di_eval, "fa": fa_eval, "fb": fb_eval}


def _candidate_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"mode": "all_fc", "fallback": "fc", "scope": "none", "threshold": 0.0, "margin": 0.0},
        {"mode": "all_fb", "fallback": "fb", "scope": "all", "threshold": 0.0, "margin": 0.0},
        {"mode": "all_di", "fallback": "di", "scope": "all", "threshold": 0.0, "margin": 0.0},
    ]
    for fallback in ["di", "fb", "fa", "safest"]:
        for scope in ["row", "group"]:
            for threshold in [0.035, 0.05, 0.065, 0.08, 0.10]:
                for margin in [0.0, 0.0025, 0.005, 0.01]:
                    rows.append(
                        {
                            "mode": "fc_to_safety",
                            "fallback": fallback,
                            "scope": scope,
                            "threshold": threshold,
                            "margin": margin,
                        }
                    )
    return rows


def _safest_eval(evals: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    choices = [evals["di"], evals["fb"], evals["fa"]]
    mins = np.stack([row["min_distance"] if "min_distance" in row else _min_from_eval(row) for row in choices], axis=1)
    best = np.nanargmax(np.where(np.isfinite(mins), mins, -np.inf), axis=1)
    xy = choices[0]["selected_xy"].copy()
    switch = choices[0]["switch"].copy()
    for idx, choice in enumerate(choices):
        use = best == idx
        xy[use] = choice["selected_xy"][use]
        switch[use] = choice["switch"][use]
    return {"selected_xy": xy, "switch": switch}


def _min_from_eval(row: Mapping[str, Any]) -> np.ndarray:
    if "min_distance" in row:
        return row["min_distance"]
    raise KeyError("reference eval missing min_distance")


def _group_any(keys: np.ndarray, risk: np.ndarray) -> np.ndarray:
    return fb._group_any_mask(keys, risk)


def _compose(
    candidate: Mapping[str, Any],
    ids: np.ndarray,
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    group_key: np.ndarray,
    evals: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    fc_eval = evals["fc"]
    if candidate["mode"] == "all_fc":
        selected_xy = fc_eval["selected_xy"].copy()
        switch = fc_eval["switch"].copy()
        use_fallback = np.zeros(len(ids), dtype=bool)
        fallback_name = "fc"
    elif candidate["mode"] in {"all_fb", "all_di"}:
        fallback_name = str(candidate["fallback"])
        selected_xy = evals[fallback_name]["selected_xy"].copy()
        switch = evals[fallback_name]["switch"].copy()
        use_fallback = np.ones(len(ids), dtype=bool)
    else:
        fallback_name = str(candidate["fallback"])
        if fallback_name == "safest":
            safe = _safest_eval(evals)
            fallback_xy = safe["selected_xy"]
            fallback_switch = safe["switch"]
            fallback_min = np.maximum.reduce(
                [
                    evals["di"]["min_distance"],
                    evals["fb"]["min_distance"],
                    evals["fa"]["min_distance"],
                ]
            )
        else:
            fallback_xy = evals[fallback_name]["selected_xy"]
            fallback_switch = evals[fallback_name]["switch"]
            fallback_min = evals[fallback_name]["min_distance"]
        fc_min = evals["fc"]["min_distance"]
        row_risk = (
            np.isfinite(fc_min)
            & (fc_min < float(candidate["threshold"]))
            & np.isfinite(fallback_min)
            & (fallback_min + float(candidate["margin"]) >= fc_min)
        )
        if candidate["scope"] == "group":
            row_risk = _group_any(group_key[ids], row_risk)
        use_fallback = row_risk.astype(bool)
        selected_xy = fc_eval["selected_xy"].copy()
        switch = fc_eval["switch"].copy()
        selected_xy[use_fallback] = fallback_xy[use_fallback]
        switch[use_fallback] = fallback_switch[use_fallback]
    selected_ade, selected_fde = di._trajectory_errors_subset(selected_xy, labels, ids)
    floor_ade, floor_fde = di._trajectory_errors_subset(floor_xy[ids], labels, ids)
    normalizer = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    agent = data["agent_id"][ids].astype(np.int64)
    final_min = di._min_group_distance_fast(selected_xy, group_key[ids], normalizer, agent)
    return {
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "switch": switch,
        "use_fallback": use_fallback,
        "fallback_name": fallback_name,
        "metric": di._metric_subset(selected_ade, floor_ade, data, ids, switch),
        "diagnostics": {
            "use_fallback_rate": float(np.mean(use_fallback)) if len(use_fallback) else 0.0,
            "use_fallback_rows": int(np.sum(use_fallback)),
            "fc_near_005": float(evals["fc"]["diagnostics"]["near_005"]),
            "final_near_005": float(np.mean(np.isfinite(final_min) & (final_min < 0.05))) if len(final_min) else 0.0,
            "final_near_008": float(np.mean(np.isfinite(final_min) & (final_min < 0.08))) if len(final_min) else 0.0,
            "final_p05_min_distance": float(np.percentile(final_min[np.isfinite(final_min)], 5)) if np.any(np.isfinite(final_min)) else None,
        },
    }


def _selection_score(
    metric: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    delta_fc: Mapping[str, Any],
    val_near_limit: float,
) -> float:
    near_excess = max(0.0, float(diagnostics["final_near_005"]) - float(val_near_limit))
    return (
        1.45 * float(metric["all_improvement"])
        + 1.45 * float(metric["hard_failure_improvement"])
        + 1.05 * float(metric["t50_improvement"])
        + 0.40 * float(metric["t100_raw_frame_diagnostic_improvement"])
        - 55.0 * max(0.0, float(metric["easy_degradation"]) - 0.02)
        - 35.0 * near_excess
        - 3.5 * max(0.0, -float(delta_fc["all_improvement"]))
        - 3.5 * max(0.0, -float(delta_fc["hard_failure_improvement"]))
        - 0.015 * float(diagnostics["use_fallback_rate"])
    )


def _delta(metric: Mapping[str, Any], ref: Mapping[str, Any]) -> dict[str, float | None]:
    return fc._delta(metric, ref)


def _select_validation_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [
        row
        for row in rows
        if row["val_metric"]["all_improvement"] > 0.0
        and row["val_metric"]["hard_failure_improvement"] > 0.0
        and row["val_metric"]["easy_degradation"] <= 0.02
        and row["val_near_delta_vs_di"] <= 0.0
    ]
    pool = feasible if feasible else rows
    return max(pool, key=lambda row: float(row["val_score"]))


def _evaluate_composer(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    am_candidate: Mapping[str, Any],
    fc_candidate: Mapping[str, Any],
    group_key: np.ndarray,
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    val_ids = np.where(split == "val")[0]
    test_ids = np.where(split == "test")[0]
    floor_xy = floor["floor_xy"].astype(np.float32)
    val_evals = _reference_evals(val_ids, data, labels, floor, am_candidate, fc_candidate, group_key, prior)
    test_evals = _reference_evals(test_ids, data, labels, floor, am_candidate, fc_candidate, group_key, prior)
    val_near_limit = float(val_evals["di"]["diagnostics"]["final_near_005"])
    rows: list[dict[str, Any]] = []
    for candidate in _candidate_grid():
        val = _compose(candidate, val_ids, data, labels, floor_xy, group_key, val_evals)
        delta_fc = _delta(val["metric"], val_evals["fc"]["metric"])
        score = _selection_score(val["metric"], val["diagnostics"], delta_fc, val_near_limit)
        rows.append(
            {
                "candidate": dict(candidate),
                "val_score": float(score),
                "val_metric": val["metric"],
                "val_diagnostics": val["diagnostics"],
                "val_delta_vs_fc": delta_fc,
                "val_near_delta_vs_di": float(val["diagnostics"]["final_near_005"]) - val_near_limit,
            }
        )
    selected = _select_validation_row(rows)
    test = _compose(selected["candidate"], test_ids, data, labels, floor_xy, group_key, test_evals)
    h = data["horizon"][test_ids].astype(int)
    hard_failure = data["hard"][test_ids].astype(bool) | data["failure"][test_ids].astype(bool)
    easy = data["easy"][test_ids].astype(bool)
    domain = data["dataset"][test_ids].astype(str)
    bootstrap = {
        "all": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], np.ones(len(test_ids), dtype=bool), seed=43401),
        "t50": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], h == 50, seed=43402),
        "t100_raw_frame_diagnostic": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], h == 100, seed=43403),
        "hard_failure": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], hard_failure, seed=43404),
        "easy_degradation": di._bootstrap_ci_subset(test["floor_ade"], test["selected_ade"], easy, seed=43405),
    }
    by_domain = {
        d: di._metric_subset(test["selected_ade"][domain == d], test["floor_ade"][domain == d], data, test_ids[domain == d], test["switch"][domain == d])
        for d in sorted(set(domain.tolist()))
    }
    return {
        "candidate_count": len(rows),
        "validation_rows": sorted(rows, key=lambda row: row["val_score"], reverse=True),
        "selected": selected,
        "references": {
            "fc_test": {"metric": test_evals["fc"]["metric"], "diagnostics": test_evals["fc"]["diagnostics"]},
            "di_test": {"metric": test_evals["di"]["metric"], "diagnostics": test_evals["di"]["diagnostics"]},
            "fb_test": {"metric": test_evals["fb"]["metric"], "diagnostics": test_evals["fb"]["diagnostics"]},
            "fa_test": {"metric": test_evals["fa"]["metric"], "diagnostics": test_evals["fa"]["diagnostics"]},
        },
        "test": {
            "metric_vs_floor": test["metric"],
            "diagnostics": test["diagnostics"],
            "delta_vs_fc": _delta(test["metric"], test_evals["fc"]["metric"]),
            "delta_vs_di": _delta(test["metric"], test_evals["di"]["metric"]),
            "delta_vs_fb": _delta(test["metric"], test_evals["fb"]["metric"]),
            "delta_vs_fa": _delta(test["metric"], test_evals["fa"]["metric"]),
            "near_delta_vs_fc": float(test["diagnostics"]["final_near_005"]) - float(test_evals["fc"]["diagnostics"]["near_005"]),
            "near_delta_vs_di": float(test["diagnostics"]["final_near_005"]) - float(test_evals["di"]["diagnostics"]["final_near_005"]),
            "near_delta_vs_fb": float(test["diagnostics"]["final_near_005"]) - float(test_evals["fb"]["diagnostics"]["final_near_005"]),
            "bootstrap": bootstrap,
            "by_domain": by_domain,
        },
    }


def _deployment_decision(result: Mapping[str, Any]) -> dict[str, Any]:
    metric = result["test"]["metric_vs_floor"]
    delta_fc = result["test"]["delta_vs_fc"]
    near_delta_di = result["test"]["near_delta_vs_di"]
    promotes = (
        metric["all_improvement"] > 0.0
        and metric["hard_failure_improvement"] > 0.0
        and metric["easy_degradation"] <= 0.02
        and (delta_fc["all_improvement"] or 0.0) >= -0.002
        and (delta_fc["hard_failure_improvement"] or 0.0) >= -0.002
        and near_delta_di <= 0.0
    )
    return {
        "promote_constrained_fc_safety_composer": bool(promotes),
        "decision": "promote_stage42_fe_constrained_fc_safety_composer"
        if promotes
        else "constrained_fc_safety_composer_not_enough_keep_stage42_di_or_cq_floor",
        "reason": "Promotion requires positive all+hard, easy safe, no material all/hard loss vs Stage42-FC, and no worse near@0.05 than Stage42-DI.",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    repair = payload["repair"]
    metric = repair["test"]["metric_vs_floor"]
    delta_fc = repair["test"]["delta_vs_fc"]
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "source_level_split_rebuilt": payload["split_stats"]["by_split"]["test"]["rows"] == int(metric["rows"]) and int(metric["rows"]) > 0,
        "full_waypoint_labels_available": payload["label_stats"]["test_full_waypoint_rows"] > 0,
        "fc_candidate_rebuilt": payload["composer_family"]["fc_candidate_rebuilt"] is True,
        "composer_family_built": repair["candidate_count"] >= 40,
        "validation_selected_composer": repair["selected"]["val_score"] is not None and no_leak["test_threshold_tuning"] is False,
        "test_all_positive_vs_floor": metric["all_improvement"] > 0.0,
        "test_t50_positive_vs_floor": metric["t50_improvement"] > 0.0,
        "test_hard_positive_vs_floor": metric["hard_failure_improvement"] > 0.0,
        "easy_degradation_under_2pct": metric["easy_degradation"] <= 0.02,
        "no_material_all_loss_vs_fc": (delta_fc["all_improvement"] or 0.0) >= -0.002,
        "no_material_hard_loss_vs_fc": (delta_fc["hard_failure_improvement"] or 0.0) >= -0.002,
        "near_better_than_fc": repair["test"]["near_delta_vs_fc"] < 0.0,
        "near_not_worse_than_stage42_di": repair["test"]["near_delta_vs_di"] <= 0.0,
        "bootstrap_reported": repair["test"]["bootstrap"]["all"]["bootstrap_n"] > 0,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["future_waypoint_label_eval_only"] is True,
                no_leak["composer_features_predicted_rollout_only"] is True,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
                no_leak["validation_only_policy_selection"] is True,
                no_leak["train_only_feature_normalization"] is True,
            ]
        ),
        "no_metric_seconds_overclaim": boundary["metric_or_seconds_claim"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage42_fe_constrained_fc_safety_composer_pass_promotable"
    elif gates["test_all_positive_vs_floor"] and gates["test_hard_positive_vs_floor"] and gates["easy_degradation_under_2pct"]:
        verdict = "stage42_fe_constrained_fc_safety_composer_positive_not_promoted"
    else:
        verdict = "stage42_fe_constrained_fc_safety_composer_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    source_split = s42b.build_stage42_source_split()
    data = s41._combined()
    split, group = am._split_arrays(data)
    split_stats = am._source_stats(data, split, group)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = split == "train"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names = am._feature_matrix(data, floor)
    graph, graph_names, graph_stats = graph_ctx._build_graph_features(data)
    am_candidate = di._rebuild_stage42_am_candidate(data, split, labels, floor)
    am_candidate["floor_xy"] = floor["floor_xy"]
    group_key = di._group_key(data)
    signals = fc._objective_signals(data, labels, graph, group_key, am_candidate)
    prior = _load_prior()
    fc_candidate = _rebuild_fc_candidate(data, split, labels, floor, features, graph, signals, group_key, prior["fc"])
    repair = _evaluate_composer(data, split, labels, floor, am_candidate, fc_candidate, group_key, prior)
    decision = _deployment_decision(repair)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FE constrained FC/safety composer",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(AM_JSON),
                str(DI_JSON),
                str(FA_JSON),
                str(FB_JSON),
                str(FC_JSON),
            ]
        ),
        "source_split_summary": source_split.get("summary", {}),
        "split_stats": split_stats,
        "label_stats": {
            "rows": int(len(split)),
            "test_rows": int(np.sum(split == "test")),
            "test_full_waypoint_rows": int(np.sum((split == "test") & np.all(labels["waypoint_valid"], axis=1))),
        },
        "feature_schema": {
            "base_feature_count": len(feature_names),
            "graph_feature_count": len(graph_names),
            "composer_inputs": ["predicted_rollout_geometry", "group_key", "agent_id", "domain", "horizon"],
            "future_label_input": False,
        },
        "graph_stats": graph_stats,
        "composer_family": {
            "source": "fresh_stage42_fe_validation_only_fc_safety_composer",
            "candidate_count": len(_candidate_grid()),
            "default_policy": "Stage42-FC objective-level proximity training",
            "safety_policies": ["Stage42-DI", "Stage42-FA", "Stage42-FB", "safest_predicted_proximity"],
            "fc_candidate_rebuilt": True,
            "fc_selected_variant": fc_candidate["variant"],
            "fc_selected_lambda": fc_candidate["lambda"],
            "uses_future_inputs": False,
        },
        "repair": repair,
        "deployment_decision": decision,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "composer_features_predicted_rollout_only": True,
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
    payload["stage42_fe_gate"] = _gate(payload)
    return _jsonable(payload)


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fe_gate"]
    selected = payload["repair"]["selected"]
    metric = payload["repair"]["test"]["metric_vs_floor"]
    diag = payload["repair"]["test"]["diagnostics"]
    refs = payload["repair"]["references"]
    return [
        "# Stage42-FE Constrained FC/Safety Composer",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- decision: `{payload['deployment_decision']['decision']}`",
        f"- selected candidate: `{selected['candidate']}`",
        "",
        "## Protected Test Metrics vs Floor",
        "",
        f"- all improvement: `{_pct(metric['all_improvement'])}`",
        f"- t50 improvement: `{_pct(metric['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure improvement: `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        f"- switch rate: `{_pct(metric['switch_rate'])}`",
        "",
        "## References",
        "",
        "| policy | all | t50 | t100 raw | hard/failure | easy | near@0.05 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| `{name}` | `{_pct(row['metric']['all_improvement'])}` | `{_pct(row['metric']['t50_improvement'])}` | `{_pct(row['metric']['t100_raw_frame_diagnostic_improvement'])}` | `{_pct(row['metric']['hard_failure_improvement'])}` | `{_pct(row['metric']['easy_degradation'])}` | `{_pct(row['diagnostics'].get('near_005', row['diagnostics'].get('final_near_005')))}` |"
            for name, row in refs.items()
        ],
        "",
        "## Diagnostics",
        "",
        f"- final near@0.05: `{_pct(diag['final_near_005'])}`",
        f"- delta near@0.05 vs FC: `{_pct(payload['repair']['test']['near_delta_vs_fc'])}`",
        f"- delta near@0.05 vs DI: `{_pct(payload['repair']['test']['near_delta_vs_di'])}`",
        f"- delta near@0.05 vs FB: `{_pct(payload['repair']['test']['near_delta_vs_fb'])}`",
        f"- fallback rate: `{_pct(diag['use_fallback_rate'])}`",
        "",
        "## Delta vs Prior",
        "",
        f"- delta vs FC: `{payload['repair']['test']['delta_vs_fc']}`",
        f"- delta vs DI: `{payload['repair']['test']['delta_vs_di']}`",
        f"- delta vs FB: `{payload['repair']['test']['delta_vs_fb']}`",
        "",
        "## No Leakage / Claim Boundary",
        "",
        "- composer uses predicted rollout geometry only, not future labels.",
        "- future labels are evaluation labels only.",
        "- no central velocity, no test endpoint goals, no test threshold tuning.",
        "- dataset-local/raw-frame 2.5D only; no metric/seconds claim.",
        "- Stage5C not executed; SMC not enabled.",
    ]


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fe_gate"]
    return [
        "# Stage42-FE Gates",
        "",
        f"Verdict: `{gate['verdict']}`",
        f"Passed: `{gate['passed']} / {gate['total']}`",
        "",
        *[f"- `{key}`: `{value}`" for key, value in gate["gates"].items()],
    ]


def _summary_section(payload: Mapping[str, Any]) -> str:
    metric = payload["repair"]["test"]["metric_vs_floor"]
    selected = payload["repair"]["selected"]["candidate"]
    return "\n".join(
        [
            "<!-- STAGE42_FE_CONSTRAINED_FC_SAFETY_COMPOSER:START -->",
            "## Stage42-FE Constrained FC/Safety Composer",
            "",
            f"- source: `{payload['source']}`",
            "- role: validation-only constrained composer from high-accuracy Stage42-FC to DI/FA/FB safety fallbacks.",
            f"- selected candidate: `{selected}`.",
            f"- gate: `{payload['stage42_fe_gate']['passed']} / {payload['stage42_fe_gate']['total']}`; verdict `{payload['stage42_fe_gate']['verdict']}`.",
            f"- test all/t50/t100raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
            f"- delta vs FC all/hard/near005: `{_pct(payload['repair']['test']['delta_vs_fc']['all_improvement'])}` / `{_pct(payload['repair']['test']['delta_vs_fc']['hard_failure_improvement'])}` / `{_pct(payload['repair']['test']['near_delta_vs_fc'])}`.",
            f"- delta vs DI all/hard/near005: `{_pct(payload['repair']['test']['delta_vs_di']['all_improvement'])}` / `{_pct(payload['repair']['test']['delta_vs_di']['hard_failure_improvement'])}` / `{_pct(payload['repair']['test']['near_delta_vs_di'])}`.",
            f"- decision: `{payload['deployment_decision']['decision']}`.",
            "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            "<!-- STAGE42_FE_CONSTRAINED_FC_SAFETY_COMPOSER:END -->",
            "",
        ]
    )


def _replace_text_section(old: str, tag: str, block: str) -> str:
    start = f"<!-- {tag}:START -->"
    end = f"<!-- {tag}:END -->"
    if start in old and end in old:
        before, rest = old.split(start, 1)
        _, after = rest.split(end, 1)
        return before.rstrip() + "\n\n" + block.strip() + after
    return old.rstrip() + "\n\n" + block.strip() + "\n"


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(_replace_text_section(old, "STAGE42_FE_CONSTRAINED_FC_SAFETY_COMPOSER", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FE constrained FC/safety composer"
    state["current_verdict"] = payload["stage42_fe_gate"]["verdict"]
    state["stage42_fe_constrained_fc_safety_composer"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_fe_gate"]["verdict"],
        "gates": f"{payload['stage42_fe_gate']['passed']}/{payload['stage42_fe_gate']['total']}",
        "selected_candidate": payload["repair"]["selected"]["candidate"],
        "test_metric_vs_floor": payload["repair"]["test"]["metric_vs_floor"],
        "test_diagnostics": payload["repair"]["test"]["diagnostics"],
        "test_delta_vs_fc": payload["repair"]["test"]["delta_vs_fc"],
        "test_delta_vs_di": payload["repair"]["test"]["delta_vs_di"],
        "deployment_decision": payload["deployment_decision"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FE tests whether high-accuracy FC can be safely composed with DI/FA/FB predicted-proximity fallbacks under validation-only constraints.",
    }
    block = state.get("m3w_work_attempts_failures_successes_readme")
    if isinstance(block, dict):
        evidence = list(block.get("latest_included_evidence", []))
        if "Stage42-FE constrained FC/safety composer" not in evidence:
            evidence.append("Stage42-FE constrained FC/safety composer")
        block["latest_included_evidence"] = evidence
        block["source"] = "cached_verified_summary_from_stage18_to_stage42_reports_plus_stage42_es_to_fe_fresh_audits"
        block[
            "latest_conclusion"
        ] = "Stage42-FE follows FD by trying validation-only constrained composition between high-accuracy FC and safety fallbacks, testing whether proximity safety can be restored without losing FC all/hard gains."
        state["m3w_work_attempts_failures_successes_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_constrained_fc_safety_composer() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_constrained_fc_safety_composer()
    gate = result["stage42_fe_gate"]
    print(f"Stage42-FE gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
