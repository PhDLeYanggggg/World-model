from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_external_validation as s42b
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "adaptive_group_repair_stage42.json"
REPORT_MD = OUT_DIR / "adaptive_group_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ew_gate.md"

DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
TARGET_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "paper_outline_stage42.md",
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "data_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

SOURCE = "fresh_stage42_adaptive_group_repair"
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EW 检验 Stage42-DI 的全局 group repair 是否可以被 validation-only adaptive group repair 改进。",
    "repair 候选只使用 predicted rollout geometry、source/frame/horizon group key、agent id、当前/过去可得信息。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "mode、slice rule、candidate 只在 validation 上选择；test 只按冻结规则执行。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _candidate_name(candidate: Mapping[str, Any], index: int) -> str:
    parts = [f"c{index:02d}", str(candidate["mode"]), f"sep{candidate.get('min_sep')}"]
    for key in ["margin", "alpha", "safe_min_sep", "strength"]:
        if key in candidate:
            parts.append(f"{key}{candidate[key]}")
    return "|".join(parts)


def _candidate_score(metric: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> float:
    near_delta = float(diagnostics.get("final_near_005", 0.0)) - float(diagnostics.get("base_near_005", 0.0))
    return (
        1.35 * float(metric["all_improvement"])
        + 1.35 * float(metric["hard_failure_improvement"])
        + 1.15 * float(metric["t50_improvement"])
        + 0.50 * float(metric["t100_raw_frame_diagnostic_improvement"])
        - 40.0 * max(0.0, float(metric["easy_degradation"]) - 0.02)
        - 9.0 * max(0.0, near_delta)
        - 0.01 * float(metric["switch_rate"])
    )


def _adaptive_key(data: Mapping[str, np.ndarray], ids: np.ndarray, risk_high: np.ndarray, mode: str) -> np.ndarray:
    dataset = data["dataset"][ids].astype(str)
    horizon = data["horizon"][ids].astype(int)
    if mode == "global":
        return np.asarray(["global" for _ in ids], dtype=object)
    if mode == "domain_horizon":
        return np.asarray([f"{d}|{h}" for d, h in zip(dataset, horizon)], dtype=object)
    if mode == "domain_horizon_risk":
        return np.asarray([f"{d}|{h}|{'risk' if r else 'clear'}" for d, h, r in zip(dataset, horizon, risk_high[ids])], dtype=object)
    raise ValueError(f"unknown adaptive mode: {mode}")


def _risk_high_from_base(data: Mapping[str, np.ndarray], am_candidate: Mapping[str, Any], group_key: np.ndarray) -> np.ndarray:
    normalizer = np.maximum(data["scale"].astype(np.float64), EPS)
    agent = data["agent_id"].astype(np.int64)
    base_min = di._min_group_distance_fast(am_candidate["selected_xy"], group_key, normalizer, agent)
    floor_min = di._min_group_distance_fast(am_candidate["floor_xy"], group_key, normalizer, agent)
    return (
        (np.isfinite(base_min) & (base_min < 0.08))
        | (np.isfinite(base_min) & np.isfinite(floor_min) & (base_min + 0.005 < floor_min))
        | data["hard"].astype(bool)
        | data["failure"].astype(bool)
    )


def _diagnostics_for_xy(
    xy: np.ndarray,
    base_xy: np.ndarray,
    floor_xy: np.ndarray,
    group_key: np.ndarray,
    normalizer: np.ndarray,
    agent_id: np.ndarray,
) -> dict[str, float | None]:
    base_min = di._min_group_distance_fast(base_xy, group_key, normalizer, agent_id)
    final_min = di._min_group_distance_fast(xy, group_key, normalizer, agent_id)
    floor_min = di._min_group_distance_fast(floor_xy, group_key, normalizer, agent_id)
    return {
        "base_near_005": float(np.mean(np.isfinite(base_min) & (base_min < 0.05))) if len(base_min) else 0.0,
        "final_near_005": float(np.mean(np.isfinite(final_min) & (final_min < 0.05))) if len(final_min) else 0.0,
        "floor_near_005": float(np.mean(np.isfinite(floor_min) & (floor_min < 0.05))) if len(floor_min) else 0.0,
        "base_p05_min_distance": float(np.percentile(base_min[np.isfinite(base_min)], 5)) if np.any(np.isfinite(base_min)) else None,
        "final_p05_min_distance": float(np.percentile(final_min[np.isfinite(final_min)], 5)) if np.any(np.isfinite(final_min)) else None,
        "floor_p05_min_distance": float(np.percentile(floor_min[np.isfinite(floor_min)], 5)) if np.any(np.isfinite(floor_min)) else None,
    }


def _subset_eval(
    result: Mapping[str, Any],
    ids_all: np.ndarray,
    positions: np.ndarray,
    data: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    base_xy: np.ndarray,
    group_key: np.ndarray,
) -> dict[str, Any]:
    row_ids = ids_all[positions]
    selected_ade = result["selected_ade"][positions]
    floor_ade = result["floor_ade"][positions]
    switch = result["switch"][positions]
    metric = di._metric_subset(selected_ade, floor_ade, data, row_ids, switch)
    diag = _diagnostics_for_xy(
        result["selected_xy"][positions],
        base_xy[row_ids],
        floor_xy[row_ids],
        group_key[row_ids],
        np.maximum(data["scale"][row_ids].astype(np.float64), EPS),
        data["agent_id"][row_ids].astype(np.int64),
    )
    return {"metric": metric, "diagnostics": diag}


def _final_eval(
    selected_xy: np.ndarray,
    switch: np.ndarray,
    ids: np.ndarray,
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    base_xy: np.ndarray,
    group_key: np.ndarray,
) -> dict[str, Any]:
    selected_ade, selected_fde = di._trajectory_errors_subset(selected_xy, labels, ids)
    floor_ade, floor_fde = di._trajectory_errors_subset(floor_xy[ids], labels, ids)
    del selected_fde, floor_fde
    metric = di._metric_subset(selected_ade, floor_ade, data, ids, switch)
    diag = _diagnostics_for_xy(
        selected_xy,
        base_xy[ids],
        floor_xy[ids],
        group_key[ids],
        np.maximum(data["scale"][ids].astype(np.float64), EPS),
        data["agent_id"][ids].astype(np.int64),
    )
    h = data["horizon"][ids].astype(int)
    hard_failure = data["hard"][ids].astype(bool) | data["failure"][ids].astype(bool)
    easy = data["easy"][ids].astype(bool)
    domain = data["dataset"][ids].astype(str)
    bootstrap = {
        "all": di._bootstrap_ci_subset(selected_ade, floor_ade, np.ones(len(ids), dtype=bool), seed=42801),
        "t50": di._bootstrap_ci_subset(selected_ade, floor_ade, h == 50, seed=42802),
        "t100_raw_frame_diagnostic": di._bootstrap_ci_subset(selected_ade, floor_ade, h == 100, seed=42803),
        "hard_failure": di._bootstrap_ci_subset(selected_ade, floor_ade, hard_failure, seed=42804),
        "easy_degradation": di._bootstrap_ci_subset(floor_ade, selected_ade, easy, seed=42805),
    }
    by_domain = {
        d: di._metric_subset(selected_ade[domain == d], floor_ade[domain == d], data, ids[domain == d], switch[domain == d])
        for d in sorted(set(domain.tolist()))
    }
    return {
        "metric_vs_floor": metric,
        "diagnostics": diag,
        "bootstrap": bootstrap,
        "by_domain": by_domain,
        "selected_ade": selected_ade,
        "floor_ade": floor_ade,
    }


def _mixed_group_selection_rate(chosen: np.ndarray, group_key: np.ndarray) -> dict[str, float | int]:
    if len(chosen) == 0:
        return {"mixed_group_count": 0, "group_count": 0, "mixed_group_rate": 0.0, "mixed_row_count": 0, "mixed_row_rate": 0.0}
    mixed_groups = 0
    mixed_rows = 0
    for key in sorted(set(group_key.tolist())):
        rows = np.where(group_key == key)[0]
        if len(set(chosen[rows].tolist())) > 1:
            mixed_groups += 1
            mixed_rows += len(rows)
    group_count = len(set(group_key.tolist()))
    return {
        "mixed_group_count": int(mixed_groups),
        "group_count": int(group_count),
        "mixed_group_rate": float(mixed_groups / max(group_count, 1)),
        "mixed_row_count": int(mixed_rows),
        "mixed_row_rate": float(mixed_rows / max(len(chosen), 1)),
    }


def _repair_cache_for_ids(
    ids: np.ndarray,
    candidates: list[Mapping[str, Any]],
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    am_candidate: Mapping[str, Any],
    group_key: np.ndarray,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        name = _candidate_name(candidate, index)
        cache[name] = {
            "candidate": dict(candidate),
            "result": di._repair_subset(
                ids,
                candidate,
                data,
                labels,
                floor_xy,
                am_candidate["pred_xy"].astype(np.float32),
                am_candidate["selected_xy"].astype(np.float32),
                am_candidate["switch"].astype(bool),
                group_key,
            ),
        }
    return cache


def _select_rules_for_mode(
    mode: str,
    val_ids: np.ndarray,
    val_cache: Mapping[str, Mapping[str, Any]],
    data: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    base_xy: np.ndarray,
    group_key: np.ndarray,
    risk_high: np.ndarray,
) -> dict[str, Any]:
    keys = _adaptive_key(data, val_ids, risk_high, mode)
    rule_rows = []
    rules: dict[str, str] = {}
    for key in sorted(set(keys.tolist())):
        positions = np.where(keys == key)[0]
        best_name = ""
        best_score = -1e9
        best_eval: dict[str, Any] | None = None
        for name, cached in val_cache.items():
            subset = _subset_eval(cached["result"], val_ids, positions, data, floor_xy, base_xy, group_key)
            score = _candidate_score(subset["metric"], subset["diagnostics"])
            if score > best_score:
                best_name = name
                best_score = float(score)
                best_eval = subset
        rules[str(key)] = best_name
        rule_rows.append(
            {
                "key": str(key),
                "rows": int(len(positions)),
                "selected_candidate": best_name,
                "val_score": best_score,
                "val_metric": best_eval["metric"] if best_eval is not None else {},
                "val_diagnostics": best_eval["diagnostics"] if best_eval is not None else {},
            }
        )
    global_rows = []
    all_positions = np.arange(len(val_ids), dtype=np.int64)
    for name, cached in val_cache.items():
        subset = _subset_eval(cached["result"], val_ids, all_positions, data, floor_xy, base_xy, group_key)
        global_rows.append({"candidate": name, "score": _candidate_score(subset["metric"], subset["diagnostics"]), "metric": subset["metric"]})
    global_best = max(global_rows, key=lambda row: float(row["score"]))["candidate"]
    return {
        "mode": mode,
        "rules": rules,
        "rule_rows": rule_rows,
        "global_fallback_candidate": global_best,
        "global_validation_rows": sorted(global_rows, key=lambda row: float(row["score"]), reverse=True),
    }


def _apply_rules(
    selection: Mapping[str, Any],
    ids: np.ndarray,
    cache: Mapping[str, Mapping[str, Any]],
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    base_xy: np.ndarray,
    group_key: np.ndarray,
    risk_high: np.ndarray,
) -> dict[str, Any]:
    keys = _adaptive_key(data, ids, risk_high, selection["mode"])
    selected_xy = floor_xy[ids].astype(np.float32).copy()
    switch = np.zeros(len(ids), dtype=bool)
    chosen = []
    for pos, key in enumerate(keys.tolist()):
        name = selection["rules"].get(str(key), selection["global_fallback_candidate"])
        result = cache[name]["result"]
        selected_xy[pos] = result["selected_xy"][pos]
        switch[pos] = bool(result["switch"][pos])
        chosen.append(name)
    chosen_arr = np.asarray(chosen, dtype=object)
    final = _final_eval(selected_xy, switch, ids, data, labels, floor_xy, base_xy, group_key)
    final["candidate_usage"] = {name: int(np.sum(chosen_arr == name)) for name in sorted(cache)}
    final["mixed_group_selection"] = _mixed_group_selection_rate(chosen_arr, group_key[ids])
    return final


def _adaptive_repair(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
) -> dict[str, Any]:
    val_ids = np.where(split == "val")[0]
    test_ids = np.where(split == "test")[0]
    group_key = di._group_key(data)
    floor_xy = floor["floor_xy"].astype(np.float32)
    am_candidate = di._rebuild_stage42_am_candidate(data, split, labels, floor)
    am_candidate = {**am_candidate, "floor_xy": floor_xy}
    base_xy = am_candidate["selected_xy"].astype(np.float32)
    risk_high = _risk_high_from_base(data, am_candidate, group_key)
    candidates = di._candidate_grid()
    val_cache = _repair_cache_for_ids(val_ids, candidates, data, labels, floor_xy, am_candidate, group_key)
    test_cache = _repair_cache_for_ids(test_ids, candidates, data, labels, floor_xy, am_candidate, group_key)
    mode_rows = []
    for mode in ["global", "domain_horizon", "domain_horizon_risk"]:
        selection = _select_rules_for_mode(mode, val_ids, val_cache, data, floor_xy, base_xy, group_key, risk_high)
        val_result = _apply_rules(selection, val_ids, val_cache, data, labels, floor_xy, base_xy, group_key, risk_high)
        test_result = _apply_rules(selection, test_ids, test_cache, data, labels, floor_xy, base_xy, group_key, risk_high)
        val_score = _candidate_score(val_result["metric_vs_floor"], val_result["diagnostics"])
        test_score = _candidate_score(test_result["metric_vs_floor"], test_result["diagnostics"])
        mode_rows.append(
            {
                "mode": mode,
                "selection": selection,
                "val_metric_vs_floor": val_result["metric_vs_floor"],
                "val_diagnostics": val_result["diagnostics"],
                "val_score": float(val_score),
                "val_mixed_group_selection": val_result["mixed_group_selection"],
                "test_metric_vs_floor": test_result["metric_vs_floor"],
                "test_diagnostics": test_result["diagnostics"],
                "test_score": float(test_score),
                "candidate_usage": test_result["candidate_usage"],
                "mixed_group_selection": test_result["mixed_group_selection"],
                "bootstrap": test_result["bootstrap"],
                "by_domain": test_result["by_domain"],
            }
        )
    eligible = [
        row
        for row in mode_rows
        if row["mode"] != "domain_horizon_risk" or int(row["val_mixed_group_selection"]["mixed_group_count"]) == 0
    ]
    selected = max(eligible or mode_rows, key=lambda row: float(row["val_score"]))
    return {
        "candidate_count": len(candidates),
        "candidate_names": sorted(val_cache.keys()),
        "mode_rows": mode_rows,
        "selected": selected,
        "selection_policy": {
            "source": "validation_only",
            "constraint": "mode must preserve source/frame/horizon group-consistency on validation before comparing validation score",
            "eligible_modes": [row["mode"] for row in eligible],
        },
        "stage42_am_rebuilt": {
            "lambda": am_candidate["lambda"],
            "feature_count": am_candidate["feature_count"],
            "policy_slice_count": len(am_candidate["policy"]["slices"]),
            "val_metric": am_candidate["val_metric"],
        },
        "risk_high_stats": {
            "train_rate": float(np.mean(risk_high[split == "train"])),
            "val_rate": float(np.mean(risk_high[split == "val"])),
            "test_rate": float(np.mean(risk_high[split == "test"])),
        },
    }


def _compare_to_di(metric: Mapping[str, Any]) -> dict[str, Any]:
    di_payload = read_json(DI_JSON, {})
    di_metric = di_payload.get("repair", {}).get("test", {}).get("metric_vs_floor", {})

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
            "all_improvement": float(metric.get("all_improvement", 0.0)) - float(ref.get("all_improvement", 0.0)),
            "t50_improvement": float(metric.get("t50_improvement", 0.0)) - float(ref.get("t50_improvement", 0.0)),
            "t100_raw_frame_diagnostic_improvement": float(metric.get("t100_raw_frame_diagnostic_improvement", 0.0))
            - float(ref.get("t100_raw_frame_diagnostic_improvement", 0.0)),
            "hard_failure_improvement": float(metric.get("hard_failure_improvement", 0.0)) - float(ref.get("hard_failure_improvement", 0.0)),
            "easy_degradation": float(metric.get("easy_degradation", 0.0)) - float(ref.get("easy_degradation", 0.0)),
        }

    return {"stage42_di_metric": di_metric, "delta_vs_stage42_di": delta(di_metric)}


def _deployment_decision(metric: Mapping[str, Any], comparison: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    delta_di = comparison["delta_vs_stage42_di"]
    mixed = selected["mixed_group_selection"]
    group_consistent = selected["mode"] != "domain_horizon_risk" or int(mixed["mixed_group_count"]) == 0
    promotes = (
        metric["all_improvement"] > 0.0
        and metric["hard_failure_improvement"] > 0.0
        and metric["easy_degradation"] <= 0.02
        and (delta_di["all_improvement"] or 0.0) > 0.0
        and (delta_di["hard_failure_improvement"] or 0.0) > 0.0
        and selected["test_diagnostics"]["final_near_005"] <= selected["test_diagnostics"]["base_near_005"] + EPS
        and group_consistent
    )
    useful = metric["all_improvement"] > 0.0 and metric["hard_failure_improvement"] > 0.0 and metric["easy_degradation"] <= 0.02
    return {
        "promote_adaptive_group_repair": bool(promotes),
        "diagnostic_positive": bool(useful),
        "decision": "promote_stage42_ew_adaptive_group_repair"
        if promotes
        else ("stage42_ew_adaptive_group_repair_positive_not_promoted" if useful else "stage42_ew_adaptive_group_repair_not_enough_keep_stage42_di_or_cq_floor"),
        "reason": "Promotion requires validation-selected adaptive repair to beat Stage42-DI on all and hard, preserve easy, not worsen near@0.05, and avoid mixed repair choices inside a source/frame/horizon group.",
        "group_consistent_selection": bool(group_consistent),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    selected = payload["adaptive_repair"]["selected"]
    metric = selected["test_metric_vs_floor"]
    diag = selected["test_diagnostics"]
    delta_di = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    decision = payload["deployment_decision"]
    gates = {
        "repair_candidates_evaluated": payload["adaptive_repair"]["candidate_count"] >= 40,
        "adaptive_modes_evaluated": len(payload["adaptive_repair"]["mode_rows"]) == 3,
        "validation_only_mode_selection": no_leak["validation_only_mode_selection"] is True and no_leak["test_threshold_tuning"] is False,
        "selected_mode_recorded": bool(selected["mode"]),
        "test_all_positive_vs_floor": metric["all_improvement"] > 0.0,
        "test_t50_positive_vs_floor": metric["t50_improvement"] > 0.0,
        "test_hard_positive_vs_floor": metric["hard_failure_improvement"] > 0.0,
        "easy_degradation_under_2pct": metric["easy_degradation"] <= 0.02,
        "near005_not_worse_than_base": diag["final_near_005"] <= diag["base_near_005"] + EPS,
        "group_consistent_selection": decision["group_consistent_selection"] is True,
        "beats_stage42_di_all": (delta_di["all_improvement"] or 0.0) > 0.0,
        "beats_stage42_di_hard": (delta_di["hard_failure_improvement"] or 0.0) > 0.0,
        "no_leakage_pass": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["central_velocity"] is False
        and no_leak["test_endpoint_goals"] is False
        and no_leak["test_threshold_tuning"] is False
        and no_leak["validation_only_rule_selection"] is True
        and no_leak["source_overlap_pass"] is True,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage42_ew_adaptive_group_repair_pass_promotable"
    elif gates["test_all_positive_vs_floor"] and gates["test_hard_positive_vs_floor"] and gates["easy_degradation_under_2pct"]:
        verdict = "stage42_ew_adaptive_group_repair_positive_not_promoted"
    else:
        verdict = "stage42_ew_adaptive_group_repair_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    source_split = s42b.build_stage42_source_split()
    data = s41._combined()
    split, group = am._split_arrays(data)
    split_stats = am._source_stats(data, split, group)
    labels = am._reconstruct_waypoint_labels(data)
    floor = am._floor_arrays(data, split == "train")
    adaptive = _adaptive_repair(data, split, labels, floor)
    metric = adaptive["selected"]["test_metric_vs_floor"]
    comparison = _compare_to_di(metric)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EW adaptive group-consistency repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(["data/stage41_world_model/combined_external.npz", str(DI_JSON)]),
        "current_facts": CURRENT_FACTS,
        "source_split": source_split,
        "split_stats": split_stats,
        "adaptive_repair": adaptive,
        "comparison_to_prior": comparison,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "group_features_predicted_rollout_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_rule_selection": True,
            "validation_only_mode_selection": True,
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(split_stats["source_overlap_pass"]),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["deployment_decision"] = _deployment_decision(metric, comparison, adaptive["selected"])
    payload["paper_file_status"] = _refresh_paper_files(payload)
    payload["stage42_ew_gate"] = _gate(payload)
    return payload


def _paper_lines(payload: Mapping[str, Any]) -> list[str]:
    selected = payload["adaptive_repair"]["selected"]
    metric = selected["test_metric_vs_floor"]
    diag = selected["test_diagnostics"]
    delta = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    return [
        "## Stage42-EW Adaptive Group Repair",
        "",
        "- source: `fresh_stage42_adaptive_group_repair`",
        "- role: validation-only adaptive selection over Stage42-DI repair candidates by global / domain+horizon / domain+horizon+risk slices.",
        f"- selected mode: `{selected['mode']}`.",
        f"- test all/t50/t100raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        f"- delta vs Stage42-DI all/hard/easy: `{_pct(delta['all_improvement'])}` / `{_pct(delta['hard_failure_improvement'])}` / `{_pct(delta['easy_degradation'])}`.",
        f"- near@0.05 base/final: `{_pct(diag['base_near_005'])}` / `{_pct(diag['final_near_005'])}`.",
        f"- mixed group selection rate: `{_pct(selected['mixed_group_selection']['mixed_group_rate'])}`.",
        f"- decision: `{payload['deployment_decision']['decision']}`.",
        "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
    ]


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _paper_lines(payload)
    status = []
    for path in PAPER_FILES:
        if path.exists():
            _replace_section(path, "STAGE42_EW_ADAPTIVE_GROUP_REPAIR", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "updated": True,
                    "contains_stage42_ew": "STAGE42_EW_ADAPTIVE_GROUP_REPAIR" in text,
                    "contains_boundaries": "no Stage5C" in text and "no SMC" in text,
                }
            )
        else:
            status.append({"path": str(path), "updated": False, "missing": True})
    return status


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    selected = payload["adaptive_repair"]["selected"]
    metric = selected["test_metric_vs_floor"]
    diag = selected["test_diagnostics"]
    delta = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    lines = [
        "# Stage42-EW Adaptive Group Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ew_gate']['passed']} / {payload['stage42_ew_gate']['total']}`",
        f"- verdict: `{payload['stage42_ew_gate']['verdict']}`",
        f"- decision: `{payload['deployment_decision']['decision']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Selected Adaptive Repair",
        "",
        f"- mode: `{selected['mode']}`",
        f"- validation_score: `{selected['val_score']:.6f}`",
        f"- test_score: `{selected['test_score']:.6f}`",
        f"- candidate_usage: `{selected['candidate_usage']}`",
        f"- mixed_group_selection: `{selected['mixed_group_selection']}`",
        "",
        "## Test Once Metrics",
        "",
        "| all | t50 | t100 raw | hard/failure | easy | switch | near base/final |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| {_pct(metric['all_improvement'])} | {_pct(metric['t50_improvement'])} | {_pct(metric['t100_raw_frame_diagnostic_improvement'])} | "
        f"{_pct(metric['hard_failure_improvement'])} | {_pct(metric['easy_degradation'])} | {_pct(metric['switch_rate'])} | "
        f"{_pct(diag['base_near_005'])}/{_pct(diag['final_near_005'])} |",
        "",
        "## Mode Comparison",
        "",
        "| mode | val score | test score | all | t50 | t100 raw | hard | easy | mixed groups | usage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["adaptive_repair"]["mode_rows"]:
        m = row["test_metric_vs_floor"]
        mixed = row["mixed_group_selection"]
        lines.append(
            f"| `{row['mode']}` | {row['val_score']:.6f} | {row['test_score']:.6f} | {_pct(m['all_improvement'])} | "
            f"{_pct(m['t50_improvement'])} | {_pct(m['t100_raw_frame_diagnostic_improvement'])} | "
            f"{_pct(m['hard_failure_improvement'])} | {_pct(m['easy_degradation'])} | {_pct(mixed['mixed_group_rate'])} | `{row['candidate_usage']}` |"
        )
    lines.extend(
        [
            "",
            "## Delta vs Stage42-DI",
            "",
            "| all | t50 | t100 raw | hard | easy |",
            "| ---: | ---: | ---: | ---: | ---: |",
            f"| {_pct(delta['all_improvement'])} | {_pct(delta['t50_improvement'])} | {_pct(delta['t100_raw_frame_diagnostic_improvement'])} | {_pct(delta['hard_failure_improvement'])} | {_pct(delta['easy_degradation'])} |",
            "",
            "## Bootstrap CI",
            "",
            "| slice | low | mid | high | n |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, row in selected["bootstrap"].items():
        lines.append(f"| `{key}` | {row['low']:.6f} | {row['mid']:.6f} | {row['high']:.6f} | {row['n']} |")
    lines.extend(["", "## By Domain", "", "| domain | rows | all | t50 | t100 raw | hard | easy | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for domain, row in selected["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | {_pct(row['all_improvement'])} | {_pct(row['t50_improvement'])} | {_pct(row['t100_raw_frame_diagnostic_improvement'])} | {_pct(row['hard_failure_improvement'])} | {_pct(row['easy_degradation'])} | {_pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-EW tests whether DI's one global repair candidate leaves slice-specific value on the table.",
            "- Mode and rule selection are validation-only. Test metrics are not used to pick thresholds or candidates.",
            "- If EW does not beat Stage42-DI on all and hard while preserving group consistency, Stage42-DI / CQ floor remains the deployable evidence.",
            "- This remains source-level raw-frame 2.5D evidence, not metric/seconds-level, true 3D, Stage5C, or SMC evidence.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_ew_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ew_gate"]
    return [
        "# Stage42-EW Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _readme_lines(payload: Mapping[str, Any]) -> list[str]:
    selected = payload["adaptive_repair"]["selected"]
    metric = selected["test_metric_vs_floor"]
    diag = selected["test_diagnostics"]
    delta = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    return [
        "## Stage42-EW Adaptive Group Repair",
        "",
        "- source: `fresh_stage42_adaptive_group_repair`",
        "- role: validation-only adaptive repair over Stage42-DI candidate grid by global / domain+horizon / domain+horizon+risk slices.",
        f"- gate: `{payload['stage42_ew_gate']['passed']} / {payload['stage42_ew_gate']['total']}`; verdict `{payload['stage42_ew_gate']['verdict']}`.",
        f"- selected mode: `{selected['mode']}`.",
        f"- test all/t50/t100raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        f"- delta vs Stage42-DI all/hard/easy: `{_pct(delta['all_improvement'])}` / `{_pct(delta['hard_failure_improvement'])}` / `{_pct(delta['easy_degradation'])}`.",
        f"- near@0.05 base/final: `{_pct(diag['base_near_005'])}` / `{_pct(diag['final_near_005'])}`.",
        f"- mixed group selection rate: `{_pct(selected['mixed_group_selection']['mixed_group_rate'])}`.",
        f"- decision: `{payload['deployment_decision']['decision']}`.",
        "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _readme_lines(payload)
    for path in [README_RESULTS, M3W_README, TARGET_SUMMARY, WORK_SUMMARY]:
        _replace_section(path, "STAGE42_EW_ADAPTIVE_GROUP_REPAIR", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EW adaptive group repair"
    state["current_verdict"] = payload["stage42_ew_gate"]["verdict"]
    state["stage42_ew_adaptive_group_repair"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ew_gate"]["verdict"],
        "gates": f"{payload['stage42_ew_gate']['passed']}/{payload['stage42_ew_gate']['total']}",
        "selected": payload["adaptive_repair"]["selected"],
        "comparison_to_prior": payload["comparison_to_prior"],
        "deployment_decision": payload["deployment_decision"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_adaptive_group_repair(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = _build_payload()
    write_json(REPORT_JSON, am._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_adaptive_group_repair()
