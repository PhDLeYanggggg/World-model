from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_adaptive_group_repair as ew
from src import stage42_external_validation as s42b
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "continuous_group_risk_repair_stage42.json"
REPORT_MD = OUT_DIR / "continuous_group_risk_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ey_gate.md"

DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
EX_JSON = OUT_DIR / "group_level_risk_repair_stage42.json"
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

SOURCE = "fresh_stage42_continuous_group_risk_repair"
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EX 证明 binary group-risk 几乎饱和，无法提供比 Stage42-DI 更强的 adaptive repair 信号。",
    "Stage42-EY 将 group risk 改为 continuous predicted-geometry risk score，再用 validation-only quantile buckets 冻结 repair rules。",
    "risk score 只使用 predicted/base/floor rollout geometry、source/frame/horizon group key、agent id、当前/过去可得信息。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "risk bucket 阈值、mode、slice rule、candidate 只在 validation 上选择；test 只按冻结规则执行。",
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


def _group_reduce_max(values: np.ndarray, group_key: np.ndarray) -> np.ndarray:
    keys = np.asarray(group_key, dtype=object)
    out = np.zeros(len(values), dtype=np.float64)
    order = np.argsort(keys)
    start = 0
    while start < len(order):
        end = start + 1
        key = keys[order[start]]
        while end < len(order) and keys[order[end]] == key:
            end += 1
        rows = order[start:end]
        out[rows] = float(np.nanmax(values[rows])) if len(rows) else 0.0
        start = end
    return out


def _continuous_group_risk_score(
    data: Mapping[str, np.ndarray],
    am_candidate: Mapping[str, Any],
    group_key: np.ndarray,
) -> dict[str, np.ndarray]:
    normalizer = np.maximum(data["scale"].astype(np.float64), EPS)
    agent = data["agent_id"].astype(np.int64)
    base_xy = am_candidate["selected_xy"].astype(np.float32)
    pred_xy = am_candidate["pred_xy"].astype(np.float32)
    floor_xy = am_candidate["floor_xy"].astype(np.float32)
    base_min = di._min_group_distance_fast(base_xy, group_key, normalizer, agent)
    pred_min = di._min_group_distance_fast(pred_xy, group_key, normalizer, agent)
    floor_min = di._min_group_distance_fast(floor_xy, group_key, normalizer, agent)
    base_min_clean = np.nan_to_num(base_min, nan=1.0, posinf=1.0, neginf=0.0)
    pred_min_clean = np.nan_to_num(pred_min, nan=1.0, posinf=1.0, neginf=0.0)
    floor_min_clean = np.nan_to_num(floor_min, nan=1.0, posinf=1.0, neginf=0.0)
    close_base = np.maximum(0.0, 0.10 - base_min_clean) / 0.10
    close_pred = np.maximum(0.0, 0.10 - pred_min_clean) / 0.10
    worse_than_floor = np.maximum(0.0, floor_min_clean - base_min_clean) / 0.10
    switch_component = am_candidate["switch"].astype(np.float64) * 0.05
    row_score = np.clip(0.55 * close_base + 0.25 * close_pred + 0.35 * worse_than_floor + switch_component, 0.0, 4.0)
    group_score = _group_reduce_max(row_score, group_key)
    return {
        "row_score": row_score.astype(np.float64),
        "group_score": group_score.astype(np.float64),
        "base_min": base_min_clean.astype(np.float64),
        "pred_min": pred_min_clean.astype(np.float64),
        "floor_min": floor_min_clean.astype(np.float64),
    }


def _unique_group_scores(scores: np.ndarray, group_key: np.ndarray, ids: np.ndarray) -> np.ndarray:
    seen: dict[str, float] = {}
    for row in ids.tolist():
        seen.setdefault(str(group_key[row]), float(scores[row]))
    return np.asarray(list(seen.values()), dtype=np.float64)


def _bucket_thresholds(scores: np.ndarray, group_key: np.ndarray, ids: np.ndarray, bucket_count: int) -> list[float]:
    vals = _unique_group_scores(scores, group_key, ids)
    if len(vals) == 0:
        return []
    quantiles = [i / bucket_count for i in range(1, bucket_count)]
    return [float(np.quantile(vals, q)) for q in quantiles]


def _bucketize(scores: np.ndarray, thresholds: list[float]) -> np.ndarray:
    if not thresholds:
        return np.zeros(len(scores), dtype=np.int64)
    return np.searchsorted(np.asarray(thresholds, dtype=np.float64), scores, side="right").astype(np.int64)


def _bucket_key(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    buckets: np.ndarray,
    mode: str,
) -> np.ndarray:
    dataset = data["dataset"][ids].astype(str)
    horizon = data["horizon"][ids].astype(int)
    if mode == "global":
        return np.asarray(["global" for _ in ids], dtype=object)
    if mode == "domain_horizon":
        return np.asarray([f"{d}|{h}" for d, h in zip(dataset, horizon)], dtype=object)
    if mode == "domain_horizon_risk3":
        return np.asarray([f"{d}|{h}|risk{int(b)}" for d, h, b in zip(dataset, horizon, buckets[ids])], dtype=object)
    if mode == "domain_horizon_risk4":
        return np.asarray([f"{d}|{h}|risk{int(b)}" for d, h, b in zip(dataset, horizon, buckets[ids])], dtype=object)
    raise ValueError(f"unknown continuous risk mode: {mode}")


def _select_rules_for_mode(
    mode: str,
    val_ids: np.ndarray,
    val_cache: Mapping[str, Mapping[str, Any]],
    data: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    base_xy: np.ndarray,
    group_key: np.ndarray,
    buckets: np.ndarray,
) -> dict[str, Any]:
    keys = _bucket_key(data, val_ids, buckets, mode)
    rules: dict[str, str] = {}
    rule_rows = []
    for key in sorted(set(keys.tolist())):
        positions = np.where(keys == key)[0]
        best_name = ""
        best_score = -1e9
        best_eval: dict[str, Any] | None = None
        for name, cached in val_cache.items():
            subset = ew._subset_eval(cached["result"], val_ids, positions, data, floor_xy, base_xy, group_key)
            score = ew._candidate_score(subset["metric"], subset["diagnostics"])
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
        subset = ew._subset_eval(cached["result"], val_ids, all_positions, data, floor_xy, base_xy, group_key)
        global_rows.append({"candidate": name, "score": ew._candidate_score(subset["metric"], subset["diagnostics"]), "metric": subset["metric"]})
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
    buckets: np.ndarray,
) -> dict[str, Any]:
    keys = _bucket_key(data, ids, buckets, selection["mode"])
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
    final = ew._final_eval(selected_xy, switch, ids, data, labels, floor_xy, base_xy, group_key)
    final["candidate_usage"] = {name: int(np.sum(chosen_arr == name)) for name in sorted(cache)}
    final["mixed_group_selection"] = ew._mixed_group_selection_rate(chosen_arr, group_key[ids])
    return final


def _continuous_risk_repair(
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
    risk = _continuous_group_risk_score(data, am_candidate, group_key)
    candidates = di._candidate_grid()
    val_cache = ew._repair_cache_for_ids(val_ids, candidates, data, labels, floor_xy, am_candidate, group_key)
    test_cache = ew._repair_cache_for_ids(test_ids, candidates, data, labels, floor_xy, am_candidate, group_key)
    bucket_specs = {
        "global": [],
        "domain_horizon": [],
        "domain_horizon_risk3": _bucket_thresholds(risk["group_score"], group_key, val_ids, 3),
        "domain_horizon_risk4": _bucket_thresholds(risk["group_score"], group_key, val_ids, 4),
    }
    bucket_arrays = {mode: _bucketize(risk["group_score"], thresholds) for mode, thresholds in bucket_specs.items()}
    mode_rows = []
    for mode in ["global", "domain_horizon", "domain_horizon_risk3", "domain_horizon_risk4"]:
        buckets = bucket_arrays[mode]
        selection = _select_rules_for_mode(mode, val_ids, val_cache, data, floor_xy, base_xy, group_key, buckets)
        val_result = _apply_rules(selection, val_ids, val_cache, data, labels, floor_xy, base_xy, group_key, buckets)
        test_result = _apply_rules(selection, test_ids, test_cache, data, labels, floor_xy, base_xy, group_key, buckets)
        mode_rows.append(
            {
                "mode": mode,
                "thresholds": bucket_specs[mode],
                "bucket_counts_val": {str(int(b)): int(np.sum(buckets[val_ids] == b)) for b in sorted(set(buckets[val_ids].tolist()))},
                "bucket_counts_test": {str(int(b)): int(np.sum(buckets[test_ids] == b)) for b in sorted(set(buckets[test_ids].tolist()))},
                "selection": selection,
                "val_metric_vs_floor": val_result["metric_vs_floor"],
                "val_diagnostics": val_result["diagnostics"],
                "val_score": float(ew._candidate_score(val_result["metric_vs_floor"], val_result["diagnostics"])),
                "val_mixed_group_selection": val_result["mixed_group_selection"],
                "test_metric_vs_floor": test_result["metric_vs_floor"],
                "test_diagnostics": test_result["diagnostics"],
                "test_score": float(ew._candidate_score(test_result["metric_vs_floor"], test_result["diagnostics"])),
                "candidate_usage": test_result["candidate_usage"],
                "mixed_group_selection": test_result["mixed_group_selection"],
                "bootstrap": test_result["bootstrap"],
                "by_domain": test_result["by_domain"],
            }
        )
    eligible = [row for row in mode_rows if int(row["val_mixed_group_selection"]["mixed_group_count"]) == 0]
    selected = max(eligible or mode_rows, key=lambda row: float(row["val_score"]))
    val_group_scores = _unique_group_scores(risk["group_score"], group_key, val_ids)
    test_group_scores = _unique_group_scores(risk["group_score"], group_key, test_ids)
    return {
        "candidate_count": len(candidates),
        "candidate_names": sorted(val_cache.keys()),
        "mode_rows": mode_rows,
        "selected": selected,
        "selection_policy": {
            "source": "validation_only",
            "constraint": "continuous risk thresholds are validation-derived and selected mode must have zero mixed group selection on validation",
            "eligible_modes": [row["mode"] for row in eligible],
        },
        "stage42_am_rebuilt": {
            "lambda": am_candidate["lambda"],
            "feature_count": am_candidate["feature_count"],
            "policy_slice_count": len(am_candidate["policy"]["slices"]),
            "val_metric": am_candidate["val_metric"],
        },
        "risk_score_stats": {
            "val_group_min": float(np.min(val_group_scores)) if len(val_group_scores) else 0.0,
            "val_group_median": float(np.median(val_group_scores)) if len(val_group_scores) else 0.0,
            "val_group_max": float(np.max(val_group_scores)) if len(val_group_scores) else 0.0,
            "test_group_min": float(np.min(test_group_scores)) if len(test_group_scores) else 0.0,
            "test_group_median": float(np.median(test_group_scores)) if len(test_group_scores) else 0.0,
            "test_group_max": float(np.max(test_group_scores)) if len(test_group_scores) else 0.0,
            "val_unique_group_scores": int(len(set(np.round(val_group_scores, 8).tolist()))),
            "test_unique_group_scores": int(len(set(np.round(test_group_scores, 8).tolist()))),
        },
    }


def _compare_to_prior(metric: Mapping[str, Any]) -> dict[str, Any]:
    di_payload = read_json(DI_JSON, {})
    ex_payload = read_json(EX_JSON, {})
    di_metric = di_payload.get("repair", {}).get("test", {}).get("metric_vs_floor", {})
    ex_metric = ex_payload.get("group_level_repair", {}).get("selected", {}).get("test_metric_vs_floor", {})

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

    return {
        "stage42_di_metric": di_metric,
        "stage42_ex_metric": ex_metric,
        "delta_vs_stage42_di": delta(di_metric),
        "delta_vs_stage42_ex": delta(ex_metric),
    }


def _deployment_decision(metric: Mapping[str, Any], comparison: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    delta_di = comparison["delta_vs_stage42_di"]
    group_consistent = int(selected["mixed_group_selection"]["mixed_group_count"]) == 0
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
        "promote_continuous_group_risk_repair": bool(promotes),
        "diagnostic_positive": bool(useful),
        "decision": "promote_stage42_ey_continuous_group_risk_repair"
        if promotes
        else ("stage42_ey_continuous_group_risk_repair_positive_not_promoted" if useful else "stage42_ey_continuous_group_risk_repair_not_enough_keep_stage42_di_or_cq_floor"),
        "reason": "Promotion requires validation-derived continuous group-risk buckets to beat Stage42-DI on all and hard, preserve easy, not worsen near@0.05, and keep one repair choice per source/frame/horizon group.",
        "group_consistent_selection": bool(group_consistent),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    selected = payload["continuous_group_risk_repair"]["selected"]
    metric = selected["test_metric_vs_floor"]
    diag = selected["test_diagnostics"]
    delta_di = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    decision = payload["deployment_decision"]
    stats = payload["continuous_group_risk_repair"]["risk_score_stats"]
    gates = {
        "repair_candidates_evaluated": payload["continuous_group_risk_repair"]["candidate_count"] >= 40,
        "continuous_group_risk_built": stats["val_unique_group_scores"] > 1,
        "risk_buckets_non_degenerate": any(len(row["bucket_counts_val"]) >= 2 for row in payload["continuous_group_risk_repair"]["mode_rows"] if "risk" in row["mode"]),
        "adaptive_modes_evaluated": len(payload["continuous_group_risk_repair"]["mode_rows"]) == 4,
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
        and no_leak["validation_only_bucket_thresholds"] is True
        and no_leak["validation_only_rule_selection"] is True
        and no_leak["source_overlap_pass"] is True,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage42_ey_continuous_group_risk_repair_pass_promotable"
    elif gates["test_all_positive_vs_floor"] and gates["test_hard_positive_vs_floor"] and gates["easy_degradation_under_2pct"]:
        verdict = "stage42_ey_continuous_group_risk_repair_positive_not_promoted"
    else:
        verdict = "stage42_ey_continuous_group_risk_repair_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    source_split = s42b.build_stage42_source_split()
    data = s41._combined()
    split, group = am._split_arrays(data)
    split_stats = am._source_stats(data, split, group)
    labels = am._reconstruct_waypoint_labels(data)
    floor = am._floor_arrays(data, split == "train")
    repair = _continuous_risk_repair(data, split, labels, floor)
    metric = repair["selected"]["test_metric_vs_floor"]
    comparison = _compare_to_prior(metric)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EY continuous group-risk adaptive repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(["data/stage41_world_model/combined_external.npz", str(DI_JSON), str(EX_JSON)]),
        "current_facts": CURRENT_FACTS,
        "source_split": source_split,
        "split_stats": split_stats,
        "continuous_group_risk_repair": repair,
        "comparison_to_prior": comparison,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "risk_score_predicted_rollout_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_bucket_thresholds": True,
            "validation_only_rule_selection": True,
            "validation_only_mode_selection": True,
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(split_stats["source_overlap_pass"]),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["deployment_decision"] = _deployment_decision(metric, comparison, repair["selected"])
    payload["paper_file_status"] = _refresh_paper_files(payload)
    payload["stage42_ey_gate"] = _gate(payload)
    return payload


def _paper_lines(payload: Mapping[str, Any]) -> list[str]:
    selected = payload["continuous_group_risk_repair"]["selected"]
    metric = selected["test_metric_vs_floor"]
    diag = selected["test_diagnostics"]
    delta = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    return [
        "## Stage42-EY Continuous Group-Risk Repair",
        "",
        "- source: `fresh_stage42_continuous_group_risk_repair`",
        "- role: validation-only continuous group-risk bucket repair over Stage42-DI repair candidates.",
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
            _replace_section(path, "STAGE42_EY_CONTINUOUS_GROUP_RISK_REPAIR", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "updated": True,
                    "contains_stage42_ey": "STAGE42_EY_CONTINUOUS_GROUP_RISK_REPAIR" in text,
                    "contains_boundaries": "no Stage5C" in text and "no SMC" in text,
                }
            )
        else:
            status.append({"path": str(path), "updated": False, "missing": True})
    return status


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    selected = payload["continuous_group_risk_repair"]["selected"]
    metric = selected["test_metric_vs_floor"]
    diag = selected["test_diagnostics"]
    delta_di = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    delta_ex = payload["comparison_to_prior"]["delta_vs_stage42_ex"]
    lines = [
        "# Stage42-EY Continuous Group-Risk Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ey_gate']['passed']} / {payload['stage42_ey_gate']['total']}`",
        f"- verdict: `{payload['stage42_ey_gate']['verdict']}`",
        f"- decision: `{payload['deployment_decision']['decision']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Selected Repair",
        "",
        f"- mode: `{selected['mode']}`",
        f"- validation_score: `{selected['val_score']:.6f}`",
        f"- test_score: `{selected['test_score']:.6f}`",
        f"- thresholds: `{selected.get('thresholds', [])}`",
        f"- bucket_counts_val: `{selected.get('bucket_counts_val', {})}`",
        f"- bucket_counts_test: `{selected.get('bucket_counts_test', {})}`",
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
        "| mode | val score | test score | all | t50 | t100 raw | hard | easy | mixed groups | buckets val |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["continuous_group_risk_repair"]["mode_rows"]:
        m = row["test_metric_vs_floor"]
        mixed = row["mixed_group_selection"]
        lines.append(
            f"| `{row['mode']}` | {row['val_score']:.6f} | {row['test_score']:.6f} | {_pct(m['all_improvement'])} | "
            f"{_pct(m['t50_improvement'])} | {_pct(m['t100_raw_frame_diagnostic_improvement'])} | "
            f"{_pct(m['hard_failure_improvement'])} | {_pct(m['easy_degradation'])} | {_pct(mixed['mixed_group_rate'])} | `{row['bucket_counts_val']}` |"
        )
    lines.extend(
        [
            "",
            "## Delta vs Prior",
            "",
            "| prior | all | t50 | t100 raw | hard | easy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| `Stage42-DI` | {_pct(delta_di['all_improvement'])} | {_pct(delta_di['t50_improvement'])} | {_pct(delta_di['t100_raw_frame_diagnostic_improvement'])} | {_pct(delta_di['hard_failure_improvement'])} | {_pct(delta_di['easy_degradation'])} |",
            f"| `Stage42-EX` | {_pct(delta_ex['all_improvement'])} | {_pct(delta_ex['t50_improvement'])} | {_pct(delta_ex['t100_raw_frame_diagnostic_improvement'])} | {_pct(delta_ex['hard_failure_improvement'])} | {_pct(delta_ex['easy_degradation'])} |",
            "",
            "## Risk Score Stats",
            "",
            f"- risk_score_stats: `{payload['continuous_group_risk_repair']['risk_score_stats']}`",
            "",
            "## Bootstrap CI",
            "",
            "| slice | low | mid | high | n |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, row in selected["bootstrap"].items():
        lines.append(f"| `{key}` | {row['low']:.6f} | {row['mid']:.6f} | {row['high']:.6f} | {row['n']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-EY tests whether EX failed because binary group-risk saturated.",
            "- Continuous risk buckets are validation-derived and frozen before test.",
            "- If EY does not beat Stage42-DI, risk-adaptive repair is not currently a better path than DI's global group-consistency repair.",
            "- This remains source-level raw-frame 2.5D evidence, not metric/seconds-level, true 3D, Stage5C, or SMC evidence.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_ey_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ey_gate"]
    return [
        "# Stage42-EY Gate",
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
    selected = payload["continuous_group_risk_repair"]["selected"]
    metric = selected["test_metric_vs_floor"]
    diag = selected["test_diagnostics"]
    delta = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    return [
        "## Stage42-EY Continuous Group-Risk Repair",
        "",
        "- source: `fresh_stage42_continuous_group_risk_repair`",
        "- role: validation-only continuous group-risk bucket repair over Stage42-DI repair candidates.",
        f"- gate: `{payload['stage42_ey_gate']['passed']} / {payload['stage42_ey_gate']['total']}`; verdict `{payload['stage42_ey_gate']['verdict']}`.",
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
        _replace_section(path, "STAGE42_EY_CONTINUOUS_GROUP_RISK_REPAIR", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EY continuous group-risk repair"
    state["current_verdict"] = payload["stage42_ey_gate"]["verdict"]
    state["stage42_ey_continuous_group_risk_repair"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ey_gate"]["verdict"],
        "gates": f"{payload['stage42_ey_gate']['passed']}/{payload['stage42_ey_gate']['total']}",
        "selected": payload["continuous_group_risk_repair"]["selected"],
        "comparison_to_prior": payload["comparison_to_prior"],
        "deployment_decision": payload["deployment_decision"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_continuous_group_risk_repair(*, refresh_readmes: bool = True) -> dict[str, Any]:
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
    run_stage42_continuous_group_risk_repair()
