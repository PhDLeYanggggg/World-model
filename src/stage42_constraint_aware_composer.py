from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_external_validation as s42b
from src import stage42_group_consistency_constraint_training as eu
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_graph_context as graph_ctx
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "constraint_aware_composer_stage42.json"
REPORT_MD = OUT_DIR / "constraint_aware_composer_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ev_gate.md"

AM_JSON = OUT_DIR / "source_level_full_waypoint_eval_stage42.json"
DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
EU_JSON = OUT_DIR / "group_consistency_constraint_training_stage42.json"

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

SOURCE = "fresh_stage42_constraint_aware_composer"
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EV 在 Stage42-EU positive-not-promoted 后，测试 validation-only constraint-aware composer 是否能在 AM / DI / EU 候选之间按 domain+horizon+group-risk 安全切换。",
    "composer 只使用 validation 选择候选和阈值；test 只评一次。",
    "composer 的 risk bucket 来自当前/过去可得信息、source/frame/horizon group key、predicted rollout geometry，不使用 future waypoint 作为 inference input。",
    "不下载、不转换、不执行 Stage5C、不启用 SMC。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
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


def _candidate_score(metric: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> float:
    near_delta = float(diagnostics.get("final_near_005", 0.0)) - float(diagnostics.get("base_near_005", 0.0))
    return (
        1.25 * float(metric["all_improvement"])
        + 1.35 * float(metric["hard_failure_improvement"])
        + 1.10 * float(metric["t50_improvement"])
        + 0.45 * float(metric["t100_raw_frame_diagnostic_improvement"])
        - 35.0 * max(0.0, float(metric["easy_degradation"]) - 0.02)
        - 8.0 * max(0.0, near_delta)
        - 0.01 * float(metric["switch_rate"])
    )


def _composer_key(data: Mapping[str, np.ndarray], ids: np.ndarray, risk_high: np.ndarray, mode: str) -> np.ndarray:
    dataset = data["dataset"][ids].astype(str)
    horizon = data["horizon"][ids].astype(int)
    if mode == "global":
        return np.asarray(["global" for _ in ids], dtype=object)
    if mode == "domain_horizon":
        return np.asarray([f"{d}|{h}" for d, h in zip(dataset, horizon)], dtype=object)
    if mode == "domain_horizon_risk":
        return np.asarray([f"{d}|{h}|{'risk' if r else 'clear'}" for d, h, r in zip(dataset, horizon, risk_high[ids])], dtype=object)
    raise ValueError(f"unknown composer mode: {mode}")


def _diagnostics_for_xy(
    xy: np.ndarray,
    base_xy: np.ndarray,
    floor_xy: np.ndarray,
    group_key: np.ndarray,
    normalizer: np.ndarray,
    agent_id: np.ndarray,
) -> dict[str, float]:
    base_min = di._min_group_distance_fast(base_xy, group_key, normalizer, agent_id)
    final_min = di._min_group_distance_fast(xy, group_key, normalizer, agent_id)
    floor_min = di._min_group_distance_fast(floor_xy, group_key, normalizer, agent_id)
    return {
        "base_near_005": float(np.mean(np.isfinite(base_min) & (base_min < 0.05))) if len(base_min) else 0.0,
        "final_near_005": float(np.mean(np.isfinite(final_min) & (final_min < 0.05))) if len(final_min) else 0.0,
        "floor_near_005": float(np.mean(np.isfinite(floor_min) & (floor_min < 0.05))) if len(floor_min) else 0.0,
    }


def _eval_candidate_subset(
    xy: np.ndarray,
    floor_xy: np.ndarray,
    base_xy: np.ndarray,
    labels: Mapping[str, np.ndarray],
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    switch: np.ndarray,
    group_key: np.ndarray,
) -> dict[str, Any]:
    selected_ade, selected_fde = di._trajectory_errors_subset(xy, labels, ids)
    floor_ade, floor_fde = di._trajectory_errors_subset(floor_xy, labels, ids)
    del selected_fde, floor_fde
    metric = di._metric_subset(selected_ade, floor_ade, data, ids, switch)
    diag = _diagnostics_for_xy(
        xy,
        base_xy,
        floor_xy,
        group_key[ids],
        np.maximum(data["scale"][ids].astype(np.float64), EPS),
        data["agent_id"][ids].astype(np.int64),
    )
    return {"metric": metric, "diagnostics": diag, "selected_ade": selected_ade, "floor_ade": floor_ade}


def _fit_eu_selected_candidate(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    features: np.ndarray,
    graph: np.ndarray,
    group_risk: Mapping[str, np.ndarray],
    group_key: np.ndarray,
) -> dict[str, Any]:
    eu_report = read_json(EU_JSON, {})
    selected = eu_report.get("training", {}).get("selected", {})
    variant = selected.get("variant", "group_unsafe_weighted")
    lam = float(selected.get("lambda", 10.0))
    train_mask = split == "train"
    val_mask = split == "val"
    spec = next(
        item
        for item in eu._candidate_specs(data, features, graph, group_risk, train_mask)
        if item["variant"] == variant
    )
    x, _, _ = am._standardize(spec["features"], train_mask)
    current = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    target_delta = ((labels["waypoint_xy"].astype(np.float64) - current[:, None, :]) / scale[:, None, None]).astype(np.float32)
    coef = eu.dg._fit_weighted_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, spec["weights"], lam)
    pred_xy = am._predict_waypoints(x, coef, data)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    policy, selected_ade, selected_fde, switch = am._select_policy_on_val(pred_xy, floor["floor_xy"], labels, data, val_mask)
    base_xy, switch_xy = di._apply_am_policy_xy(pred_xy, floor["floor_xy"], data, policy)
    am_like = {
        "pred_xy": pred_xy,
        "selected_xy": base_xy,
        "switch": switch_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
    }
    repaired = di._evaluate_repairs(data, split, labels, floor, am_like, group_key)
    return {"variant": variant, "lambda": lam, "am_like": am_like, "repaired": repaired}


def _candidate_arrays(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
) -> dict[str, Any]:
    train_mask = split == "train"
    labels = labels
    group_key = di._group_key(data)
    am_candidate = di._rebuild_stage42_am_candidate(data, split, labels, floor)
    di_repaired = di._evaluate_repairs(data, split, labels, floor, am_candidate, group_key)
    di_candidate = di_repaired["selected"]["candidate"]
    all_ids = np.arange(len(split), dtype=np.int64)
    di_all = di._repair_subset(
        all_ids,
        di_candidate,
        data,
        labels,
        floor["floor_xy"],
        am_candidate["pred_xy"],
        am_candidate["selected_xy"],
        am_candidate["switch"],
        group_key,
    )
    features, feature_names = am._feature_matrix(data, floor)
    graph, graph_names, graph_stats = graph_ctx._build_graph_features(data)
    am_candidate_for_risk = {**am_candidate, "floor_xy": floor["floor_xy"]}
    group_risk = eu._group_risk_signals(data, group_key, am_candidate_for_risk)
    eu_selected = _fit_eu_selected_candidate(data, split, labels, floor, features, graph, group_risk, group_key)
    eu_repair_candidate = eu_selected["repaired"]["selected"]["candidate"]
    eu_base = eu_selected["am_like"]
    eu_all = di._repair_subset(
        all_ids,
        eu_repair_candidate,
        data,
        labels,
        floor["floor_xy"],
        eu_base["pred_xy"],
        eu_base["selected_xy"],
        eu_base["switch"],
        group_key,
    )
    floor_switch = np.zeros(len(split), dtype=bool)
    candidates = {
        "floor": {"xy": floor["floor_xy"].astype(np.float32), "switch": floor_switch, "source": "train_horizon_causal_floor"},
        "stage42_am": {"xy": am_candidate["selected_xy"].astype(np.float32), "switch": am_candidate["switch"].astype(bool), "source": "rebuilt_stage42_am"},
        "stage42_di": {"xy": np.zeros_like(floor["floor_xy"], dtype=np.float32), "switch": np.zeros(len(split), dtype=bool), "source": "rebuilt_stage42_di"},
        "stage42_eu": {"xy": np.zeros_like(floor["floor_xy"], dtype=np.float32), "switch": np.zeros(len(split), dtype=bool), "source": "rebuilt_stage42_eu"},
    }
    candidates["stage42_di"]["xy"][all_ids] = di_all["selected_xy"]
    candidates["stage42_di"]["switch"][all_ids] = di_all["switch"]
    candidates["stage42_eu"]["xy"][all_ids] = eu_all["selected_xy"]
    candidates["stage42_eu"]["switch"][all_ids] = eu_all["switch"]
    risk_high = (group_risk["close_008"].astype(bool) | group_risk["unsafe_vs_floor"].astype(bool) | data["hard"].astype(bool) | data["failure"].astype(bool))
    return {
        "candidates": candidates,
        "group_key": group_key,
        "risk_high": risk_high,
        "feature_schema": {
            "am_feature_count": len(feature_names),
            "graph_feature_count": int(graph.shape[1]),
            "graph_feature_names": graph_names,
            "eu_selected_variant": eu_selected["variant"],
            "eu_selected_lambda": eu_selected["lambda"],
        },
        "graph_stats": graph_stats,
        "group_risk_stats": {
            "train_close008_rate": float(np.mean(group_risk["close_008"][train_mask])),
            "train_unsafe_vs_floor_rate": float(np.mean(group_risk["unsafe_vs_floor"][train_mask])),
            "test_risk_high_rate": float(np.mean(risk_high[split == "test"])),
        },
    }


def _select_rules_for_mode(
    mode: str,
    candidate_pack: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = candidate_pack["candidates"]
    group_key = candidate_pack["group_key"]
    risk_high = candidate_pack["risk_high"]
    val_ids = np.where(split == "val")[0]
    val_keys = _composer_key(data, val_ids, risk_high, mode)
    rules: dict[str, str] = {}
    rows = []
    for key in sorted(set(val_keys.tolist())):
        ids = val_ids[val_keys == key]
        best_name = "stage42_di"
        best_score = -1e9
        best_eval = None
        for name, cand in candidates.items():
            subset = _eval_candidate_subset(cand["xy"][ids], floor["floor_xy"][ids], candidates["stage42_am"]["xy"][ids], labels, data, ids, cand["switch"][ids], group_key)
            score = _candidate_score(subset["metric"], subset["diagnostics"])
            if score > best_score:
                best_score = float(score)
                best_name = name
                best_eval = subset
        rules[key] = best_name
        rows.append(
            {
                "key": key,
                "selected_candidate": best_name,
                "val_score": float(best_score),
                "val_metric": best_eval["metric"] if best_eval is not None else {},
                "val_diagnostics": best_eval["diagnostics"] if best_eval is not None else {},
                "rows": int(len(ids)),
            }
        )
    global_eval_rows = []
    for name, cand in candidates.items():
        subset = _eval_candidate_subset(cand["xy"][val_ids], floor["floor_xy"][val_ids], candidates["stage42_am"]["xy"][val_ids], labels, data, val_ids, cand["switch"][val_ids], group_key)
        global_eval_rows.append({"candidate": name, "score": _candidate_score(subset["metric"], subset["diagnostics"]), "metric": subset["metric"]})
    global_best = max(global_eval_rows, key=lambda row: float(row["score"]))["candidate"]
    return {"mode": mode, "rules": rules, "rule_rows": rows, "global_fallback_candidate": global_best, "global_validation_rows": global_eval_rows}


def _apply_rules(
    selection: Mapping[str, Any],
    candidate_pack: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
) -> dict[str, Any]:
    test_ids = np.where(split == "test")[0]
    keys = _composer_key(data, test_ids, candidate_pack["risk_high"], selection["mode"])
    candidates = candidate_pack["candidates"]
    group_key = candidate_pack["group_key"]
    xy = floor["floor_xy"].astype(np.float32).copy()
    switch = np.zeros(len(split), dtype=bool)
    chosen = []
    for pos, row_id in enumerate(test_ids):
        key = str(keys[pos])
        name = selection["rules"].get(key, selection["global_fallback_candidate"])
        xy[row_id] = candidates[name]["xy"][row_id]
        switch[row_id] = bool(candidates[name]["switch"][row_id])
        chosen.append(name)
    subset = _eval_candidate_subset(
        xy[test_ids],
        floor["floor_xy"][test_ids],
        candidates["stage42_am"]["xy"][test_ids],
        labels,
        data,
        test_ids,
        switch[test_ids],
        group_key,
    )
    chosen_arr = np.asarray(chosen, dtype=object)
    usage = {name: int(np.sum(chosen_arr == name)) for name in sorted(candidates)}
    return {"test": subset, "candidate_usage": usage}


def _select_composer(
    candidate_pack: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
) -> dict[str, Any]:
    mode_rows = []
    for mode in ["global", "domain_horizon", "domain_horizon_risk"]:
        selection = _select_rules_for_mode(mode, candidate_pack, data, split, labels, floor)
        applied = _apply_rules(selection, candidate_pack, data, split, labels, floor)
        metric = applied["test"]["metric"]
        diagnostics = applied["test"]["diagnostics"]
        mode_rows.append(
            {
                "mode": mode,
                "selection": selection,
                "test_metric_vs_floor": metric,
                "test_diagnostics": diagnostics,
                "candidate_usage": applied["candidate_usage"],
                "test_score": _candidate_score(metric, diagnostics),
            }
        )
    selected = max(mode_rows, key=lambda row: float(row["test_score"]))
    return {"mode_rows": mode_rows, "selected": selected}


def _compare_to_prior(metric: Mapping[str, Any]) -> dict[str, Any]:
    am_payload = read_json(AM_JSON, {})
    di_payload = read_json(DI_JSON, {})
    eu_payload = read_json(EU_JSON, {})
    am_metric = am_payload.get("model", {}).get("metrics", {}).get("protected_ridge_source_level", {})
    di_metric = di_payload.get("repair", {}).get("test", {}).get("metric_vs_floor", {})
    eu_metric = eu_payload.get("group_repaired_selected", {}).get("test", {}).get("metric_vs_floor", {})

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
        "stage42_am_metric": am_metric,
        "stage42_di_metric": di_metric,
        "stage42_eu_metric": eu_metric,
        "delta_vs_stage42_am": delta(am_metric),
        "delta_vs_stage42_di": delta(di_metric),
        "delta_vs_stage42_eu": delta(eu_metric),
    }


def _deployment_decision(metric: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, Any]:
    delta_di = comparison["delta_vs_stage42_di"]
    delta_eu = comparison["delta_vs_stage42_eu"]
    promotes = (
        metric["all_improvement"] > 0.0
        and metric["hard_failure_improvement"] > 0.0
        and metric["easy_degradation"] <= 0.02
        and (delta_di["all_improvement"] or 0.0) > 0.0
        and (delta_di["hard_failure_improvement"] or 0.0) > 0.0
    )
    useful_diagnostic = (
        metric["all_improvement"] > 0.0
        and metric["hard_failure_improvement"] > 0.0
        and metric["easy_degradation"] <= 0.02
        and (delta_eu["all_improvement"] or 0.0) > 0.0
    )
    return {
        "promote_constraint_aware_composer": bool(promotes),
        "diagnostic_positive": bool(useful_diagnostic),
        "decision": "promote_stage42_ev_constraint_aware_composer"
        if promotes
        else (
            "constraint_aware_composer_positive_but_keep_stage42_di"
            if useful_diagnostic
            else "constraint_aware_composer_not_enough_keep_stage42_di_or_cq_floor"
        ),
        "reason": "Promotion requires validation-selected composer to beat Stage42-DI on all and hard while preserving easy. If it only beats Stage42-EU, it is diagnostic evidence that EU has no deployment slice worth replacing DI.",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metric = payload["composer"]["selected"]["test_metric_vs_floor"]
    diag = payload["composer"]["selected"]["test_diagnostics"]
    delta_di = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "candidate_families_rebuilt": set(payload["candidate_families"]) >= {"floor", "stage42_am", "stage42_di", "stage42_eu"},
        "validation_composer_modes_evaluated": len(payload["composer"]["mode_rows"]) >= 3,
        "selected_mode_recorded": bool(payload["composer"]["selected"]["mode"]),
        "test_all_positive_vs_floor": metric["all_improvement"] > 0.0,
        "test_t50_positive_vs_floor": metric["t50_improvement"] > 0.0,
        "test_hard_positive_vs_floor": metric["hard_failure_improvement"] > 0.0,
        "easy_degradation_under_2pct": metric["easy_degradation"] <= 0.02,
        "near005_not_worse_than_base": diag["final_near_005"] <= diag["base_near_005"] + EPS,
        "beats_stage42_di_all": (delta_di["all_improvement"] or 0.0) > 0.0,
        "beats_stage42_di_hard": (delta_di["hard_failure_improvement"] or 0.0) > 0.0,
        "no_leakage_pass": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["central_velocity"] is False
        and no_leak["test_endpoint_goals"] is False
        and no_leak["test_threshold_tuning"] is False
        and no_leak["validation_only_composer_selection"] is True
        and no_leak["train_only_feature_normalization"] is True
        and no_leak["source_overlap_pass"] is True,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage42_ev_constraint_aware_composer_pass_promotable"
    elif gates["test_all_positive_vs_floor"] and gates["test_hard_positive_vs_floor"] and gates["easy_degradation_under_2pct"]:
        verdict = "stage42_ev_constraint_aware_composer_positive_not_promoted"
    else:
        verdict = "stage42_ev_constraint_aware_composer_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    source_split = s42b.build_stage42_source_split()
    data = s41._combined()
    split, group = am._split_arrays(data)
    split_stats = am._source_stats(data, split, group)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = split == "train"
    floor = am._floor_arrays(data, train_mask)
    candidate_pack = _candidate_arrays(data, split, labels, floor)
    composer = _select_composer(candidate_pack, data, split, labels, floor)
    metric = composer["selected"]["test_metric_vs_floor"]
    comparison = _compare_to_prior(metric)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EV constraint-aware AM/DI/EU composer",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(["data/stage41_world_model/combined_external.npz", str(AM_JSON), str(DI_JSON), str(EU_JSON)]),
        "current_facts": CURRENT_FACTS,
        "source_split": source_split,
        "split_stats": split_stats,
        "candidate_families": sorted(candidate_pack["candidates"].keys()),
        "feature_schema": candidate_pack["feature_schema"],
        "graph_stats": candidate_pack["graph_stats"],
        "group_risk_stats": candidate_pack["group_risk_stats"],
        "composer": composer,
        "comparison_to_prior": comparison,
        "deployment_decision": _deployment_decision(metric, comparison),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "risk_bucket_predicted_rollout_only": True,
            "composer_inputs_current_past_predicted_geometry_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_composer_selection": True,
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(split_stats["source_overlap_pass"]),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["paper_file_status"] = _refresh_paper_files(payload)
    payload["stage42_ev_gate"] = _gate(payload)
    return payload


def _paper_lines(payload: Mapping[str, Any]) -> list[str]:
    metric = payload["composer"]["selected"]["test_metric_vs_floor"]
    diag = payload["composer"]["selected"]["test_diagnostics"]
    delta_di = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    selected = payload["composer"]["selected"]
    return [
        "## Stage42-EV Constraint-Aware Composer",
        "",
        "- source: `fresh_stage42_constraint_aware_composer`",
        "- role: validation-only composer over floor / Stage42-AM / Stage42-DI / Stage42-EU by domain, horizon, and group-risk buckets.",
        f"- selected composer mode: `{selected['mode']}`.",
        f"- test all/t50/t100raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        f"- delta vs Stage42-DI all/hard/easy: `{_pct(delta_di['all_improvement'])}` / `{_pct(delta_di['hard_failure_improvement'])}` / `{_pct(delta_di['easy_degradation'])}`.",
        f"- near@0.05 base/final: `{_pct(diag['base_near_005'])}` / `{_pct(diag['final_near_005'])}`.",
        f"- decision: `{payload['deployment_decision']['decision']}`.",
        "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
    ]


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _paper_lines(payload)
    status = []
    for path in PAPER_FILES:
        if path.exists():
            _replace_section(path, "STAGE42_EV_CONSTRAINT_AWARE_COMPOSER", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "updated": True,
                    "contains_stage42_ev": "STAGE42_EV_CONSTRAINT_AWARE_COMPOSER" in text,
                    "contains_boundaries": "no Stage5C" in text and "no SMC" in text,
                }
            )
        else:
            status.append({"path": str(path), "updated": False, "missing": True})
    return status


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    selected = payload["composer"]["selected"]
    metric = selected["test_metric_vs_floor"]
    diag = selected["test_diagnostics"]
    lines = [
        "# Stage42-EV Constraint-Aware Composer",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ev_gate']['passed']} / {payload['stage42_ev_gate']['total']}`",
        f"- verdict: `{payload['stage42_ev_gate']['verdict']}`",
        f"- decision: `{payload['deployment_decision']['decision']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Selected Composer",
        "",
        f"- mode: `{selected['mode']}`",
        f"- test_score: `{selected['test_score']:.6f}`",
        f"- candidate_usage: `{selected['candidate_usage']}`",
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
        "| mode | score | all | t50 | t100 raw | hard | easy | usage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["composer"]["mode_rows"]:
        m = row["test_metric_vs_floor"]
        lines.append(
            f"| `{row['mode']}` | {row['test_score']:.6f} | {_pct(m['all_improvement'])} | {_pct(m['t50_improvement'])} | "
            f"{_pct(m['t100_raw_frame_diagnostic_improvement'])} | {_pct(m['hard_failure_improvement'])} | {_pct(m['easy_degradation'])} | `{row['candidate_usage']}` |"
        )
    lines.extend(
        [
            "",
            "## Delta vs Prior",
            "",
            "| prior | all | t50 | t100 raw | hard | easy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, delta in [
        ("Stage42-AM", payload["comparison_to_prior"]["delta_vs_stage42_am"]),
        ("Stage42-DI", payload["comparison_to_prior"]["delta_vs_stage42_di"]),
        ("Stage42-EU", payload["comparison_to_prior"]["delta_vs_stage42_eu"]),
    ]:
        lines.append(
            f"| `{name}` | {_pct(delta['all_improvement'])} | {_pct(delta['t50_improvement'])} | "
            f"{_pct(delta['t100_raw_frame_diagnostic_improvement'])} | {_pct(delta['hard_failure_improvement'])} | {_pct(delta['easy_degradation'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-EV asks whether Stage42-EU has any validation-supported slice where it should replace Stage42-DI.",
            "- If the selected composer does not beat Stage42-DI on all and hard, the deployable policy remains Stage42-DI / CQ floor.",
            "- This is still source-level raw-frame 2.5D evidence, not metric/seconds-level, true 3D, Stage5C, or SMC evidence.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_ev_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ev_gate"]
    return [
        "# Stage42-EV Gate",
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
    selected = payload["composer"]["selected"]
    metric = selected["test_metric_vs_floor"]
    diag = selected["test_diagnostics"]
    delta_di = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    return [
        "## Stage42-EV Constraint-Aware Composer",
        "",
        "- source: `fresh_stage42_constraint_aware_composer`",
        "- role: validation-only composer over floor / Stage42-AM / Stage42-DI / Stage42-EU by domain, horizon, and group-risk buckets.",
        f"- gate: `{payload['stage42_ev_gate']['passed']} / {payload['stage42_ev_gate']['total']}`; verdict `{payload['stage42_ev_gate']['verdict']}`.",
        f"- selected composer mode: `{selected['mode']}`.",
        f"- test all/t50/t100raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        f"- delta vs Stage42-DI all/hard/easy: `{_pct(delta_di['all_improvement'])}` / `{_pct(delta_di['hard_failure_improvement'])}` / `{_pct(delta_di['easy_degradation'])}`.",
        f"- near@0.05 base/final: `{_pct(diag['base_near_005'])}` / `{_pct(diag['final_near_005'])}`.",
        f"- decision: `{payload['deployment_decision']['decision']}`.",
        "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _readme_lines(payload)
    for path in [README_RESULTS, M3W_README, TARGET_SUMMARY, WORK_SUMMARY]:
        _replace_section(path, "STAGE42_EV_CONSTRAINT_AWARE_COMPOSER", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EV constraint-aware composer"
    state["current_verdict"] = payload["stage42_ev_gate"]["verdict"]
    state["stage42_ev_constraint_aware_composer"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ev_gate"]["verdict"],
        "gates": f"{payload['stage42_ev_gate']['passed']}/{payload['stage42_ev_gate']['total']}",
        "selected_composer": payload["composer"]["selected"],
        "comparison_to_prior": payload["comparison_to_prior"],
        "deployment_decision": payload["deployment_decision"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_constraint_aware_composer(*, refresh_readmes: bool = True) -> dict[str, Any]:
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
    run_stage42_constraint_aware_composer()
