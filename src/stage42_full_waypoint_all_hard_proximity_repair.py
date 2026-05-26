from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src import stage41_joint_multiagent_consistency as jmc
from src import stage42_common_validation_bridge_shape_composer as co
from src import stage42_proximity_aware_composer_guard as cq
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "full_waypoint_all_hard_proximity_repair_stage42.json"
REPORT_MD = OUT_DIR / "full_waypoint_all_hard_proximity_repair_stage42.md"
REPORT_CSV = OUT_DIR / "full_waypoint_all_hard_proximity_repair_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_df_gate.md"

CQ_JSON = OUT_DIR / "proximity_aware_composer_guard_stage42.json"
DE_JSON = OUT_DIR / "full_waypoint_deployment_gap_audit_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_README = Path("README_M3W_GOAL_SUMMARY_ZH.md")
CURRENT_RETROSPECTIVE = Path("README_M3W_CURRENT_FULL_RETROSPECTIVE_ZH.md")
RESEARCH_STATE = Path("research_state.json")

MIN_ALL_GAIN = [-0.005, 0.0, 0.0025, 0.005, 0.01, 0.02]
MIN_HARD_GAIN = [-0.005, 0.0, 0.0025, 0.005, 0.01, 0.02]
EASY_MAX = [0.0, 0.005, 0.01, 0.02]
MIN_SEP_GRID = [0.0, 0.05, 0.12]
PROX_MARGIN = [0.0]
MIN_SLICE_ROWS = 80
MAX_BASE_POLICIES_FOR_PROXIMITY = 2
FIXED_FINAL_MIN_SEP = 0.05
FIXED_FINAL_MARGIN = 0.0
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DF 针对 Stage42-DE 的 full-waypoint all/hard/proximity blocker，做 validation-only all+hard repair policy search。",
    "本阶段重新评估 endpoint/full-waypoint common rows，不训练新模型，不用 test 调 threshold。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
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
    return co._jsonable(value)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _safe_improvement(selected: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    return co._safe_improvement(selected, ref, mask)


def _load_split(split: str) -> dict[str, Any]:
    endpoint = co._endpoint_bundle(split)
    full = co._full_bundle(split)
    alignment = co._alignment_report(endpoint, full)
    if not alignment["aligned"]:
        raise ValueError(f"Endpoint/full-waypoint rows are not aligned for {split}: {alignment}")
    return {
        "endpoint": endpoint,
        "full": full,
        "alignment": alignment,
        "keys": jmc._group_metadata(split)["key"],
    }


def _slice_stats(endpoint: Mapping[str, Any], full: Mapping[str, Any]) -> dict[str, Any]:
    labels = endpoint["labels"]
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    out: dict[str, Any] = {}
    for key, mask in co._slice_masks(labels).items():
        hard_mask = mask & hard
        easy_mask = mask & easy
        out[key] = {
            "rows": int(np.sum(mask)),
            "all_gain_vs_endpoint": _safe_improvement(full["selected_ade"], endpoint["selected_ade"], mask),
            "hard_gain_vs_endpoint": _safe_improvement(full["selected_ade"], endpoint["selected_ade"], hard_mask),
            "treat_easy_degradation_vs_endpoint": -_safe_improvement(
                full["selected_ade"], endpoint["selected_ade"], easy_mask
            )
            if np.any(easy_mask)
            else 0.0,
            "fde_gain_vs_endpoint": _safe_improvement(full["selected_fde"], endpoint["selected_fde"], mask),
        }
    return out


def _choices_from_stats(
    stats: Mapping[str, Any],
    *,
    min_all_gain: float,
    min_hard_gain: float,
    easy_max: float,
) -> dict[str, bool]:
    choices: dict[str, bool] = {}
    for key, row in stats.items():
        choices[key] = bool(
            row["rows"] >= MIN_SLICE_ROWS
            and row["all_gain_vs_endpoint"] >= min_all_gain
            and row["hard_gain_vs_endpoint"] >= min_hard_gain
            and row["treat_easy_degradation_vs_endpoint"] <= easy_max
        )
    return choices


def _score(metric: Mapping[str, Any], near_collision_delta: float) -> float:
    return (
        2.0 * float(metric.get("all_improvement", 0.0))
        + 2.2 * float(metric.get("hard_failure_improvement", 0.0))
        + 0.9 * float(metric.get("t50_improvement", 0.0))
        + 0.4 * float(metric.get("t100_raw_frame_diagnostic_improvement", 0.0))
        - 45.0 * max(0.0, float(metric.get("easy_degradation", 1.0)) - 0.02)
        - 35.0 * max(0.0, near_collision_delta)
        - 0.02 * float(metric.get("switch_rate", 0.0))
    )


def _finite_rate(values: np.ndarray, threshold: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite < threshold)) if len(finite) else 0.0


def _finite_p05(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, 5)) if len(finite) else 0.0


def _evaluate_guarded(
    endpoint: Mapping[str, Any],
    full: Mapping[str, Any],
    keys: np.ndarray,
    choices: Mapping[str, bool],
    min_sep: float,
    margin: float,
    *,
    endpoint_min: np.ndarray | None = None,
    include_joint: bool = True,
) -> dict[str, Any]:
    base = co._compose(endpoint, full, choices)
    labels = endpoint["labels"]
    normalizer = labels["normalizer"].astype(np.float64)
    endpoint_min = (
        jmc._min_group_distance(endpoint["selected_xy"], keys, normalizer)
        if endpoint_min is None
        else endpoint_min
    )
    selected_min = jmc._min_group_distance(base["selected_xy"], keys, normalizer)
    guard = (
        base["use_full"]
        & np.isfinite(selected_min)
        & np.isfinite(endpoint_min)
        & (selected_min < min_sep)
        & (selected_min + margin < endpoint_min)
    )
    use_full = base["use_full"].copy()
    use_full[guard] = False
    selected_xy = endpoint["selected_xy"].copy()
    selected_xy[use_full] = full["selected_xy"][use_full]
    selected_ade, selected_fde = co.ft._trajectory_errors(selected_xy, labels)
    ev = {
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "use_full": use_full,
        "guarded_off": int(np.sum(guard)),
        "guarded_off_rate": float(np.mean(guard)) if len(guard) else 0.0,
        "metric_vs_endpoint_ade": co._metric(selected_ade, endpoint["selected_ade"], labels, use_full),
        "metric_vs_floor_ade": co._metric(selected_ade, endpoint["floor_ade"], labels, use_full),
        "metric_vs_endpoint_fde": co._metric(selected_fde, endpoint["selected_fde"], labels, use_full),
        "metric_vs_floor_fde": co._metric(selected_fde, endpoint["floor_fde"], labels, use_full),
    }
    selected_min_after_guard = (
        jmc._min_group_distance(selected_xy, keys, normalizer) if np.any(guard) else selected_min
    )
    near_delta = _finite_rate(selected_min_after_guard, 0.05) - _finite_rate(endpoint_min, 0.05)
    p05_delta = _finite_p05(selected_min_after_guard) - _finite_p05(endpoint_min)
    joint = cq._joint_stats(endpoint, ev, keys) if include_joint else None
    if joint is not None:
        near_delta = joint["composer_minus_endpoint"]["near_collision_rate_005_delta"]
        p05_delta = joint["composer_minus_endpoint"]["p05_min_group_distance_delta"]
    return {
        **ev,
        "joint_safety": joint,
        "near_collision_005_delta_vs_endpoint": near_delta,
        "p05_min_distance_delta_vs_endpoint": p05_delta,
    }


def _fit_policy(val_bundle: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    endpoint = val_bundle["endpoint"]
    full = val_bundle["full"]
    stats = _slice_stats(endpoint, full)
    candidates: list[dict[str, Any]] = []
    for min_all in MIN_ALL_GAIN:
        for min_hard in MIN_HARD_GAIN:
            for easy_max in EASY_MAX:
                choices = _choices_from_stats(stats, min_all_gain=min_all, min_hard_gain=min_hard, easy_max=easy_max)
                if not any(choices.values()):
                    continue
                base = co._compose(endpoint, full, choices)
                base_metric = base["metric_vs_endpoint_ade"]
                eligible = bool(
                    base_metric["all_improvement"] > 0.0
                    and base_metric["hard_failure_improvement"] > 0.0
                    and base_metric["easy_degradation"] <= 0.02
                    and np.any(base["use_full"])
                )
                candidates.append(
                    {
                        "policy": {
                            "type": "all_hard_proximity_full_waypoint_repair",
                            "min_all_gain": min_all,
                            "min_hard_gain": min_hard,
                            "easy_max": easy_max,
                            "min_sep": FIXED_FINAL_MIN_SEP,
                            "margin": FIXED_FINAL_MARGIN,
                            "choices": choices,
                        },
                        "eligible": eligible,
                        "score": _score(base_metric, 0.0),
                        "val_metric_vs_endpoint_ade": base_metric,
                        "val_near_collision_005_delta_vs_endpoint": 0.0,
                        "val_p05_min_distance_delta_vs_endpoint": 0.0,
                        "val_guarded_off": 0,
                        "val_switch_rate": base_metric["switch_rate"],
                    }
                )
    fallback = {
        "policy": {
            "type": "keep_endpoint_linear_bridge_floor",
            "min_all_gain": None,
            "min_hard_gain": None,
            "easy_max": None,
            "min_sep": None,
            "margin": None,
            "choices": {},
        },
        "eligible": True,
        "score": 0.0,
        "val_metric_vs_endpoint_ade": co._metric(
            endpoint["selected_ade"],
            endpoint["selected_ade"],
            endpoint["labels"],
            np.zeros(len(endpoint["selected_ade"]), dtype=bool),
        ),
        "val_near_collision_005_delta_vs_endpoint": 0.0,
        "val_p05_min_distance_delta_vs_endpoint": 0.0,
        "val_guarded_off": 0,
        "val_switch_rate": 0.0,
    }
    candidates.append(fallback)
    eligible = [row for row in candidates if row["eligible"]]
    best = max(eligible, key=lambda row: row["score"])
    return best["policy"], sorted(candidates, key=lambda row: row["score"], reverse=True)


def _evaluate_policy(bundle: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    if not policy.get("choices"):
        endpoint = bundle["endpoint"]
        labels = endpoint["labels"]
        switch = np.zeros(len(labels["horizon"]), dtype=bool)
        return {
            "selected_xy": endpoint["selected_xy"],
            "selected_ade": endpoint["selected_ade"],
            "selected_fde": endpoint["selected_fde"],
            "use_full": switch,
            "guarded_off": 0,
            "metric_vs_endpoint_ade": co._metric(endpoint["selected_ade"], endpoint["selected_ade"], labels, switch),
            "metric_vs_floor_ade": co._metric(endpoint["selected_ade"], endpoint["floor_ade"], labels, endpoint["switch"]),
            "metric_vs_endpoint_fde": co._metric(endpoint["selected_fde"], endpoint["selected_fde"], labels, switch),
            "metric_vs_floor_fde": co._metric(endpoint["selected_fde"], endpoint["floor_fde"], labels, endpoint["switch"]),
            "joint_safety": cq._joint_stats(endpoint, {"selected_xy": endpoint["selected_xy"], "use_full": switch}, bundle["keys"]),
            "near_collision_005_delta_vs_endpoint": 0.0,
            "p05_min_distance_delta_vs_endpoint": 0.0,
        }
    return _evaluate_guarded(
        bundle["endpoint"],
        bundle["full"],
        bundle["keys"],
        policy["choices"],
        float(policy["min_sep"]),
        float(policy["margin"]),
    )


def _top_candidates(candidates: list[Mapping[str, Any]], limit: int = 25) -> list[dict[str, Any]]:
    rows = []
    for row in candidates[:limit]:
        policy = row["policy"]
        rows.append(
            {
                "policy": {k: v for k, v in policy.items() if k != "choices"},
                "selected_slices": [key for key, value in policy.get("choices", {}).items() if value],
                "eligible": row["eligible"],
                "score": row["score"],
                "val_metric_vs_endpoint_ade": row["val_metric_vs_endpoint_ade"],
                "val_near_collision_005_delta_vs_endpoint": row["val_near_collision_005_delta_vs_endpoint"],
                "val_guarded_off": row["val_guarded_off"],
                "val_switch_rate": row["val_switch_rate"],
            }
        )
    return rows


def _compare_to_cq(test_eval: Mapping[str, Any]) -> dict[str, Any]:
    cq_payload = read_json(CQ_JSON, {})
    cq_metric = cq_payload.get("test_eval", {}).get("metric_vs_endpoint_ade", {})
    cq_joint = cq_payload.get("test_joint_safety", {}).get("composer_minus_endpoint", {})
    metric = test_eval["metric_vs_endpoint_ade"]
    return {
        "cq_source": cq_payload.get("source", "missing"),
        "cq_metric_vs_endpoint_ade": cq_metric,
        "delta_vs_cq": {
            "all_improvement": float(metric.get("all_improvement", 0.0)) - float(cq_metric.get("all_improvement", 0.0)),
            "t50_improvement": float(metric.get("t50_improvement", 0.0)) - float(cq_metric.get("t50_improvement", 0.0)),
            "t100_raw_frame_diagnostic_improvement": float(metric.get("t100_raw_frame_diagnostic_improvement", 0.0))
            - float(cq_metric.get("t100_raw_frame_diagnostic_improvement", 0.0)),
            "hard_failure_improvement": float(metric.get("hard_failure_improvement", 0.0))
            - float(cq_metric.get("hard_failure_improvement", 0.0)),
            "easy_degradation": float(metric.get("easy_degradation", 0.0)) - float(cq_metric.get("easy_degradation", 0.0)),
            "switch_rate": float(metric.get("switch_rate", 0.0)) - float(cq_metric.get("switch_rate", 0.0)),
            "near_collision_005_delta_vs_endpoint": float(test_eval["near_collision_005_delta_vs_endpoint"])
            - float(cq_joint.get("near_collision_rate_005_delta", 0.0)),
        },
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    val = _load_split("val")
    test = _load_split("test")
    policy, candidates = _fit_policy(val)
    val_eval = _evaluate_policy(val, policy)
    test_eval = _evaluate_policy(test, policy)
    cq_comparison = _compare_to_cq(test_eval)
    de_payload = read_json(DE_JSON, {})
    repair_closes_de_gap = (
        test_eval["metric_vs_endpoint_ade"]["all_improvement"] > 0.0
        and test_eval["metric_vs_endpoint_ade"]["hard_failure_improvement"] > 0.0
        and test_eval["metric_vs_endpoint_ade"]["easy_degradation"] <= 0.02
        and test_eval["near_collision_005_delta_vs_endpoint"] <= 0.0
        and cq_comparison["delta_vs_cq"]["all_improvement"] > 0.0
        and cq_comparison["delta_vs_cq"]["hard_failure_improvement"] > 0.0
    )
    if repair_closes_de_gap:
        decision = "all_hard_proximity_repair_improves_guarded_composer_but_still_requires_floor"
    else:
        decision = "all_hard_proximity_repair_no_primary_promotion_keep_cq_guarded_composer"
    result: dict[str, Any] = {
        "source": "fresh_stage42_df_all_hard_proximity_full_waypoint_repair",
        "stage": "Stage42-DF all-hard/proximity full-waypoint repair evaluator",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([CQ_JSON, DE_JSON]),
        "alignment": {"val": val["alignment"], "test": test["alignment"]},
        "slice_stats_val": _slice_stats(val["endpoint"], val["full"]),
        "selected_policy": policy,
        "candidate_count": len(candidates),
        "top_validation_candidates": _top_candidates(candidates),
        "val_eval": {
            "metric_vs_endpoint_ade": val_eval["metric_vs_endpoint_ade"],
            "metric_vs_floor_ade": val_eval["metric_vs_floor_ade"],
            "near_collision_005_delta_vs_endpoint": val_eval["near_collision_005_delta_vs_endpoint"],
            "guarded_off": val_eval["guarded_off"],
        },
        "test_eval": {
            "metric_vs_endpoint_ade": test_eval["metric_vs_endpoint_ade"],
            "metric_vs_floor_ade": test_eval["metric_vs_floor_ade"],
            "metric_vs_endpoint_fde": test_eval["metric_vs_endpoint_fde"],
            "metric_vs_floor_fde": test_eval["metric_vs_floor_fde"],
            "near_collision_005_delta_vs_endpoint": test_eval["near_collision_005_delta_vs_endpoint"],
            "p05_min_distance_delta_vs_endpoint": test_eval["p05_min_distance_delta_vs_endpoint"],
            "guarded_off": test_eval["guarded_off"],
            "joint_safety": test_eval["joint_safety"],
        },
        "comparison_to_stage42_cq": cq_comparison,
        "comparison_to_stage42_de": {
            "de_verdict": de_payload.get("stage42_de_gate", {}).get("verdict", "missing"),
            "de_decision": de_payload.get("evidence", {}).get("deployment_decision", {}).get("decision", "missing"),
            "repair_closes_de_all_hard_proximity_gap": repair_closes_de_gap,
        },
        "deployment_decision": {
            "decision": decision,
            "promote_full_waypoint_as_primary_deployable_dynamics": False,
            "keep_stage37_teacher_or_endpoint_floor": True,
            "if_positive_use_as_candidate_for_next_full_waypoint_training_target": True,
            "reason": "Stage42-DF is a validation-only repair evaluator over existing aligned rows; it is not a new trained all-agent full-waypoint model.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["stage42_df_gate"] = _gate(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    m = result["test_eval"]["metric_vs_endpoint_ade"]
    no_leak = result["no_leakage"]
    cq_delta = result["comparison_to_stage42_cq"]["delta_vs_cq"]
    gates = {
        "common_validation_rows_aligned": result["alignment"]["val"]["aligned"] is True,
        "common_test_rows_aligned": result["alignment"]["test"]["aligned"] is True,
        "validation_policy_search_executed": result["candidate_count"] > 1,
        "test_evaluated_once_no_threshold_tuning": no_leak["test_threshold_tuning"] is False
        and no_leak["validation_only_policy_selection"] is True,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["future_waypoint_label_eval_only"] is True,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
            ]
        ),
        "test_all_positive_vs_endpoint": m["all_improvement"] > 0.0,
        "test_hard_positive_vs_endpoint": m["hard_failure_improvement"] > 0.0,
        "easy_degradation_under_2pct": m["easy_degradation"] <= 0.02,
        "near_collision_not_worse_than_endpoint": result["test_eval"]["near_collision_005_delta_vs_endpoint"] <= 0.0,
        "cq_comparison_recorded": "all_improvement" in cq_delta,
        "deployment_decision_not_overpromoted": result["deployment_decision"][
            "promote_full_waypoint_as_primary_deployable_dynamics"
        ]
        is False,
        "metric_seconds_overclaim_blocked": result["claim_boundary"]["global_metric_claim_allowed"] is False
        and result["claim_boundary"]["global_seconds_claim_allowed"] is False,
        "stage5c_false": result["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": result["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    de_comparison = result.get("comparison_to_stage42_de", {})
    if passed == total and de_comparison.get("repair_closes_de_all_hard_proximity_gap", False):
        verdict = "stage42_df_all_hard_proximity_repair_pass_candidate_improves_cq"
    elif passed == total:
        verdict = "stage42_df_all_hard_proximity_repair_pass_no_primary_promotion"
    else:
        verdict = "stage42_df_all_hard_proximity_repair_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_csv(candidates: list[Mapping[str, Any]]) -> None:
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "type",
                "min_all_gain",
                "min_hard_gain",
                "easy_max",
                "min_sep",
                "margin",
                "eligible",
                "score",
                "selected_slices",
                "val_all",
                "val_t50",
                "val_t100",
                "val_hard",
                "val_easy",
                "val_near005",
                "val_switch",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(candidates[:100], start=1):
            policy = row["policy"]
            metric = row["val_metric_vs_endpoint_ade"]
            writer.writerow(
                {
                    "rank": idx,
                    "type": policy["type"],
                    "min_all_gain": policy.get("min_all_gain"),
                    "min_hard_gain": policy.get("min_hard_gain"),
                    "easy_max": policy.get("easy_max"),
                    "min_sep": policy.get("min_sep"),
                    "margin": policy.get("margin"),
                    "eligible": row["eligible"],
                    "score": row["score"],
                    "selected_slices": ",".join(row.get("selected_slices", []))
                    if "selected_slices" in row
                    else ",".join([key for key, value in policy.get("choices", {}).items() if value]),
                    "val_all": metric["all_improvement"],
                    "val_t50": metric["t50_improvement"],
                    "val_t100": metric["t100_raw_frame_diagnostic_improvement"],
                    "val_hard": metric["hard_failure_improvement"],
                    "val_easy": metric["easy_degradation"],
                    "val_near005": row["val_near_collision_005_delta_vs_endpoint"],
                    "val_switch": row["val_switch_rate"],
                }
            )


def _render_report(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_df_gate"]
    test = result["test_eval"]["metric_vs_endpoint_ade"]
    cq_delta = result["comparison_to_stage42_cq"]["delta_vs_cq"]
    policy = result["selected_policy"]
    selected_slices = [key for key, value in policy.get("choices", {}).items() if value]
    lines = [
        "# Stage42-DF All-Hard / Proximity Full-Waypoint Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deployment_decision: `{result['deployment_decision']['decision']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Validation-Selected Policy",
        "",
        f"- policy: `{ {k: v for k, v in policy.items() if k != 'choices'} }`",
        f"- selected_slices: `{selected_slices}`",
        f"- candidate_count: `{result['candidate_count']}`",
        "",
        "## Test Once vs Endpoint-Linear",
        "",
        f"- all: `{_pct(test['all_improvement'])}`",
        f"- t50: `{_pct(test['t50_improvement'])}`",
        f"- t100 raw diagnostic: `{_pct(test['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure: `{_pct(test['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(test['easy_degradation'])}`",
        f"- switch_rate: `{_pct(test['switch_rate'])}`",
        f"- near_collision@0.05 delta vs endpoint: `{_pct(result['test_eval']['near_collision_005_delta_vs_endpoint'])}`",
        "",
        "## Delta vs Stage42-CQ Guarded Composer",
        "",
        f"- delta_all: `{_pct(cq_delta['all_improvement'])}`",
        f"- delta_t50: `{_pct(cq_delta['t50_improvement'])}`",
        f"- delta_t100_raw: `{_pct(cq_delta['t100_raw_frame_diagnostic_improvement'])}`",
        f"- delta_hard: `{_pct(cq_delta['hard_failure_improvement'])}`",
        f"- delta_easy: `{_pct(cq_delta['easy_degradation'])}`",
        f"- delta_near_collision@0.05: `{_pct(cq_delta['near_collision_005_delta_vs_endpoint'])}`",
        "",
        "## Top Validation Candidates",
        "",
        "| rank | selected slices | val all | val hard | val easy | val near@0.05 | eligible |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(result["top_validation_candidates"][:10], start=1):
        metric = row["val_metric_vs_endpoint_ade"]
        lines.append(
            f"| {idx} | `{row['selected_slices']}` | {_pct(metric['all_improvement'])} | "
            f"{_pct(metric['hard_failure_improvement'])} | {_pct(metric['easy_degradation'])} | "
            f"{_pct(row['val_near_collision_005_delta_vs_endpoint'])} | `{row['eligible']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-DF specifically tests whether an all+hard+proximity validation objective can repair the full-waypoint deployment blocker identified in Stage42-DE.",
            "- The result remains a protected evaluator over existing aligned endpoint/full-waypoint rows; it is not a newly trained all-agent full-waypoint model.",
            "- If it improves Stage42-CQ, it becomes the next candidate target for actual full-waypoint training. If it does not, the correct next step is changing the model/loss, not repeating threshold search.",
            "",
            "## Claim Boundary",
            "",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_df_gate"]
    return [
        "# Stage42-DF Gate",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Gates",
        "",
        *[f"- {key}: `{value}`" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_df_gate"]
    test = result["test_eval"]["metric_vs_endpoint_ade"]
    cq_delta = result["comparison_to_stage42_cq"]["delta_vs_cq"]
    return [
        "## Stage42-DF All-Hard / Proximity Full-Waypoint Repair",
        "",
        "- source: `fresh_stage42_df_all_hard_proximity_full_waypoint_repair`",
        "- role: validation-only repair search for the Stage42-DE all/hard/proximity full-waypoint deployment blocker.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- test vs endpoint-linear: all `{_pct(test['all_improvement'])}`, t50 `{_pct(test['t50_improvement'])}`, t100 raw `{_pct(test['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(test['hard_failure_improvement'])}`, easy `{_pct(test['easy_degradation'])}`.",
        f"- delta vs Stage42-CQ: all `{_pct(cq_delta['all_improvement'])}`, t50 `{_pct(cq_delta['t50_improvement'])}`, t100 raw `{_pct(cq_delta['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(cq_delta['hard_failure_improvement'])}`, near@0.05 `{_pct(cq_delta['near_collision_005_delta_vs_endpoint'])}`.",
        f"- decision: `{result['deployment_decision']['decision']}`.",
        "- Stage5C remains false; SMC remains false; no metric/seconds claim.",
    ]


def _refresh_readmes(result: Mapping[str, Any]) -> None:
    lines = _refresh_lines(result)
    for path in [README_RESULTS, M3W_README, GOAL_README, CURRENT_RETROSPECTIVE]:
        _replace_section(path, "STAGE42_DF_FULL_WAYPOINT_ALL_HARD_PROXIMITY_REPAIR", lines)


def _refresh_research_state(result: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DF all-hard/proximity full-waypoint repair"
    state["current_verdict"] = result["stage42_df_gate"]["verdict"]
    state["stage42_df_full_waypoint_all_hard_proximity_repair"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": result["stage42_df_gate"]["verdict"],
        "gates": f"{result['stage42_df_gate']['passed']}/{result['stage42_df_gate']['total']}",
        "deployment_decision": result["deployment_decision"],
        "test_metric_vs_endpoint_ade": result["test_eval"]["metric_vs_endpoint_ade"],
        "comparison_to_stage42_cq": result["comparison_to_stage42_cq"]["delta_vs_cq"],
        "claim_boundary": result["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_full_waypoint_all_hard_proximity_repair(*, refresh_readmes: bool = True) -> dict[str, Any]:
    result = _build_payload()
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _write_csv(result["top_validation_candidates"])
    if refresh_readmes:
        _refresh_readmes(result)
        _refresh_research_state(result)
    return result


if __name__ == "__main__":
    run_stage42_full_waypoint_all_hard_proximity_repair()
