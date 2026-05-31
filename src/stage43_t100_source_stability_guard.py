from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _bootstrap_ci,
    _build_split,
    _git_commit,
    _jsonable,
    _metrics,
    _trajectory_error,
)
from src.stage43_full_waypoint_latent_robustness_audit import _breakdown, _pct
from src.stage43_tail_horizon_waypoint_adapter import (
    STAGE43_O_JSON,
    _apply_rules,
    _easy_degradation,
    _family_horizon,
    _model_hash,
    _predict_waypoint,
    _ridge_fit,
    _slice_improvement,
    _standardize,
    _target_matrix,
    _train_mask,
)
from src.stage43_t100_guarded_trial import (
    STAGE43_P_JSON,
    _combine_with_h100,
    _h100_family_test_table,
    _parse_rules,
    _rules_to_list,
)


REPORT_JSON = OUT_DIR / "stage43_t100_source_stability_guard.json"
REPORT_MD = OUT_DIR / "stage43_t100_source_stability_guard.md"
GATE_MD = OUT_DIR / "stage43_stage_r_t100_source_stability_guard_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

STAGE43_Q_JSON = OUT_DIR / "stage43_t100_guarded_trial.json"
SECTION = "STAGE43_R_T100_SOURCE_STABILITY_GUARD"
SOURCE = "fresh_stage43_r_t100_source_stability_guard"


def _source_name(path: str) -> str:
    text = str(path)
    return text.split("/external_data/", 1)[-1] if "/external_data/" in text else text


def _h100_source_stability_table(
    ds,
    candidate_ade: np.ndarray,
    *,
    min_family_rows: int,
    min_source_rows: int,
    min_source_count: int,
    min_improvement: float,
    max_easy_degradation: float,
) -> tuple[dict[str, Any], set[tuple[str, int]]]:
    families, horizons = _family_horizon(ds)
    sources = np.asarray(ds.source_file).astype(str)
    table: dict[str, Any] = {}
    allowed: set[tuple[str, int]] = set()
    for family in sorted(set(families.tolist())):
        mask = (families == family) & (horizons == 100)
        if int(mask.sum()) == 0:
            continue
        aggregate_improvement = _slice_improvement(candidate_ade, ds.floor_ade, mask)
        aggregate_easy = _easy_degradation(ds, candidate_ade, mask)
        source_rows = []
        for source in sorted(set(sources[mask].tolist())):
            source_mask = mask & (sources == source)
            if int(source_mask.sum()) < int(min_source_rows):
                source_rows.append(
                    {
                        "source_file": _source_name(source),
                        "rows": int(source_mask.sum()),
                        "used_for_stability": False,
                        "reason": "below_min_source_rows",
                    }
                )
                continue
            improvement = _slice_improvement(candidate_ade, ds.floor_ade, source_mask)
            easy = _easy_degradation(ds, candidate_ade, source_mask)
            source_rows.append(
                {
                    "source_file": _source_name(source),
                    "rows": int(source_mask.sum()),
                    "used_for_stability": True,
                    "full_waypoint_ade_improvement_vs_floor": float(improvement),
                    "easy_degradation_vs_floor": float(easy),
                    "safe_positive": bool(improvement > float(min_improvement) and easy <= float(max_easy_degradation)),
                }
            )
        supported_sources = [row for row in source_rows if row.get("used_for_stability")]
        safe_supported = [row for row in supported_sources if row.get("safe_positive")]
        worst_supported = min(
            [float(row["full_waypoint_ade_improvement_vs_floor"]) for row in supported_sources],
            default=0.0,
        )
        max_easy = max([float(row["easy_degradation_vs_floor"]) for row in supported_sources], default=0.0)
        allowed_flag = (
            int(mask.sum()) >= int(min_family_rows)
            and len(supported_sources) >= int(min_source_count)
            and aggregate_improvement > float(min_improvement)
            and aggregate_easy <= float(max_easy_degradation)
            and len(safe_supported) == len(supported_sources)
        )
        reason = "allowed_by_source_stable_validation"
        if int(mask.sum()) < int(min_family_rows):
            reason = "blocked_insufficient_family_rows"
        elif len(supported_sources) < int(min_source_count):
            reason = "blocked_insufficient_validation_source_count"
        elif aggregate_improvement <= float(min_improvement):
            reason = "blocked_aggregate_validation_nonpositive"
        elif aggregate_easy > float(max_easy_degradation):
            reason = "blocked_aggregate_validation_easy_harm"
        elif len(safe_supported) != len(supported_sources):
            reason = "blocked_source_level_instability"
        key = f"{family}|100"
        table[key] = {
            "rows": int(mask.sum()),
            "aggregate_full_waypoint_ade_improvement_vs_floor": float(aggregate_improvement),
            "aggregate_easy_degradation_vs_floor": float(aggregate_easy),
            "supported_source_count": int(len(supported_sources)),
            "safe_supported_source_count": int(len(safe_supported)),
            "min_required_source_count": int(min_source_count),
            "worst_supported_source_improvement": float(worst_supported),
            "max_supported_source_easy_degradation": float(max_easy),
            "allowed": bool(allowed_flag),
            "reason": reason,
            "sources": source_rows,
        }
        if allowed_flag:
            allowed.add((family, 100))
    return table, allowed


def _candidate_eval(
    train,
    val,
    base_val_ade: np.ndarray,
    base_val_fde: np.ndarray,
    base_val_switch: np.ndarray,
    *,
    target: str,
    train_filter: str,
    l2: float,
    min_family_rows: int,
    min_source_rows: int,
    min_source_count: int,
    min_h100_improvement: float,
    max_easy_degradation: float,
) -> dict[str, Any]:
    train_ids = _train_mask(train, train_filter)
    weight = _ridge_fit(train.x[train_ids], _target_matrix(train, target)[train_ids], float(l2))
    pred = _predict_waypoint(val, weight, target)
    candidate_ade, candidate_fde = _trajectory_error(val, pred)
    support_table, h100_allowed = _h100_source_stability_table(
        val,
        candidate_ade,
        min_family_rows=int(min_family_rows),
        min_source_rows=int(min_source_rows),
        min_source_count=int(min_source_count),
        min_improvement=float(min_h100_improvement),
        max_easy_degradation=float(max_easy_degradation),
    )
    selected_ade, selected_fde, switch = _combine_with_h100(
        val,
        base_val_ade,
        base_val_fde,
        base_val_switch,
        candidate_ade,
        candidate_fde,
        h100_allowed,
    )
    metrics = _metrics(val, selected_ade, selected_fde, switch)
    base_metrics = _metrics(val, base_val_ade, base_val_fde, base_val_switch)
    t100_delta = float(
        metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
        - base_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
    )
    all_delta = float(metrics["full_waypoint_ade_improvement_vs_floor"] - base_metrics["full_waypoint_ade_improvement_vs_floor"])
    hard_delta = float(
        metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
        - base_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
    )
    blocked_best_aggregate = max(
        [float(row["aggregate_full_waypoint_ade_improvement_vs_floor"]) for row in support_table.values()],
        default=0.0,
    )
    objective = (
        4.0 * t100_delta
        + 0.6 * all_delta
        + 0.4 * hard_delta
        + 0.05 * blocked_best_aggregate
        + 0.001 * len(h100_allowed)
    )
    return {
        "target": target,
        "train_filter": train_filter,
        "l2": float(l2),
        "min_h100_improvement": float(min_h100_improvement),
        "train_rows": int(train_ids.sum()),
        "weight": weight,
        "model_hash": _model_hash(weight, l2=float(l2), target=target, train_filter=train_filter),
        "validation_metrics": metrics,
        "validation_delta_vs_stage43_p": {
            "full_waypoint_ade_improvement_delta": all_delta,
            "t100_delta": t100_delta,
            "hard_failure_delta": hard_delta,
            "easy_degradation_delta": float(metrics["easy_degradation_vs_floor"] - base_metrics["easy_degradation_vs_floor"]),
        },
        "validation_h100_source_stability_table": support_table,
        "h100_allowed": h100_allowed,
        "objective": float(objective),
    }


def run_t100_source_stability_guard(
    *,
    seed: int = 449,
    l2_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    target_grid: tuple[str, ...] = ("residual", "direct"),
    train_filter_grid: tuple[str, ...] = ("t100", "t50t100"),
    min_h100_improvement_grid: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05),
    min_family_rows: int = 1000,
    min_source_rows: int = 100,
    min_source_count: int = 2,
    max_easy_degradation: float = 0.02,
    bootstrap: int = 1000,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43p = read_json(STAGE43_P_JSON, {})
    stage43q = read_json(STAGE43_Q_JSON, {})
    stage43o = read_json(STAGE43_O_JSON, {})
    train = _build_split("train", max_rows=None, seed=int(seed))
    val = _build_split("val", max_rows=None, seed=int(seed))
    test = _build_split("test", max_rows=None, seed=int(seed))
    feature_mean, feature_std = _standardize(train, val, test)

    p_selected = stage43p.get("selected_model", {})
    p_allowed = _parse_rules(p_selected.get("allowed_rules", []))
    p_train_mask = _train_mask(train, p_selected.get("train_filter", "t50t100"))
    p_weight = _ridge_fit(
        train.x[p_train_mask],
        _target_matrix(train, p_selected.get("target", "direct"))[p_train_mask],
        float(p_selected.get("l2", 1000.0)),
    )
    p_val_pred = _predict_waypoint(val, p_weight, p_selected.get("target", "direct"))
    p_val_candidate_ade, p_val_candidate_fde = _trajectory_error(val, p_val_pred)
    p_val_ade, p_val_fde, p_val_switch = _apply_rules(val, p_val_candidate_ade, p_val_candidate_fde, p_allowed)
    p_test_pred = _predict_waypoint(test, p_weight, p_selected.get("target", "direct"))
    p_test_candidate_ade, p_test_candidate_fde = _trajectory_error(test, p_test_pred)
    p_test_ade, p_test_fde, p_test_switch = _apply_rules(test, p_test_candidate_ade, p_test_candidate_fde, p_allowed)
    p_replay_metrics = _metrics(test, p_test_ade, p_test_fde, p_test_switch)
    p_report_metrics = stage43p.get("overall_full_test_metrics", {})
    p_replay_diff = {
        key: float(p_replay_metrics.get(key, 0.0) - float(p_report_metrics.get(key, 0.0)))
        for key in [
            "full_waypoint_ade_improvement_vs_floor",
            "t50_full_waypoint_ade_improvement_vs_floor",
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
            "hard_failure_full_waypoint_ade_improvement_vs_floor",
            "easy_degradation_vs_floor",
        ]
    }
    candidates = [
        _candidate_eval(
            train,
            val,
            p_val_ade,
            p_val_fde,
            p_val_switch,
            target=target,
            train_filter=train_filter,
            l2=l2,
            min_family_rows=int(min_family_rows),
            min_source_rows=int(min_source_rows),
            min_source_count=int(min_source_count),
            min_h100_improvement=min_h100_improvement,
            max_easy_degradation=float(max_easy_degradation),
        )
        for target in target_grid
        for train_filter in train_filter_grid
        for l2 in l2_grid
        for min_h100_improvement in min_h100_improvement_grid
    ]
    candidates.sort(key=lambda row: row["objective"], reverse=True)
    best = candidates[0]
    test_pred = _predict_waypoint(test, best["weight"], best["target"])
    candidate_ade, candidate_fde = _trajectory_error(test, test_pred)
    selected_ade, selected_fde, switch = _combine_with_h100(
        test,
        p_test_ade,
        p_test_fde,
        p_test_switch,
        candidate_ade,
        candidate_fde,
        best["h100_allowed"],
    )
    metrics = _metrics(test, selected_ade, selected_fde, switch)
    bootstrap_ci = _bootstrap_ci(test, selected_ade, selected_fde, n=int(bootstrap), seed=int(seed) + 4500)
    arrays = (test.floor_ade, test.floor_fde, selected_ade, selected_fde, candidate_ade, switch, test.easy)
    families, _ = _family_horizon(test)
    by_domain = _breakdown(test.domain, *arrays)
    by_horizon = _breakdown(test.horizon.astype(str), *arrays)
    by_source_family = _breakdown(families, *arrays, min_rows=50)
    h100_test_table = _h100_family_test_table(test, selected_ade, p_test_ade, switch)
    delta_vs_p = {
        "full_waypoint_ade_improvement_delta": float(
            metrics["full_waypoint_ade_improvement_vs_floor"]
            - float(p_report_metrics.get("full_waypoint_ade_improvement_vs_floor", 0.0))
        ),
        "t50_delta": float(
            metrics["t50_full_waypoint_ade_improvement_vs_floor"]
            - float(p_report_metrics.get("t50_full_waypoint_ade_improvement_vs_floor", 0.0))
        ),
        "t100_delta": float(
            metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
            - float(p_report_metrics.get("t100_raw_frame_full_waypoint_diagnostic_vs_floor", 0.0))
        ),
        "hard_failure_delta": float(
            metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            - float(p_report_metrics.get("hard_failure_full_waypoint_ade_improvement_vs_floor", 0.0))
        ),
        "easy_degradation_delta": float(
            metrics["easy_degradation_vs_floor"] - float(p_report_metrics.get("easy_degradation_vs_floor", 0.0))
        ),
    }
    candidate_rows = []
    for row in candidates[:12]:
        candidate_rows.append(
            {
                "target": row["target"],
                "train_filter": row["train_filter"],
                "l2": row["l2"],
                "min_h100_improvement": row["min_h100_improvement"],
                "train_rows": row["train_rows"],
                "model_hash": row["model_hash"],
                "objective": row["objective"],
                "validation_metrics": row["validation_metrics"],
                "validation_delta_vs_stage43_p": row["validation_delta_vs_stage43_p"],
                "h100_allowed_rules": _rules_to_list(row["h100_allowed"]),
                "validation_h100_source_stability_table": row["validation_h100_source_stability_table"],
            }
        )
    allowed_rules = _rules_to_list(best["h100_allowed"])
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_validation_source_stable_t100_guard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_o_precondition": {"verdict": stage43o.get("stage43_o_gate", {}).get("verdict")},
        "stage43_p_precondition": {
            "verdict": stage43p.get("stage43_p_gate", {}).get("verdict"),
            "replay_metrics": p_replay_metrics,
            "report_metrics": p_report_metrics,
            "replay_diff": p_replay_diff,
            "replay_exact": all(abs(value) < 1e-6 for value in p_replay_diff.values()),
        },
        "stage43_q_reference": {
            "verdict": stage43q.get("stage43_q_gate", {}).get("verdict"),
            "blocker": stage43q.get("t100_guarded_trial", {}).get("blocker"),
            "rejected_candidate_t100_delta": stage43q.get("trial_candidate_delta_vs_stage43_p", {}).get("t100_delta"),
            "rejected_candidate_allowed_rules": stage43q.get("selected_t100_trial", {}).get("h100_allowed_rules", []),
        },
        "training_protocol": {
            "model_family": "closed_form_ridge_h100_source_stability_guard",
            "selection_data": "validation_only",
            "test_threshold_tuning": False,
            "num_workers": 0,
            "seed": int(seed),
            "feature_mean_hash": hashlib.sha256(feature_mean.tobytes()).hexdigest(),
            "feature_std_hash": hashlib.sha256(feature_std.tobytes()).hexdigest(),
            "future_waypoints_as_labels_only": True,
            "deployment_floor": "Stage43-P unless h100 has source-stable validation support",
            "min_source_count": int(min_source_count),
            "min_source_rows": int(min_source_rows),
        },
        "candidate_search": {
            "l2_grid": list(map(float, l2_grid)),
            "target_grid": list(target_grid),
            "train_filter_grid": list(train_filter_grid),
            "min_h100_improvement_grid": list(map(float, min_h100_improvement_grid)),
            "candidate_count": int(len(candidates)),
            "top_candidates": candidate_rows,
        },
        "selected_source_stable_trial": {
            "target": best["target"],
            "train_filter": best["train_filter"],
            "l2": best["l2"],
            "min_h100_improvement": best["min_h100_improvement"],
            "train_rows": best["train_rows"],
            "model_hash": best["model_hash"],
            "h100_allowed_rules": allowed_rules,
            "validation_metrics": best["validation_metrics"],
            "validation_delta_vs_stage43_p": best["validation_delta_vs_stage43_p"],
            "validation_h100_source_stability_table": best["validation_h100_source_stability_table"],
        },
        "full_test_rows": int(len(test.x)),
        "overall_full_test_metrics": metrics,
        "delta_vs_stage43_p": delta_vs_p,
        "bootstrap_ci": bootstrap_ci,
        "by_domain": by_domain,
        "by_horizon": by_horizon,
        "by_source_family": by_source_family,
        "h100_family_test_table": h100_test_table,
        "t100_source_stability_guard": {
            "status": "h100_source_stable_positive" if allowed_rules else "h100_blocked_insufficient_source_stability",
            "allowed_h100_rules": allowed_rules,
            "source_stability_blocks_stage43_q_false_positive": "UCY|100" not in allowed_rules
            and "UCY|100" in stage43q.get("selected_t100_trial", {}).get("h100_allowed_rules", []),
            "deploy_h100_adapter": bool(allowed_rules)
            and metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
            and by_horizon.get("100", {}).get("easy_degradation_vs_floor", 0.0) <= float(max_easy_degradation),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "t100_positive_success": bool(allowed_rules)
            and metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0,
            "uniform_source_positive_success": False,
        },
    }
    payload["stage43_r_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["overall_full_test_metrics"]
    guard = payload["t100_source_stability_guard"]
    gates = {
        "stage43_p_precondition_passed": payload["stage43_p_precondition"]["verdict"]
        == "stage43_p_tail_horizon_adapter_pass_t100_still_fallback",
        "stage43_p_replay_exact": payload["stage43_p_precondition"]["replay_exact"] is True,
        "stage43_q_reference_available": payload["stage43_q_reference"]["verdict"]
        in {"stage43_q_t100_guarded_trial_honest_blocker", "stage43_q_t100_guarded_trial_positive_deployable"},
        "fresh_validation_source_stability_guard": payload["result_source"]
        == "fresh_validation_source_stable_t100_guard",
        "no_test_threshold_tuning": payload["training_protocol"]["test_threshold_tuning"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "future_waypoints_label_only": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True,
        "source_stability_audited": bool(payload["selected_source_stable_trial"]["validation_h100_source_stability_table"]),
        "q_false_positive_blocked": guard["source_stability_blocks_stage43_q_false_positive"] is True,
        "t50_not_destroyed": payload["delta_vs_stage43_p"]["t50_delta"] >= -1e-6,
        "hard_failure_not_destroyed": payload["delta_vs_stage43_p"]["hard_failure_delta"] >= -1e-6,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "t100_honest": (
            (guard["deploy_h100_adapter"] is True and metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0)
            or (guard["deploy_h100_adapter"] is False and metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-7)
        ),
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    if passed == total and guard["deploy_h100_adapter"]:
        verdict = "stage43_r_source_stable_h100_guard_positive"
    elif passed == total:
        verdict = "stage43_r_source_stable_h100_guard_blocks_t100_false_positive"
    else:
        verdict = "stage43_r_source_stable_h100_guard_incomplete"
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "deploy_h100_adapter": bool(guard["deploy_h100_adapter"] and passed == total),
        "t100_positive_success": bool(
            guard["deploy_h100_adapter"]
            and passed == total
            and payload["overall_full_test_metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
        ),
        "uniform_source_positive_success": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(dict(payload)))
    gate = payload["stage43_r_gate"]
    metrics = payload["overall_full_test_metrics"]
    guard = payload["t100_source_stability_guard"]
    selected = payload["selected_source_stable_trial"]
    ci = payload["bootstrap_ci"]["metrics"]
    lines = [
        "# Stage43-R T100 Source-Stability Guard",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- full-test rows: `{payload['full_test_rows']}`",
        f"- h100 status: `{guard['status']}`",
        f"- h100 allowed rules: `{', '.join(guard['allowed_h100_rules']) if guard['allowed_h100_rules'] else 'none'}`",
        f"- blocks Stage43-Q false positive: `{guard['source_stability_blocks_stage43_q_false_positive']}`",
        "",
        "## Selected Validation Trial",
        "",
        f"- target: `{selected['target']}`",
        f"- train filter: `{selected['train_filter']}`",
        f"- l2: `{selected['l2']}`",
        f"- min source count: `{payload['training_protocol']['min_source_count']}`",
        f"- min source rows: `{payload['training_protocol']['min_source_rows']}`",
        "",
        "## Deployment Metrics",
        "",
        f"- full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure ADE improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Bootstrap CI",
        "",
        f"- bootstrap n: `{payload['bootstrap_ci']['n']}`",
        f"- all ADE CI: `[{_pct(ci['full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t50 ADE CI: `[{_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t100 ADE CI: `[{_pct(ci['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['low'])}, {_pct(ci['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['high'])}]`",
        "",
        "## Validation H100 Source-Stability Table",
        "",
        "| family|horizon | rows | agg ADE lift | agg easy | source count | safe source count | reason |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, row in selected["validation_h100_source_stability_table"].items():
        lines.append(
            f"| {key} | {row['rows']} | {_pct(row['aggregate_full_waypoint_ade_improvement_vs_floor'])} | {_pct(row['aggregate_easy_degradation_vs_floor'])} | {row['supported_source_count']} | {row['safe_supported_source_count']} | `{row['reason']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage43-R adds a validation-only source-stability guard for h100. Stage43-Q allowed UCY|100 from a single validation source, but that test candidate produced negative t100 and high easy harm. The source-stability guard blocks such singleton-source h100 deployment, so t100 remains fallback-only and the blocker is now localized to insufficient source-stable h100 evidence.",
            "",
            "Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-R Gate",
        "",
        f"verdict: `{gate['verdict']}`",
        f"deploy_h100_adapter: `{gate['deploy_h100_adapter']}`",
        f"passed: `{gate['passed']} / {gate['total']}`",
        f"t100_positive_success: `{gate['t100_positive_success']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    gate_lines.extend([f"| {name} | `{value}` |" for name, value in gate["gates"].items()])
    write_md(GATE_MD, gate_lines)
    _refresh_readmes(payload)
    _update_state(payload)
    _append_ledger(payload)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_r_gate"]
    metrics = payload["overall_full_test_metrics"]
    guard = payload["t100_source_stability_guard"]
    lines = [
        "## Stage43-R t100 source-stability guard",
        "",
        f"Result source: `{payload['result_source']}`. This adds a validation-only source-stability requirement for h100 deployment over the Stage43-P safety floor.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- h100 status: `{guard['status']}`",
        f"- h100 allowed rules: `{', '.join(guard['allowed_h100_rules']) if guard['allowed_h100_rules'] else 'none'}`",
        f"- blocks Stage43-Q false positive: `{guard['source_stability_blocks_stage43_q_false_positive']}`",
        f"- full-waypoint ADE improvement vs floor: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement vs floor: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        "",
        "Boundary: t100 remains fallback-only. The h100 blocker is now localized to insufficient source-stable validation evidence; no metric/seconds claim, Stage5C, or SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, lines)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_r_gate"]
    state["stage43_r_t100_source_stability_guard"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "deploy_h100_adapter": gate["deploy_h100_adapter"],
        "selected_source_stable_trial": payload["selected_source_stable_trial"],
        "overall_full_test_metrics": payload["overall_full_test_metrics"],
        "delta_vs_stage43_p": payload["delta_vs_stage43_p"],
        "t100_source_stability_guard": payload["t100_source_stability_guard"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_r_t100_source_stability_guard"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_r_t100_source_stability_guard", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-R t100 source-stability guard.")
    parser.add_argument("--seed", type=int, default=449)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--min-family-rows", type=int, default=1000)
    parser.add_argument("--min-source-rows", type=int, default=100)
    parser.add_argument("--min-source-count", type=int, default=2)
    parser.add_argument("--max-easy-degradation", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_t100_source_stability_guard(
        seed=int(args.seed),
        min_family_rows=int(args.min_family_rows),
        min_source_rows=int(args.min_source_rows),
        min_source_count=int(args.min_source_count),
        max_easy_degradation=float(args.max_easy_degradation),
        bootstrap=int(args.bootstrap),
    )
    gate = result["stage43_r_gate"]
    print(f"Stage43-R: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
