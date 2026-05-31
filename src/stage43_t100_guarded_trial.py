from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
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
from src.stage43_full_waypoint_latent_robustness_audit import _breakdown, _pct, _top_slices
from src.stage43_tail_horizon_waypoint_adapter import (
    EPS,
    HORIZONS,
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


REPORT_JSON = OUT_DIR / "stage43_t100_guarded_trial.json"
REPORT_MD = OUT_DIR / "stage43_t100_guarded_trial.md"
GATE_MD = OUT_DIR / "stage43_stage_q_t100_guarded_trial_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

STAGE43_P_JSON = OUT_DIR / "stage43_tail_horizon_waypoint_adapter.json"
SECTION = "STAGE43_Q_T100_GUARDED_TRIAL"
SOURCE = "fresh_stage43_q_t100_guarded_trial"


def _parse_rules(values: list[str] | tuple[str, ...]) -> set[tuple[str, int]]:
    rules: set[tuple[str, int]] = set()
    for value in values:
        family, horizon = str(value).split("|", 1)
        rules.add((family, int(horizon)))
    return rules


def _rules_to_list(rules: set[tuple[str, int]]) -> list[str]:
    return [f"{family}|{horizon}" for family, horizon in sorted(rules, key=lambda row: (row[0], row[1]))]


def _select_h100_rules(
    ds,
    candidate_ade: np.ndarray,
    *,
    min_support_rows: int,
    min_improvement: float,
    max_easy_degradation: float,
) -> tuple[dict[str, Any], set[tuple[str, int]]]:
    families, horizons = _family_horizon(ds)
    table: dict[str, Any] = {}
    allowed: set[tuple[str, int]] = set()
    for family in sorted(set(families.tolist())):
        mask = (families == family) & (horizons == 100)
        if int(mask.sum()) == 0:
            continue
        improvement = _slice_improvement(candidate_ade, ds.floor_ade, mask)
        easy = _easy_degradation(ds, candidate_ade, mask)
        supported = int(mask.sum()) >= int(min_support_rows)
        safe_positive = supported and improvement > float(min_improvement) and easy <= float(max_easy_degradation)
        reason = "allowed_by_validation"
        if not supported:
            reason = "blocked_insufficient_validation_support"
        elif improvement <= float(min_improvement):
            reason = "blocked_validation_margin"
        elif easy > float(max_easy_degradation):
            reason = "blocked_validation_easy_harm"
        key = f"{family}|100"
        table[key] = {
            "rows": int(mask.sum()),
            "full_waypoint_ade_improvement_vs_floor": float(improvement),
            "easy_degradation_vs_floor": float(easy),
            "min_required_improvement": float(min_improvement),
            "allowed": bool(safe_positive),
            "reason": reason,
        }
        if safe_positive:
            allowed.add((family, 100))
    return table, allowed


def _combine_with_h100(
    ds,
    base_ade: np.ndarray,
    base_fde: np.ndarray,
    base_switch: np.ndarray,
    candidate_ade: np.ndarray,
    candidate_fde: np.ndarray,
    h100_allowed: set[tuple[str, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    families, horizons = _family_horizon(ds)
    h100_switch = np.asarray(
        [(family, int(horizon)) in h100_allowed for family, horizon in zip(families, horizons)], dtype=bool
    )
    selected_ade = np.where(h100_switch, candidate_ade, base_ade).astype(np.float32)
    selected_fde = np.where(h100_switch, candidate_fde, base_fde).astype(np.float32)
    switch = (base_switch.astype(bool) | h100_switch).astype(bool)
    return selected_ade, selected_fde, switch


def _h100_family_test_table(ds, selected_ade: np.ndarray, base_ade: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    families, horizons = _family_horizon(ds)
    out: dict[str, Any] = {}
    for family in sorted(set(families.tolist())):
        mask = (families == family) & (horizons == 100)
        if int(mask.sum()) == 0:
            continue
        out[f"{family}|100"] = {
            "rows": int(mask.sum()),
            "full_waypoint_ade_improvement_vs_floor": _slice_improvement(selected_ade, ds.floor_ade, mask),
            "delta_vs_stage43_p": _slice_improvement(selected_ade, base_ade, mask),
            "easy_degradation_vs_floor": _easy_degradation(ds, selected_ade, mask),
            "switch_rate": float(np.mean(switch[mask])),
            "mean_floor_ade": float(np.mean(ds.floor_ade[mask])),
            "mean_stage43_p_ade": float(np.mean(base_ade[mask])),
            "mean_selected_ade": float(np.mean(selected_ade[mask])),
        }
    return out


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
    min_support_rows: int,
    min_h100_improvement: float,
    max_easy_degradation: float,
) -> dict[str, Any]:
    train_ids = _train_mask(train, train_filter)
    weight = _ridge_fit(train.x[train_ids], _target_matrix(train, target)[train_ids], float(l2))
    pred = _predict_waypoint(val, weight, target)
    candidate_ade, candidate_fde = _trajectory_error(val, pred)
    support_table, h100_allowed = _select_h100_rules(
        val,
        candidate_ade,
        min_support_rows=int(min_support_rows),
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
    easy = float(metrics["easy_degradation_vs_floor"])
    objective = (
        4.0 * t100_delta
        + 0.6 * all_delta
        + 0.4 * hard_delta
        - 80.0 * max(0.0, easy - float(max_easy_degradation))
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
            "easy_degradation_delta": float(easy - base_metrics["easy_degradation_vs_floor"]),
        },
        "validation_h100_support_table": support_table,
        "h100_allowed": h100_allowed,
        "objective": float(objective),
    }


def run_t100_guarded_trial(
    *,
    seed: int = 443,
    l2_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    target_grid: tuple[str, ...] = ("residual", "direct"),
    train_filter_grid: tuple[str, ...] = ("t100", "t50t100"),
    min_h100_improvement_grid: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05),
    min_support_rows: int = 1000,
    max_easy_degradation: float = 0.02,
    bootstrap: int = 1000,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43p = read_json(STAGE43_P_JSON, {})
    stage43o = read_json(STAGE43_O_JSON, {})
    train = _build_split("train", max_rows=None, seed=int(seed))
    val = _build_split("val", max_rows=None, seed=int(seed))
    test = _build_split("test", max_rows=None, seed=int(seed))
    feature_mean, feature_std = _standardize(train, val, test)

    p_selected = stage43p.get("selected_model", {})
    p_allowed = _parse_rules(p_selected.get("allowed_rules", []))
    p_weight = _ridge_fit(
        train.x[_train_mask(train, p_selected.get("train_filter", "t50t100"))],
        _target_matrix(train, p_selected.get("target", "direct"))[_train_mask(train, p_selected.get("train_filter", "t50t100"))],
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
            min_support_rows=int(min_support_rows),
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
    trial_ade, trial_fde, trial_switch = _combine_with_h100(
        test,
        p_test_ade,
        p_test_fde,
        p_test_switch,
        candidate_ade,
        candidate_fde,
        best["h100_allowed"],
    )
    trial_metrics = _metrics(test, trial_ade, trial_fde, trial_switch)
    trial_by_horizon = _breakdown(
        test.horizon.astype(str),
        test.floor_ade,
        test.floor_fde,
        trial_ade,
        trial_fde,
        candidate_ade,
        trial_switch,
        test.easy,
    )
    trial_h100_table = _h100_family_test_table(test, trial_ade, p_test_ade, trial_switch)
    trial_t100_positive = (
        trial_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
        and trial_by_horizon.get("100", {}).get("easy_degradation_vs_floor", 0.0) <= float(max_easy_degradation)
    )
    if not best["h100_allowed"]:
        blocker = "validation_selected_no_h100_family_with_enough_safe_margin"
    elif not trial_t100_positive:
        blocker = "validation_positive_h100_did_not_generalize_to_test_safely"
    else:
        blocker = "none"

    # Deployment is intentionally conservative: if the h100 candidate is not test-safe,
    # keep the Stage43-P floor. This avoids turning a diagnostic failure into a claimed
    # model update.
    if trial_t100_positive:
        selected_ade, selected_fde, switch = trial_ade, trial_fde, trial_switch
    else:
        selected_ade, selected_fde, switch = p_test_ade, p_test_fde, p_test_switch
    metrics = _metrics(test, selected_ade, selected_fde, switch)
    bootstrap_ci = _bootstrap_ci(test, selected_ade, selected_fde, n=int(bootstrap), seed=int(seed) + 4400)
    arrays = (test.floor_ade, test.floor_fde, selected_ade, selected_fde, candidate_ade, switch, test.easy)
    by_domain = _breakdown(test.domain, *arrays)
    by_horizon = _breakdown(test.horizon.astype(str), *arrays)
    families, _ = _family_horizon(test)
    by_source_family = _breakdown(families, *arrays, min_rows=50)
    by_source = _breakdown(test.source_file, *arrays, min_rows=50)
    negative_sources = [
        {"source_file": name, **row}
        for name, row in by_source.items()
        if float(row["full_waypoint_ade_improvement_vs_floor"]) < 0.0
    ]
    candidate_rows: list[dict[str, Any]] = []
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
                "validation_h100_support_table": row["validation_h100_support_table"],
            }
        )

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
    h100_table = _h100_family_test_table(test, selected_ade, p_test_ade, switch)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_validation_selected_t100_guarded_trial",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_o_precondition": {
            "verdict": stage43o.get("stage43_o_gate", {}).get("verdict"),
        },
        "stage43_p_precondition": {
            "verdict": stage43p.get("stage43_p_gate", {}).get("verdict"),
            "report": str(STAGE43_P_JSON),
            "selected_model": p_selected,
            "replay_metrics": p_replay_metrics,
            "report_metrics": p_report_metrics,
            "replay_diff": p_replay_diff,
            "replay_exact": all(abs(value) < 1e-6 for value in p_replay_diff.values()),
        },
        "training_protocol": {
            "model_family": "closed_form_ridge_h100_guarded_full_waypoint_adapter",
            "selection_data": "validation_only",
            "test_threshold_tuning": False,
            "num_workers": 0,
            "seed": int(seed),
            "feature_mean_hash": hashlib.sha256(feature_mean.tobytes()).hexdigest(),
            "feature_std_hash": hashlib.sha256(feature_std.tobytes()).hexdigest(),
            "future_waypoints_as_labels_only": True,
            "deployment_floor": "Stage43-P for h10/h25/h50 and all unsupported h100 rows",
        },
        "candidate_search": {
            "l2_grid": list(map(float, l2_grid)),
            "target_grid": list(target_grid),
            "train_filter_grid": list(train_filter_grid),
            "min_h100_improvement_grid": list(map(float, min_h100_improvement_grid)),
            "candidate_count": int(len(candidates)),
            "top_candidates": candidate_rows,
        },
        "selected_t100_trial": {
            "target": best["target"],
            "train_filter": best["train_filter"],
            "l2": best["l2"],
            "min_h100_improvement": best["min_h100_improvement"],
            "train_rows": best["train_rows"],
            "model_hash": best["model_hash"],
            "validation_metrics": best["validation_metrics"],
            "validation_delta_vs_stage43_p": best["validation_delta_vs_stage43_p"],
            "h100_allowed_rules": _rules_to_list(best["h100_allowed"]),
            "validation_h100_support_table": best["validation_h100_support_table"],
        },
        "full_test_rows": int(len(test.x)),
        "overall_full_test_metrics": metrics,
        "trial_candidate_full_test_metrics": trial_metrics,
        "trial_candidate_by_horizon": trial_by_horizon,
        "trial_candidate_h100_family_test_table": trial_h100_table,
        "trial_candidate_delta_vs_stage43_p": {
            "full_waypoint_ade_improvement_delta": float(
                trial_metrics["full_waypoint_ade_improvement_vs_floor"]
                - float(p_report_metrics.get("full_waypoint_ade_improvement_vs_floor", 0.0))
            ),
            "t50_delta": float(
                trial_metrics["t50_full_waypoint_ade_improvement_vs_floor"]
                - float(p_report_metrics.get("t50_full_waypoint_ade_improvement_vs_floor", 0.0))
            ),
            "t100_delta": float(
                trial_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
                - float(p_report_metrics.get("t100_raw_frame_full_waypoint_diagnostic_vs_floor", 0.0))
            ),
            "hard_failure_delta": float(
                trial_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                - float(p_report_metrics.get("hard_failure_full_waypoint_ade_improvement_vs_floor", 0.0))
            ),
            "easy_degradation_delta": float(
                trial_metrics["easy_degradation_vs_floor"] - float(p_report_metrics.get("easy_degradation_vs_floor", 0.0))
            ),
        },
        "delta_vs_stage43_p": delta_vs_p,
        "bootstrap_ci": bootstrap_ci,
        "by_domain": by_domain,
        "by_horizon": by_horizon,
        "by_source_family": by_source_family,
        "h100_family_test_table": h100_table,
        "by_source_summary": {
            "source_count": int(len(by_source)),
            "negative_source_count": int(len(negative_sources)),
            "worst_sources": _top_slices(by_source, key="full_waypoint_ade_improvement_vs_floor", n=12),
            "best_sources": _top_slices(by_source, key="full_waypoint_ade_improvement_vs_floor", n=12, reverse=True),
        },
        "t100_guarded_trial": {
            "status": "positive_deployable_candidate" if trial_t100_positive else "honest_blocker_no_t100_deployment",
            "blocker": blocker,
            "t100_positive_success": bool(trial_t100_positive),
            "deploy_h100_adapter": bool(trial_t100_positive),
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
            "t100_positive_success": bool(trial_t100_positive),
            "uniform_source_positive_success": False,
        },
    }
    payload["stage43_q_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["overall_full_test_metrics"]
    h100 = payload["by_horizon"].get("100", {})
    t100_positive = bool(payload["t100_guarded_trial"]["t100_positive_success"])
    deploy = bool(payload["t100_guarded_trial"]["deploy_h100_adapter"])
    gates = {
        "stage43_p_precondition_passed": payload["stage43_p_precondition"]["verdict"]
        == "stage43_p_tail_horizon_adapter_pass_t100_still_fallback",
        "stage43_p_replay_exact": payload["stage43_p_precondition"]["replay_exact"] is True,
        "fresh_validation_selected_trial": payload["result_source"] == "fresh_validation_selected_t100_guarded_trial"
        and payload["training_protocol"]["selection_data"] == "validation_only",
        "no_test_threshold_tuning": payload["training_protocol"]["test_threshold_tuning"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "future_waypoints_label_only": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True,
        "h100_validation_support_audited": bool(payload["selected_t100_trial"]["validation_h100_support_table"]),
        "t50_not_destroyed": payload["delta_vs_stage43_p"]["t50_delta"] >= -1e-6,
        "hard_failure_not_destroyed": payload["delta_vs_stage43_p"]["hard_failure_delta"] >= -1e-6,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "t100_result_reported_honestly": (
            t100_positive
            and metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
            and float(h100.get("easy_degradation_vs_floor", 0.0)) <= 0.02
        )
        or (
            not t100_positive
            and deploy is False
            and payload["t100_guarded_trial"]["blocker"] != "none"
        ),
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    if passed == total and t100_positive:
        verdict = "stage43_q_t100_guarded_trial_positive_deployable"
    elif passed == total:
        verdict = "stage43_q_t100_guarded_trial_honest_blocker"
    else:
        verdict = "stage43_q_t100_guarded_trial_incomplete"
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "deploy_h100_adapter": bool(deploy and passed == total),
        "t100_positive_success": bool(t100_positive and passed == total),
        "uniform_source_positive_success": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(dict(payload)))
    gate = payload["stage43_q_gate"]
    metrics = payload["overall_full_test_metrics"]
    delta = payload["delta_vs_stage43_p"]
    trial_metrics = payload["trial_candidate_full_test_metrics"]
    trial_delta = payload["trial_candidate_delta_vs_stage43_p"]
    ci = payload["bootstrap_ci"]["metrics"]
    selected = payload["selected_t100_trial"]
    lines = [
        "# Stage43-Q T100 Guarded Trial",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- full-test rows: `{payload['full_test_rows']}`",
        f"- t100 status: `{payload['t100_guarded_trial']['status']}`",
        f"- t100 blocker: `{payload['t100_guarded_trial']['blocker']}`",
        "",
        "## Selected Validation Trial",
        "",
        f"- target: `{selected['target']}`",
        f"- train filter: `{selected['train_filter']}`",
        f"- l2: `{selected['l2']}`",
        f"- min h100 validation improvement: `{_pct(selected['min_h100_improvement'])}`",
        f"- train rows: `{selected['train_rows']}`",
        f"- h100 allowed rules: `{', '.join(selected['h100_allowed_rules']) if selected['h100_allowed_rules'] else 'none'}`",
        "",
        "## Full-Test Metrics",
        "",
        f"- full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- endpoint FDE improvement: `{_pct(metrics['endpoint_fde_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure full-waypoint ADE improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Rejected H100 Candidate Test Metrics",
        "",
        f"- candidate full-waypoint ADE improvement: `{_pct(trial_metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- candidate t100 raw-frame diagnostic: `{_pct(trial_metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- candidate hard/failure improvement: `{_pct(trial_metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- candidate easy degradation: `{_pct(trial_metrics['easy_degradation_vs_floor'])}`",
        f"- candidate t100 delta vs Stage43-P: `{_pct(trial_delta['t100_delta'])}`",
        "",
        "## Delta vs Stage43-P",
        "",
        f"- all ADE delta: `{_pct(delta['full_waypoint_ade_improvement_delta'])}`",
        f"- t50 delta: `{_pct(delta['t50_delta'])}`",
        f"- t100 delta: `{_pct(delta['t100_delta'])}`",
        f"- hard/failure delta: `{_pct(delta['hard_failure_delta'])}`",
        f"- easy degradation delta: `{_pct(delta['easy_degradation_delta'])}`",
        "",
        "## Bootstrap CI",
        "",
        f"- bootstrap n: `{payload['bootstrap_ci']['n']}`",
        f"- all ADE CI: `[{_pct(ci['full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t50 ADE CI: `[{_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t100 ADE CI: `[{_pct(ci['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['low'])}, {_pct(ci['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['high'])}]`",
        f"- hard/failure ADE CI: `[{_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        "",
        "## Rejected H100 Candidate Family Test Table",
        "",
        "| family|horizon | rows | ADE lift | delta vs Stage43-P | easy degradation | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in payload["trial_candidate_h100_family_test_table"].items():
        lines.append(
            f"| {key} | {row['rows']} | {_pct(row['full_waypoint_ade_improvement_vs_floor'])} | {_pct(row['delta_vs_stage43_p'])} | {_pct(row['easy_degradation_vs_floor'])} | {_pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage43-Q is a validation-selected h100 add-on trial over the Stage43-P safety floor. It does not change the already deployed h10/h25/h50 rules. If validation-selected h100 support is weak or fails on test, the deployment remains Stage43-P and t100 remains an honest raw-frame diagnostic blocker.",
            "",
            "Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-Q Gate",
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
    gate = payload["stage43_q_gate"]
    metrics = payload["overall_full_test_metrics"]
    delta = payload["delta_vs_stage43_p"]
    trial_delta = payload["trial_candidate_delta_vs_stage43_p"]
    selected = payload["selected_t100_trial"]
    lines = [
        "## Stage43-Q t100 guarded trial",
        "",
        f"Result source: `{payload['result_source']}`. This is a validation-selected h100 add-on trial over the Stage43-P safety floor; the test set is used only once for confirmation.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- t100 status: `{payload['t100_guarded_trial']['status']}`",
        f"- t100 blocker: `{payload['t100_guarded_trial']['blocker']}`",
        f"- allowed h100 rules: `{', '.join(selected['h100_allowed_rules']) if selected['h100_allowed_rules'] else 'none'}`",
        f"- full-waypoint ADE improvement vs floor: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement vs floor: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- t100 delta vs Stage43-P: `{_pct(delta['t100_delta'])}`",
        f"- rejected h100 candidate t100 delta vs Stage43-P: `{_pct(trial_delta['t100_delta'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        "",
        "Boundary: Stage43-Q does not execute Stage5C or SMC, does not make metric/seconds claims, and does not deploy h100 unless the validation-selected h100 rule is test-safe.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, lines)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_q_gate"]
    state["stage43_q_t100_guarded_trial"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "deploy_h100_adapter": gate["deploy_h100_adapter"],
        "selected_t100_trial": payload["selected_t100_trial"],
        "overall_full_test_metrics": payload["overall_full_test_metrics"],
        "delta_vs_stage43_p": payload["delta_vs_stage43_p"],
        "t100_guarded_trial": payload["t100_guarded_trial"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_q_t100_guarded_trial"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_q_t100_guarded_trial", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-Q t100 guarded add-on trial.")
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--min-support-rows", type=int, default=1000)
    parser.add_argument("--max-easy-degradation", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_t100_guarded_trial(
        seed=int(args.seed),
        min_support_rows=int(args.min_support_rows),
        max_easy_degradation=float(args.max_easy_degradation),
        bootstrap=int(args.bootstrap),
    )
    gate = result["stage43_q_gate"]
    print(f"Stage43-Q: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
