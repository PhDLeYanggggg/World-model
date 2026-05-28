from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_eth_ucy_harm_aware_source_guard as jh
from src import stage42_eth_ucy_source_specific_easy_guard as jg
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_rotation_full_waypoint_eval as je
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "eth_ucy_source_robust_blocked_repair_stage42.json"
REPORT_MD = OUT_DIR / "eth_ucy_source_robust_blocked_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ji_gate.md"
JH_JSON = OUT_DIR / "eth_ucy_harm_aware_source_guard_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JI_ETH_UCY_SOURCE_ROBUST_BLOCKED_REPAIR"
SOURCE = "fresh_stage42_ji_eth_ucy_source_robust_blocked_repair"
EPS = 1e-6

SCORE_LAMBDAS = [0.1, 1.0, 10.0]
HARM_WEIGHTS = [0.5, 1.0, 2.0, 5.0]
CAPS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JI targets only the Stage42-JH blocked ETH_UCY sources with source-robust support checks.",
    "Candidate policies are selected on non-heldout train/validation sources only; held-out source is evaluated once.",
    "future waypoints / endpoints are labels/eval only, never inference inputs.",
    "No central velocity, no test endpoint goals, and no test-threshold tuning are used.",
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


def _fit_scalar_ridge(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, lam: float) -> np.ndarray:
    ids = np.where(train_mask)[0]
    xt = x[ids].astype(np.float64, copy=False)
    yt = y[ids].astype(np.float64, copy=False)
    reg = np.eye(xt.shape[1], dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xt.T @ xt + reg, xt.T @ yt).astype(np.float32)


def _candidate_score(metric: Mapping[str, Any], support: Mapping[str, Any]) -> float:
    if metric["easy_degradation"] > 0.02:
        return -1e9
    if support["max_easy_degradation"] > 0.02:
        return -1e9
    if metric["all_improvement"] <= 0 and metric["t50_improvement"] <= 0 and metric["hard_failure_improvement"] <= 0:
        return -1e9
    return (
        1.0 * float(metric["all_improvement"])
        + 1.6 * float(metric["t50_improvement"])
        + 1.2 * float(metric["hard_failure_improvement"])
        + 0.5 * float(support["mean_all_improvement"])
        + 0.8 * float(support["mean_t50_improvement"])
        - 0.25 * float(metric["switch_rate"])
        - 25.0 * max(0.0, float(support["max_easy_degradation"]) - 0.005)
        - 10.0 * max(0.0, -float(support["min_all_improvement"]))
    )


def _policy_errors(
    pred_xy: np.ndarray,
    floor_xy: np.ndarray,
    labels: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
    horizon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    floor_ade, floor_fde = am._trajectory_errors(floor_xy, labels)
    policy_ade = floor_ade.copy()
    policy_fde = floor_fde.copy()
    policy_switch = np.zeros(len(floor_ade), dtype=bool)
    for key, params in policy.get("slices", {}).items():
        h = int(str(key).replace("h", ""))
        local = horizon == h
        if not np.any(local):
            continue
        alpha = float(params["alpha"])
        blended = floor_xy + alpha * (pred_xy - floor_xy)
        b_ade, b_fde = am._trajectory_errors(blended, labels)
        policy_ade[local] = b_ade[local]
        policy_fde[local] = b_fde[local]
        policy_switch[local] = True
    return policy_ade, policy_fde, policy_switch


def _source_support_summary(
    selected: np.ndarray,
    floor: np.ndarray,
    switch: np.ndarray,
    data: Mapping[str, np.ndarray],
    source_ids: np.ndarray,
    split: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for source in sorted(set(source_ids[split != "test"].tolist())):
        mask = source_ids == source
        metric = am._metric(selected, floor, data, switch, mask)
        rows.append({"source": source, "split_role": sorted(set(split[mask].tolist())), "metric": metric})
    if not rows:
        return {
            "rows": [],
            "mean_all_improvement": 0.0,
            "mean_t50_improvement": 0.0,
            "mean_hard_failure_improvement": 0.0,
            "min_all_improvement": 0.0,
            "max_easy_degradation": 0.0,
            "all_sources_easy_safe": False,
            "all_sources_positive": False,
        }
    all_vals = [float(row["metric"]["all_improvement"]) for row in rows]
    t50_vals = [float(row["metric"]["t50_improvement"]) for row in rows]
    hard_vals = [float(row["metric"]["hard_failure_improvement"]) for row in rows]
    easy_vals = [float(row["metric"]["easy_degradation"]) for row in rows]
    return {
        "rows": rows,
        "mean_all_improvement": float(np.mean(all_vals)),
        "mean_t50_improvement": float(np.mean(t50_vals)),
        "mean_hard_failure_improvement": float(np.mean(hard_vals)),
        "min_all_improvement": float(np.min(all_vals)),
        "max_easy_degradation": float(np.max(easy_vals)),
        "all_sources_easy_safe": bool(np.max(easy_vals) <= 0.02),
        "all_sources_positive": bool(np.min(all_vals) > 0.0 or np.mean(t50_vals) > 0.03 or np.mean(hard_vals) > 0.10),
    }


def _test_deployable(metric: Mapping[str, Any]) -> bool:
    positive = metric["all_improvement"] > 0.03 and (
        metric["t50_improvement"] > 0.03 or metric["hard_failure_improvement"] > 0.10
    )
    easy_safe = metric["easy_degradation"] <= 0.02
    return bool(positive and easy_safe)


def _jh_blocked_sources() -> list[str]:
    report = read_json(JH_JSON, {})
    blocked = report.get("summary", {}).get("blocked_heldout_sources")
    if blocked:
        return list(blocked)
    payload = jh.run_stage42_eth_ucy_harm_aware_source_guard(refresh_readmes=False)
    return list(payload["summary"]["blocked_heldout_sources"])


def _jh_metrics_by_source() -> dict[str, Any]:
    report = read_json(JH_JSON, {})
    out = {}
    for row in report.get("folds", []):
        out[row["heldout_source"]] = row
    return out


def _evaluate_target_source(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], heldout_source: str) -> dict[str, Any]:
    split, split_stats = jg._source_cv_split(data, heldout_source)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    source_ids = jg._source_ids(data)
    floor = am._floor_arrays(data, train_mask)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    features, feature_names, removed = je._domain_invariant_features(data, floor)
    x, mean, std = am._standardize(features, train_mask)
    target_delta = (
        (
            labels["waypoint_xy"].astype(np.float64)
            - np.stack([data["current_x"], data["current_y"]], axis=1)[:, None, :]
        )
        / np.maximum(data["scale"].astype(np.float64)[:, None, None], EPS)
    ).astype(np.float32)
    horizon = data["horizon"].astype(int)

    best: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    for ridge_lam in am.LAMBDAS:
        coef = am._fit_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, ridge_lam)
        pred_xy = am._predict_waypoints(x, coef, data)
        base_policy, _base_ade_unused, _base_fde_unused, base_switch = je._select_horizon_policy_on_val(
            pred_xy, floor["floor_xy"], labels, data, val_mask
        )
        if not base_policy.get("slices"):
            continue
        policy_ade, policy_fde, policy_switch = _policy_errors(pred_xy, floor["floor_xy"], labels, base_policy, horizon)
        base_gain = floor_ade - policy_ade
        harm = np.maximum(policy_ade - floor_ade, 0.0)
        residual_norm = np.linalg.norm(pred_xy[:, -1] - floor["floor_xy"][:, -1], axis=1) / np.maximum(
            data["scale"].astype(np.float64), EPS
        )
        score_features = np.concatenate(
            [x, residual_norm[:, None].astype(np.float32), policy_switch[:, None].astype(np.float32)], axis=1
        )
        for score_lam in SCORE_LAMBDAS:
            gain_coef = _fit_scalar_ridge(score_features, base_gain, train_mask, score_lam)
            harm_coef = _fit_scalar_ridge(score_features, harm, train_mask, score_lam)
            pred_gain = (score_features.astype(np.float64) @ gain_coef.astype(np.float64)).astype(np.float64)
            pred_harm = np.maximum(
                (score_features.astype(np.float64) @ harm_coef.astype(np.float64)).astype(np.float64),
                0.0,
            )
            for harm_weight in HARM_WEIGHTS:
                score = pred_gain - float(harm_weight) * pred_harm
                thresholds = jh._thresholds_from_validation(score, val_mask & policy_switch)
                for threshold in thresholds:
                    threshold_switch = policy_switch & (score >= float(threshold))
                    for cap in CAPS:
                        guarded_switch = jh._cap_by_score(threshold_switch, score, horizon, cap)
                        selected_ade = np.where(guarded_switch, policy_ade, floor_ade)
                        selected_fde = np.where(guarded_switch, policy_fde, floor_fde)
                        val_metric = am._metric(selected_ade, floor_ade, data, guarded_switch, val_mask)
                        support = _source_support_summary(selected_ade, floor_ade, guarded_switch, data, source_ids, split)
                        candidate_score = _candidate_score(val_metric, support)
                        row = {
                            "ridge_lambda": float(ridge_lam),
                            "score_lambda": float(score_lam),
                            "harm_weight": float(harm_weight),
                            "predicted_net_gain_threshold": float(threshold),
                            "switch_cap": float(cap),
                            "score": float(candidate_score),
                            "policy_slice_count": int(len(base_policy["slices"])),
                            "val_metric": val_metric,
                            "support_summary": {
                                "mean_all_improvement": support["mean_all_improvement"],
                                "mean_t50_improvement": support["mean_t50_improvement"],
                                "mean_hard_failure_improvement": support["mean_hard_failure_improvement"],
                                "min_all_improvement": support["min_all_improvement"],
                                "max_easy_degradation": support["max_easy_degradation"],
                                "all_sources_easy_safe": support["all_sources_easy_safe"],
                                "all_sources_positive": support["all_sources_positive"],
                            },
                        }
                        candidates.append(row)
                        if best is None or candidate_score > best["score"]:
                            best = {
                                **row,
                                "selected_ade": selected_ade,
                                "selected_fde": selected_fde,
                                "switch": guarded_switch,
                                "floor_ade": floor_ade,
                                "floor_fde": floor_fde,
                                "support_rows": support["rows"],
                                "pred_score_stats": {
                                    "train_mean": float(np.mean(score[train_mask])),
                                    "val_mean": float(np.mean(score[val_mask])),
                                    "test_mean": float(np.mean(score[test_mask])),
                                    "test_switch_mean": float(np.mean(score[test_mask & guarded_switch])) if np.any(test_mask & guarded_switch) else 0.0,
                                },
                            }
    if best is None:
        selected_ade = floor_ade.copy()
        selected_fde = floor_fde.copy()
        switch = np.zeros(len(floor_ade), dtype=bool)
        metric = am._metric(selected_ade, floor_ade, data, switch, test_mask)
        return {
            "source": "fresh_run",
            "heldout_source": heldout_source,
            "split_stats": split_stats,
            "candidate_count": 0,
            "selected_candidate": {"fallback_only": True, "test_threshold_tuning": False},
            "metrics": {"source_robust_repair": metric},
            "bootstrap": {},
            "deployable_after_repair": False,
        }

    test_metric = am._metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask)
    test_fde = am._metric(best["selected_fde"], best["floor_fde"], data, best["switch"], test_mask)
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "source": "fresh_run",
        "heldout_source": heldout_source,
        "split_stats": split_stats,
        "feature_schema": {
            "feature_count": int(len(feature_names) + 2),
            "domain_features_removed": removed,
            "extra_features": ["residual_norm", "base_policy_switch_indicator"],
            "normalization": "train_split_mean_std_only",
            "future_inputs": False,
        },
        "candidate_count": int(len(candidates)),
        "top_validation_candidates": sorted(candidates, key=lambda r: r["score"], reverse=True)[:8],
        "selected_candidate": {
            "ridge_lambda": best["ridge_lambda"],
            "score_lambda": best["score_lambda"],
            "harm_weight": best["harm_weight"],
            "predicted_net_gain_threshold": best["predicted_net_gain_threshold"],
            "switch_cap": best["switch_cap"],
            "score": best["score"],
            "validation_selection_source": "non_heldout_source_robust_train_val_only",
            "test_threshold_tuning": False,
            "policy_slice_count": best["policy_slice_count"],
            "val_metric": best["val_metric"],
            "support_summary": best["support_summary"],
            "pred_score_stats": best["pred_score_stats"],
        },
        "support_source_metrics": best["support_rows"],
        "metrics": {
            "source_robust_repair": test_metric,
            "source_robust_repair_fde": test_fde,
        },
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask, seed=42191),
            "t50": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 50), seed=42192),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 100), seed=42193),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & hard_failure, seed=42194),
            "easy_degradation": am._bootstrap_ci(best["floor_ade"], best["selected_ade"], test_mask & easy, seed=42195),
        },
        "deployable_after_repair": _test_deployable(test_metric),
    }


def _summary(targets: list[Mapping[str, Any]], jh_by_source: Mapping[str, Any]) -> dict[str, Any]:
    repaired = []
    still_blocked = []
    improved_easy = []
    for row in targets:
        source = row["heldout_source"]
        metric = row["metrics"]["source_robust_repair"]
        prior = jh_by_source.get(source, {}).get("metrics", {}).get("harm_aware_source_guard", {})
        if row.get("deployable_after_repair"):
            repaired.append(source)
        else:
            still_blocked.append(source)
        if prior and metric["easy_degradation"] < prior.get("easy_degradation", metric["easy_degradation"]):
            improved_easy.append(source)
    decision = (
        "eth_ucy_blocked_sources_repaired"
        if repaired and not still_blocked
        else "eth_ucy_blocked_sources_partially_repaired"
        if repaired
        else "eth_ucy_blocked_sources_still_blocked"
    )
    return {
        "source": SOURCE,
        "targeted_blocked_sources": [row["heldout_source"] for row in targets],
        "repaired_sources": repaired,
        "still_blocked_sources": still_blocked,
        "easy_improved_sources": improved_easy,
        "decision": decision,
        "next_action": "Promote only repaired sources; keep still-blocked ETH_UCY sources fallback-only and investigate source-specific geometry/history support.",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    jh_by_source = _jh_metrics_by_source()
    blocked_sources = _jh_blocked_sources()
    data = jg._eth_ucy_data()
    labels = am._reconstruct_waypoint_labels(data)
    targets = [_evaluate_target_source(data, labels, source) for source in blocked_sources]
    payload: dict[str, Any] = {
        "stage": "Stage42-JI",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(JH_JSON),
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "target_selection": {
            "source": "cached_verified_stage42_jh_blocked_heldout_sources",
            "blocked_sources_from_jh": blocked_sources,
        },
        "targets": targets,
        "previous_jh_metrics": {
            source: jh_by_source.get(source, {}).get("metrics", {}).get("harm_aware_source_guard", {})
            for source in blocked_sources
        },
        "summary": _summary(targets, jh_by_source),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "source_overlap_pass": all(row["split_stats"]["source_overlap_pass"] for row in targets),
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
    payload["stage42_ji_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "blocked_sources_targeted": len(payload["targets"]) == len(payload["target_selection"]["blocked_sources_from_jh"]) and len(payload["targets"]) > 0,
        "candidate_search_recorded": all(row["candidate_count"] > 0 and row.get("top_validation_candidates") for row in payload["targets"]),
        "source_robust_support_recorded": all(row["selected_candidate"].get("support_summary") for row in payload["targets"]),
        "validation_only_selection": all(row["selected_candidate"].get("test_threshold_tuning") is False for row in payload["targets"]),
        "repaired_or_blocked_recorded": bool(payload["summary"]["repaired_sources"] or payload["summary"]["still_blocked_sources"]),
        "no_overclaim_all_eth_ucy": payload["summary"]["decision"] != "eth_ucy_blocked_sources_repaired"
        or not payload["summary"]["still_blocked_sources"],
        "no_leakage_pass": all(
            payload["no_leakage"][key] is False
            for key in [
                "future_endpoint_input",
                "future_waypoint_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
            ]
        )
        and payload["no_leakage"]["train_only_feature_normalization"]
        and payload["no_leakage"]["source_overlap_pass"],
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    verdict = "stage42_ji_eth_ucy_source_robust_blocked_repair_pass" if passed == len(gates) else "stage42_ji_eth_ucy_source_robust_blocked_repair_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _fmt(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ji_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-JI ETH_UCY Source-Robust Blocked-Source Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- decision: `{summary['decision']}`",
        f"- targeted_blocked_sources: `{summary['targeted_blocked_sources']}`",
        f"- repaired_sources: `{summary['repaired_sources']}`",
        f"- still_blocked_sources: `{summary['still_blocked_sources']}`",
        f"- easy_improved_sources: `{summary['easy_improved_sources']}`",
        f"- next_action: {summary['next_action']}",
        "",
        "## Target Source Test Metrics",
        "",
        "| heldout source | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | JH all | JH t50 | JH easy | deployable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["targets"]:
        metric = row["metrics"]["source_robust_repair"]
        prior = payload["previous_jh_metrics"].get(row["heldout_source"], {})
        lines.append(
            f"| `{row['heldout_source']}` | {metric['rows']} | {_fmt(metric['all_improvement'])} | {_fmt(metric['t50_improvement'])} | "
            f"{_fmt(metric['t100_raw_frame_diagnostic_improvement'])} | {_fmt(metric['hard_failure_improvement'])} | "
            f"{_fmt(metric['easy_degradation'])} | {_fmt(metric['switch_rate'])} | {_fmt(prior.get('all_improvement'))} | "
            f"{_fmt(prior.get('t50_improvement'))} | {_fmt(prior.get('easy_degradation'))} | `{row['deployable_after_repair']}` |"
        )
    lines.extend(["", "## Selected Candidate Support Summary", ""])
    for row in payload["targets"]:
        cand = row["selected_candidate"]
        lines.extend(
            [
                f"### `{row['heldout_source']}`",
                "",
                f"- candidate_count: `{row['candidate_count']}`",
                f"- ridge_lambda: `{cand.get('ridge_lambda')}`",
                f"- score_lambda: `{cand.get('score_lambda')}`",
                f"- harm_weight: `{cand.get('harm_weight')}`",
                f"- threshold: `{cand.get('predicted_net_gain_threshold')}`",
                f"- switch_cap: `{cand.get('switch_cap')}`",
                f"- support_summary: `{cand.get('support_summary')}`",
                "",
            ]
        )
    lines.extend(["## Bootstrap CI", "", "| heldout source | slice | low | mid | high | n |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in payload["targets"]:
        for key, ci in row.get("bootstrap", {}).items():
            lines.append(f"| `{row['heldout_source']}` | `{key}` | {_fmt(ci['low'])} | {_fmt(ci['mid'])} | {_fmt(ci['high'])} | {ci['n']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a stricter blocked-source repair attempt: non-heldout source support must be easy-safe before held-out evaluation.",
            "- If a blocked source remains blocked, deployment remains fallback-only for that source.",
            "- This remains dataset-local/raw-frame 2.5D evidence and does not enable metric/seconds claims, Stage5C, or SMC.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ji_gate"]
    lines = [
        "# Stage42-JI Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    gate = payload["stage42_ji_gate"]
    bits = []
    for row in payload["targets"]:
        metric = row["metrics"]["source_robust_repair"]
        bits.append(
            f"{row['heldout_source']}: all {_fmt(metric['all_improvement'])}, t50 {_fmt(metric['t50_improvement'])}, hard {_fmt(metric['hard_failure_improvement'])}, easy {_fmt(metric['easy_degradation'])}, deployable={row['deployable_after_repair']}"
        )
    return [
        "## Stage42-JI ETH_UCY Source-Robust Blocked-Source Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- targets from JH blocked sources: `{summary['targeted_blocked_sources']}`",
        f"- repair folds: {'; '.join(bits)}.",
        f"- decision: `{summary['decision']}`; repaired: `{summary['repaired_sources']}`; still blocked: `{summary['still_blocked_sources']}`; easy improved: `{summary['easy_improved_sources']}`.",
        "- boundary: held-out sources still blocked remain fallback-only; no global ETH_UCY/cross-domain overclaim, no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["eth_ucy_source_robust_blocked_repair"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_ji_gate"]["verdict"],
        "gate": {"passed": payload["stage42_ji_gate"]["passed"], "total": payload["stage42_ji_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "targeted_blocked_sources": payload["summary"]["targeted_blocked_sources"],
        "repaired_sources": payload["summary"]["repaired_sources"],
        "still_blocked_sources": payload["summary"]["still_blocked_sources"],
        "metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JI",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_ji_gate"]["verdict"],
                    "result": payload["summary"]["decision"],
                    "fresh_run": True,
                    "downloaded": False,
                    "converted": False,
                    "trained": True,
                    "evaluated": True,
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_eth_ucy_source_robust_blocked_repair(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    run_stage42_eth_ucy_source_robust_blocked_repair(refresh_readmes=True)


if __name__ == "__main__":
    main()
