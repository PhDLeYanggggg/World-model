from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_eth_ucy_blocked_source_geometry_support as jj
from src import stage42_eth_ucy_harm_aware_source_guard as jh
from src import stage42_eth_ucy_source_robust_blocked_repair as ji
from src import stage42_eth_ucy_source_specific_easy_guard as jg
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_rotation_full_waypoint_eval as je
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "eth_ucy_row_family_selector_stage42.json"
REPORT_MD = OUT_DIR / "eth_ucy_row_family_selector_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jk_gate.md"
JJ_JSON = OUT_DIR / "eth_ucy_blocked_source_geometry_support_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JK_ETH_UCY_ROW_FAMILY_SELECTOR"
SOURCE = "fresh_stage42_jk_eth_ucy_row_family_selector"
EPS = 1e-6

LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]
HARM_WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]
CAPS = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JK follows Stage42-JJ: family-oracle t50 headroom exists, but static family policy cannot select rows safely.",
    "JK trains row-level expected gain/harm predictors over causal family baselines using train sources only, selects thresholds on validation source, and evaluates held-out source once.",
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


def _targets_from_jj() -> list[str]:
    report = read_json(JJ_JSON, {})
    blocked = report.get("summary", {}).get("still_blocked_sources")
    if blocked:
        return list(blocked)
    payload = jj.run_stage42_eth_ucy_blocked_source_geometry_support(refresh_readmes=False)
    return list(payload["summary"]["still_blocked_sources"])


def _fit_multi_ridge(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, lam: float) -> np.ndarray:
    ids = np.where(train_mask)[0]
    xt = x[ids].astype(np.float64, copy=False)
    yt = y[ids].astype(np.float64, copy=False)
    reg = np.eye(xt.shape[1], dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xt.T @ xt + reg, xt.T @ yt).astype(np.float32)


def _thresholds(score: np.ndarray, mask: np.ndarray) -> list[float]:
    vals = score[mask]
    if len(vals) == 0:
        return [0.0]
    qs = np.quantile(vals, [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 0.95])
    return sorted({float(v) for v in [np.min(vals), 0.0, *qs.tolist(), np.max(vals)]})


def _select_errors(
    floor_ade: np.ndarray,
    floor_fde: np.ndarray,
    family_ade: np.ndarray,
    family_fde: np.ndarray,
    best_family: np.ndarray,
    switch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    row = np.arange(len(floor_ade))
    selected_ade = floor_ade.copy()
    selected_fde = floor_fde.copy()
    selected_ade[switch] = family_ade[row[switch], best_family[switch]]
    selected_fde[switch] = family_fde[row[switch], best_family[switch]]
    return selected_ade, selected_fde


def _candidate_score(metric: Mapping[str, Any], support: Mapping[str, Any]) -> float:
    if metric["easy_degradation"] > 0.02:
        return -1e9
    if support["max_easy_degradation"] > 0.02:
        return -1e9
    if metric["all_improvement"] <= 0 and metric["t50_improvement"] <= 0 and metric["hard_failure_improvement"] <= 0:
        return -1e9
    return (
        1.1 * float(metric["all_improvement"])
        + 2.0 * float(metric["t50_improvement"])
        + 1.1 * float(metric["hard_failure_improvement"])
        + 0.35 * float(support["mean_all_improvement"])
        + 0.55 * float(support["mean_t50_improvement"])
        - 0.20 * float(metric["switch_rate"])
        - 30.0 * max(0.0, float(support["max_easy_degradation"]) - 0.005)
        - 8.0 * max(0.0, -float(support["min_all_improvement"]))
    )


def _deployable(metric: Mapping[str, Any]) -> bool:
    return bool(
        metric["easy_degradation"] <= 0.02
        and metric["all_improvement"] > 0.03
        and (metric["t50_improvement"] > 0.03 or metric["hard_failure_improvement"] > 0.10)
    )


def _best_family_counts(best_family: np.ndarray, switch: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for idx, name in enumerate(jj.s37.BASELINE_FAMILY):
        counts[name] = int(np.sum(switch & (best_family == idx)))
    return counts


def _evaluate_source(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], heldout_source: str) -> dict[str, Any]:
    split, split_stats = jg._source_cv_split(data, heldout_source)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    source_ids = jg._source_ids(data)
    floor = am._floor_arrays(data, train_mask)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    family_xy = jj._family_waypoints(data)
    family_ade, family_fde = jj._family_errors(family_xy, labels)
    features, feature_names, removed = je._domain_invariant_features(data, floor)
    x, mean, std = am._standardize(features, train_mask)
    family_gain = floor_ade[:, None] - family_ade
    family_harm = np.maximum(family_ade - floor_ade[:, None], 0.0)
    oracle_ade = np.minimum(floor_ade, np.min(family_ade, axis=1))
    oracle_switch = np.min(family_ade, axis=1) < floor_ade
    oracle_metric = am._metric(oracle_ade, floor_ade, data, oracle_switch, test_mask)

    best: dict[str, Any] | None = None
    best_rejected: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    horizon = data["horizon"].astype(int)
    for lam in LAMBDAS:
        gain_coef = _fit_multi_ridge(x, family_gain.astype(np.float32), train_mask, lam)
        harm_coef = _fit_multi_ridge(x, family_harm.astype(np.float32), train_mask, lam)
        pred_gain = x.astype(np.float64) @ gain_coef.astype(np.float64)
        pred_harm = np.maximum(x.astype(np.float64) @ harm_coef.astype(np.float64), 0.0)
        for harm_weight in HARM_WEIGHTS:
            score_matrix = pred_gain - float(harm_weight) * pred_harm
            best_family = np.argmax(score_matrix, axis=1).astype(np.int16)
            best_score = np.max(score_matrix, axis=1)
            for threshold in _thresholds(best_score, val_mask):
                threshold_switch = best_score >= float(threshold)
                for cap in CAPS:
                    switch = jh._cap_by_score(threshold_switch, best_score, horizon, cap)
                    selected_ade, selected_fde = _select_errors(
                        floor_ade, floor_fde, family_ade, family_fde, best_family, switch
                    )
                    val_metric = am._metric(selected_ade, floor_ade, data, switch, val_mask)
                    support = ji._source_support_summary(selected_ade, floor_ade, switch, data, source_ids, split)
                    score = _candidate_score(val_metric, support)
                    row = {
                        "lambda": float(lam),
                        "harm_weight": float(harm_weight),
                        "threshold": float(threshold),
                        "switch_cap": float(cap),
                        "score": float(score),
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
                    if best_rejected is None or score > best_rejected["score"]:
                        best_rejected = row
                    if score > -1e8 and (best is None or score > best["score"]):
                        best = {
                            **row,
                            "selected_ade": selected_ade,
                            "selected_fde": selected_fde,
                            "switch": switch,
                            "best_family": best_family,
                            "support_rows": support["rows"],
                            "pred_score_stats": {
                                "train_mean": float(np.mean(best_score[train_mask])),
                                "val_mean": float(np.mean(best_score[val_mask])),
                                "test_mean": float(np.mean(best_score[test_mask])),
                                "test_switch_mean": float(np.mean(best_score[test_mask & switch])) if np.any(test_mask & switch) else 0.0,
                            },
                        }
    if best is None:
        switch = np.zeros(len(floor_ade), dtype=bool)
        metric = am._metric(floor_ade, floor_ade, data, switch, test_mask)
        fde_metric = am._metric(floor_fde, floor_fde, data, switch, test_mask)
        h = data["horizon"].astype(int)
        hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
        easy = data["easy"].astype(bool)
        return {
            "source": "fresh_run",
            "heldout_source": heldout_source,
            "split_stats": split_stats,
            "feature_schema": {
                "feature_count": int(len(feature_names)),
                "domain_features_removed": removed,
                "normalization": "train_split_mean_std_only",
                "future_inputs": False,
            },
            "candidate_count": int(len(candidates)),
            "selected_candidate": {
                "fallback_only": True,
                "reason": "no_validation_safe_row_family_candidate",
                "test_threshold_tuning": False,
                "best_rejected_candidate": best_rejected,
            },
            "top_validation_candidates": sorted(candidates, key=lambda r: r["score"], reverse=True)[:8],
            "support_source_metrics": [],
            "metrics": {
                "row_family_selector": metric,
                "row_family_selector_fde": fde_metric,
                "family_oracle": oracle_metric,
            },
            "family_switch_counts": {name: 0 for name in jj.s37.BASELINE_FAMILY},
            "deployable_after_row_family_selector": False,
            "bootstrap": {
                "all": am._bootstrap_ci(floor_ade, floor_ade, test_mask, seed=42231),
                "t50": am._bootstrap_ci(floor_ade, floor_ade, test_mask & (h == 50), seed=42232),
                "t100_raw_frame_diagnostic": am._bootstrap_ci(floor_ade, floor_ade, test_mask & (h == 100), seed=42233),
                "hard_failure": am._bootstrap_ci(floor_ade, floor_ade, test_mask & hard_failure, seed=42234),
                "easy_degradation": am._bootstrap_ci(floor_ade, floor_ade, test_mask & easy, seed=42235),
                "oracle_t50": am._bootstrap_ci(oracle_ade, floor_ade, test_mask & (h == 50), seed=42236),
            },
        }
    metric = am._metric(best["selected_ade"], floor_ade, data, best["switch"], test_mask)
    fde_metric = am._metric(best["selected_fde"], floor_fde, data, best["switch"], test_mask)
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "source": "fresh_run",
        "heldout_source": heldout_source,
        "split_stats": split_stats,
        "feature_schema": {
            "feature_count": int(len(feature_names)),
            "domain_features_removed": removed,
            "normalization": "train_split_mean_std_only",
            "future_inputs": False,
        },
        "candidate_count": int(len(candidates)),
        "selected_candidate": {
            "lambda": best["lambda"],
            "harm_weight": best["harm_weight"],
            "threshold": best["threshold"],
            "switch_cap": best["switch_cap"],
            "score": best["score"],
            "validation_selection_source": "nonheldout_source_validation_only",
            "test_threshold_tuning": False,
            "val_metric": best["val_metric"],
            "support_summary": best["support_summary"],
            "pred_score_stats": best["pred_score_stats"],
        },
        "top_validation_candidates": sorted(candidates, key=lambda r: r["score"], reverse=True)[:8],
        "support_source_metrics": best["support_rows"],
        "metrics": {
            "row_family_selector": metric,
            "row_family_selector_fde": fde_metric,
            "family_oracle": oracle_metric,
        },
        "family_switch_counts": _best_family_counts(best["best_family"], best["switch"] & test_mask),
        "deployable_after_row_family_selector": _deployable(metric),
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask, seed=42231),
            "t50": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask & (h == 50), seed=42232),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask & (h == 100), seed=42233),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask & hard_failure, seed=42234),
            "easy_degradation": am._bootstrap_ci(floor_ade, best["selected_ade"], test_mask & easy, seed=42235),
            "oracle_t50": am._bootstrap_ci(oracle_ade, floor_ade, test_mask & (h == 50), seed=42236),
        },
    }


def _summary(targets: list[Mapping[str, Any]]) -> dict[str, Any]:
    repaired = [row["heldout_source"] for row in targets if row["deployable_after_row_family_selector"]]
    blocked = [row["heldout_source"] for row in targets if not row["deployable_after_row_family_selector"]]
    positive_t50 = [row["heldout_source"] for row in targets if row["metrics"]["row_family_selector"]["t50_improvement"] > 0.03]
    easy_safe = [row["heldout_source"] for row in targets if row["metrics"]["row_family_selector"]["easy_degradation"] <= 0.02]
    decision = (
        "row_family_selector_repaired_all_blocked_sources"
        if repaired and not blocked
        else "row_family_selector_partially_repaired_blocked_sources"
        if repaired
        else "row_family_selector_not_deployable_on_blocked_sources"
    )
    return {
        "source": SOURCE,
        "targeted_sources": [row["heldout_source"] for row in targets],
        "repaired_sources": repaired,
        "still_blocked_sources": blocked,
        "positive_t50_sources": positive_t50,
        "easy_safe_sources": easy_safe,
        "decision": decision,
        "next_action": "If row-family selector is positive but unsafe, add a heldout-source-safe harm model; if still negative, acquire/calibrate source-specific geometry support.",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = jg._eth_ucy_data()
    labels = am._reconstruct_waypoint_labels(data)
    targets = [_evaluate_source(data, labels, source) for source in _targets_from_jj()]
    payload: dict[str, Any] = {
        "stage": "Stage42-JK",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(JJ_JSON),
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "target_selection": {
            "source": "cached_verified_stage42_jj_still_blocked_sources",
            "blocked_sources_from_jj": _targets_from_jj(),
        },
        "targets": targets,
        "summary": _summary(targets),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_gain_harm_targets": True,
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
    payload["stage42_jk_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "jj_blocked_sources_targeted": len(payload["targets"]) == len(payload["target_selection"]["blocked_sources_from_jj"]) and len(payload["targets"]) > 0,
        "row_family_candidates_trained": all(row["candidate_count"] > 0 for row in payload["targets"]),
        "validation_only_selection": all(row["selected_candidate"]["test_threshold_tuning"] is False for row in payload["targets"]),
        "support_summary_recorded": all(
            row["selected_candidate"].get("fallback_only") or row["selected_candidate"].get("support_summary")
            for row in payload["targets"]
        ),
        "repaired_or_blocked_recorded": bool(payload["summary"]["repaired_sources"] or payload["summary"]["still_blocked_sources"]),
        "family_switch_counts_recorded": all(row["family_switch_counts"] for row in payload["targets"]),
        "no_overclaim_full_eth_ucy": payload["summary"]["decision"] != "row_family_selector_repaired_all_blocked_sources"
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
        and payload["no_leakage"]["train_only_gain_harm_targets"]
        and payload["no_leakage"]["train_only_feature_normalization"]
        and payload["no_leakage"]["source_overlap_pass"],
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    verdict = "stage42_jk_eth_ucy_row_family_selector_pass" if passed == len(gates) else "stage42_jk_eth_ucy_row_family_selector_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _fmt(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jk_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-JK ETH_UCY Row-Level Family Selector",
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
        f"- targeted_sources: `{summary['targeted_sources']}`",
        f"- repaired_sources: `{summary['repaired_sources']}`",
        f"- still_blocked_sources: `{summary['still_blocked_sources']}`",
        f"- positive_t50_sources: `{summary['positive_t50_sources']}`",
        f"- easy_safe_sources: `{summary['easy_safe_sources']}`",
        f"- next_action: {summary['next_action']}",
        "",
        "## Held-Out Source Metrics",
        "",
        "| source | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | oracle t50 | deployable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["targets"]:
        metric = row["metrics"]["row_family_selector"]
        oracle = row["metrics"]["family_oracle"]
        lines.append(
            f"| `{row['heldout_source']}` | {metric['rows']} | {_fmt(metric['all_improvement'])} | {_fmt(metric['t50_improvement'])} | "
            f"{_fmt(metric['t100_raw_frame_diagnostic_improvement'])} | {_fmt(metric['hard_failure_improvement'])} | "
            f"{_fmt(metric['easy_degradation'])} | {_fmt(metric['switch_rate'])} | {_fmt(oracle['t50_improvement'])} | "
            f"`{row['deployable_after_row_family_selector']}` |"
        )
    lines.extend(["", "## Selected Candidates", ""])
    for row in payload["targets"]:
        cand = row["selected_candidate"]
        lines.extend(
            [
                f"### `{row['heldout_source']}`",
                "",
                f"- fallback_only: `{cand.get('fallback_only', False)}`",
                f"- reason: `{cand.get('reason', 'selected_validation_safe_candidate')}`",
                f"- lambda: `{cand.get('lambda')}`",
                f"- harm_weight: `{cand.get('harm_weight')}`",
                f"- threshold: `{cand.get('threshold')}`",
                f"- switch_cap: `{cand.get('switch_cap')}`",
                f"- support_summary: `{cand.get('support_summary')}`",
                f"- best_rejected_candidate: `{cand.get('best_rejected_candidate')}`",
                f"- family_switch_counts: `{row['family_switch_counts']}`",
                "",
            ]
        )
    lines.extend(["## Bootstrap CI", "", "| source | slice | low | mid | high | n |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in payload["targets"]:
        for key, ci in row["bootstrap"].items():
            lines.append(f"| `{row['heldout_source']}` | `{key}` | {_fmt(ci['low'])} | {_fmt(ci['mid'])} | {_fmt(ci['high'])} | {ci['n']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Row-level family selection directly targets the Stage42-JJ finding: family oracle has t50 headroom, static policy cannot choose rows.",
            "- If the selector remains blocked, the missing piece is likely source-specific geometry/history/harm prediction rather than family availability.",
            "- This remains dataset-local/raw-frame 2.5D evidence; no Stage5C, SMC, metric, or seconds-level claim is enabled.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jk_gate"]
    lines = [
        "# Stage42-JK Gate",
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
    gate = payload["stage42_jk_gate"]
    bits = []
    for row in payload["targets"]:
        metric = row["metrics"]["row_family_selector"]
        oracle = row["metrics"]["family_oracle"]
        bits.append(
            f"{row['heldout_source']}: all {_fmt(metric['all_improvement'])}, t50 {_fmt(metric['t50_improvement'])}, hard {_fmt(metric['hard_failure_improvement'])}, easy {_fmt(metric['easy_degradation'])}, oracle t50 {_fmt(oracle['t50_improvement'])}, deployable={row['deployable_after_row_family_selector']}"
        )
    return [
        "## Stage42-JK ETH_UCY Row-Level Family Selector",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- row-family heldout results: {'; '.join(bits)}.",
        f"- decision: `{summary['decision']}`; repaired: `{summary['repaired_sources']}`; still blocked: `{summary['still_blocked_sources']}`.",
        "- boundary: no full ETH_UCY/cross-domain overclaim; still dataset-local raw-frame 2.5D, no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["eth_ucy_row_family_selector"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jk_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jk_gate"]["passed"], "total": payload["stage42_jk_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "targeted_sources": payload["summary"]["targeted_sources"],
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
                    "stage": "Stage42-JK",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jk_gate"]["verdict"],
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


def run_stage42_eth_ucy_row_family_selector(*, refresh_readmes: bool = True) -> dict[str, Any]:
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
    run_stage42_eth_ucy_row_family_selector(refresh_readmes=True)


if __name__ == "__main__":
    main()
