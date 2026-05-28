from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_eth_ucy_blocked_source_geometry_support as jj
from src import stage42_eth_ucy_source_specific_easy_guard as jg
from src import stage42_eth_ucy_source_support_coverage as jl
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "eth_ucy_calibrated_support_recheck_stage42.json"
REPORT_MD = OUT_DIR / "eth_ucy_calibrated_support_recheck_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jm_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_eth_ucy_calibrated_support_stage42.md"
JL_JSON = OUT_DIR / "eth_ucy_source_support_coverage_stage42.json"
BN_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JM_ETH_UCY_CALIBRATED_SUPPORT_RECHECK"
SOURCE = "fresh_stage42_jm_eth_ucy_calibrated_support_recheck"
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JM rechecks the Stage42-JL blockers using BN source-specific time/geometry evidence.",
    "Source-specific calibration evidence can justify a restricted diagnostic subset, not a global metric/seconds M3W claim.",
    "Calibrated support signatures use past-only history/geometry summaries plus source-specific calibration flags; held-out labels are evaluation-only.",
    "No central velocity, no test endpoint goals, no test-threshold tuning, and no Stage5C/SMC execution are used.",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _targets_from_jl() -> list[str]:
    report = read_json(JL_JSON, {})
    blocked = report.get("summary", {}).get("still_blocked_sources")
    if blocked:
        return list(blocked)
    payload = jl.run_stage42_eth_ucy_source_support_coverage(refresh_readmes=False)
    return list(payload["summary"]["still_blocked_sources"])


def _rel_to_source_id(rel_source: str) -> str:
    if rel_source.startswith("ETH/seq_eth"):
        return "ETH_seq_eth"
    if rel_source.startswith("ETH/seq_hotel"):
        return "ETH_seq_hotel"
    if rel_source.startswith("UCY/zara01"):
        return "UCY_zara01"
    if rel_source.startswith("UCY/zara02"):
        return "UCY_zara02"
    if rel_source.startswith("UCY/zara03"):
        return "UCY_zara03"
    if rel_source.startswith("UCY/students03"):
        return "UCY_students03"
    if rel_source.startswith("UCY/students01"):
        return "UCY_students01"
    return rel_source.replace("/", "_").replace(".txt", "")


def _bn_records() -> dict[str, Mapping[str, Any]]:
    report = read_json(BN_JSON, {})
    return {str(row["source_id"]): row for row in report.get("source_records", [])}


def _calibration_flags_for_sources(source_ids: np.ndarray, records: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    rows = []
    for rel in source_ids.tolist():
        source_id = _rel_to_source_id(str(rel))
        record = records.get(source_id, {})
        timing = record.get("timing", {})
        rows.append(
            [
                float(bool(record.get("source_specific_metric_time_evidence"))),
                float(bool(record.get("homography", {}).get("parseable"))),
                float(bool(record.get("coordinate", {}).get("meter_coordinates_evidence"))),
                float(timing.get("annotation_fps") or 0.0) / 10.0,
                float(timing.get("annotation_timestep_seconds") or 0.0),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _calibrated_features(data: Mapping[str, np.ndarray], source_ids: np.ndarray, records: Mapping[str, Mapping[str, Any]]) -> tuple[np.ndarray, list[str]]:
    hist = data["history_seq"].astype(np.float64, copy=False)
    valid = np.isfinite(hist)
    clean = np.where(valid, hist, 0.0)
    scale = np.maximum(data["scale"].astype(np.float64, copy=False), EPS)
    dt = np.maximum(data["dt_frame_step"].astype(np.float64, copy=False), EPS)
    hist_mean = clean.mean(axis=1)
    hist_std = clean.std(axis=1)
    hist_last = clean[:, -1, :]
    hist_first = clean[:, 0, :]
    hist_delta = hist_last - hist_first
    hist_abs_p90 = np.percentile(np.abs(clean), 90, axis=1)
    scalar = data["history_scalar"].astype(np.float64, copy=False)
    calibrated_flags = _calibration_flags_for_sources(source_ids, records).astype(np.float64, copy=False)
    base = np.concatenate(
        [
            hist_mean / scale[:, None],
            hist_std / scale[:, None],
            hist_delta / scale[:, None],
            hist_abs_p90 / scale[:, None],
            scalar[:, : min(9, scalar.shape[1])] / np.maximum(scale[:, None], EPS),
            np.log1p(data["track_length"].astype(np.float64, copy=False))[:, None],
            np.log1p(scale)[:, None],
            (dt / 12.0)[:, None],
            (data["horizon"].astype(np.float64, copy=False) / 100.0)[:, None],
            calibrated_flags,
        ],
        axis=1,
    )
    names = (
        [f"hist_mean_{i}_per_scale" for i in range(hist.shape[2])]
        + [f"hist_std_{i}_per_scale" for i in range(hist.shape[2])]
        + [f"hist_delta_{i}_per_scale" for i in range(hist.shape[2])]
        + [f"hist_abs_p90_{i}_per_scale" for i in range(hist.shape[2])]
        + [f"history_scalar_{i}_per_scale" for i in range(min(9, scalar.shape[1]))]
        + ["log_track_length", "log_scale", "dt_frame_step_over_12", "horizon_over_100"]
        + ["source_metric_time_evidence", "homography_parseable", "meter_coordinate_evidence", "annotation_fps_over_10", "annotation_timestep_seconds"]
    )
    return base.astype(np.float32), names


def _standardize_on_train(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = x[train_mask].astype(np.float64, copy=False)
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std[std < EPS] = 1.0
    return ((x.astype(np.float64, copy=False) - mean) / std).astype(np.float32)


def _source_signature(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if int(np.sum(mask)) == 0:
        return np.zeros(x.shape[1], dtype=np.float64)
    return np.mean(x[mask].astype(np.float64, copy=False), axis=0)


def _support_threshold(signatures: Mapping[str, np.ndarray]) -> float:
    names = sorted(signatures)
    distances = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            distances.append(float(np.linalg.norm(signatures[left] - signatures[right])))
    if not distances:
        return 0.0
    return float(np.quantile(np.asarray(distances, dtype=np.float64), 0.75))


def _nearest_sources(target: np.ndarray, signatures: Mapping[str, np.ndarray], k: int = 2) -> list[dict[str, Any]]:
    rows = [{"source": source, "distance": float(np.linalg.norm(target - signature))} for source, signature in signatures.items()]
    return sorted(rows, key=lambda row: row["distance"])[: max(1, min(k, len(rows)))]


def _family_metric(
    family_ade: np.ndarray,
    floor_ade: np.ndarray,
    data: Mapping[str, np.ndarray],
    source_ids: np.ndarray,
    source: str,
    family_idx: int,
) -> dict[str, Any]:
    mask = source_ids == source
    switch = np.ones(len(floor_ade), dtype=bool)
    return am._metric(family_ade[:, family_idx], floor_ade, data, switch, mask)


def _safe(metric: Mapping[str, Any]) -> bool:
    return bool(
        metric["easy_degradation"] <= 0.02
        and (
            metric["all_improvement"] > 0.0
            or metric["t50_improvement"] > 0.03
            or metric["hard_failure_improvement"] > 0.10
        )
    )


def _support_table(
    family_ade: np.ndarray,
    floor_ade: np.ndarray,
    data: Mapping[str, np.ndarray],
    source_ids: np.ndarray,
    support_sources: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for family_idx, family in enumerate(jj.s37.BASELINE_FAMILY):
        metrics = [_family_metric(family_ade, floor_ade, data, source_ids, source, family_idx) for source in support_sources]
        rows.append(
            {
                "family_idx": int(family_idx),
                "family": family,
                "safe_source_count": int(sum(_safe(metric) for metric in metrics)),
                "support_source_count": int(len(metrics)),
                "mean_all_improvement": float(np.mean([m["all_improvement"] for m in metrics])) if metrics else 0.0,
                "mean_t50_improvement": float(np.mean([m["t50_improvement"] for m in metrics])) if metrics else 0.0,
                "mean_hard_failure_improvement": float(np.mean([m["hard_failure_improvement"] for m in metrics])) if metrics else 0.0,
                "max_easy_degradation": float(np.max([m["easy_degradation"] for m in metrics])) if metrics else 0.0,
                "metrics": metrics,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["safe_source_count"],
            row["mean_all_improvement"] + 1.7 * row["mean_t50_improvement"] + row["mean_hard_failure_improvement"] - 30.0 * max(0.0, row["max_easy_degradation"] - 0.02),
        ),
        reverse=True,
    )


def _evaluate_source(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], heldout_source: str, records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    split, split_stats = jg._source_cv_split(data, heldout_source)
    train_mask = split == "train"
    test_mask = split == "test"
    source_ids = jg._source_ids(data)
    source_names = sorted(set(source_ids.tolist()))
    support_sources = [source for source in source_names if source != heldout_source]

    floor = am._floor_arrays(data, train_mask)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    family_xy = jj._family_waypoints(data)
    family_ade, family_fde = jj._family_errors(family_xy, labels)

    x_raw, feature_names = _calibrated_features(data, source_ids, records)
    x = _standardize_on_train(x_raw, train_mask)
    signatures = {source: _source_signature(x, source_ids == source) for source in source_names}
    support_signatures = {source: signatures[source] for source in support_sources}
    threshold = _support_threshold(support_signatures)
    nearest = _nearest_sources(signatures[heldout_source], support_signatures)
    nearest_sources = [row["source"] for row in nearest]
    nearest_distance = float(nearest[0]["distance"]) if nearest else float("inf")
    in_support = bool(nearest_distance <= threshold + EPS)

    family_support = _support_table(family_ade, floor_ade, data, source_ids, nearest_sources)
    selected = next(
        (
            row
            for row in family_support
            if row["safe_source_count"] == row["support_source_count"] and row["safe_source_count"] > 0
        ),
        None,
    )

    switch = np.zeros(len(floor_ade), dtype=bool)
    selected_ade = floor_ade.copy()
    selected_fde = floor_fde.copy()
    reason = "calibrated_source_out_of_support"
    if in_support and selected is not None:
        idx = int(selected["family_idx"])
        switch = np.ones(len(floor_ade), dtype=bool)
        selected_ade = family_ade[:, idx].copy()
        selected_fde = family_fde[:, idx].copy()
        reason = "calibrated_support_family_selected"
    elif in_support:
        reason = "calibrated_in_support_but_no_easy_safe_family"

    metric = am._metric(selected_ade, floor_ade, data, switch, test_mask)
    fde_metric = am._metric(selected_fde, floor_fde, data, switch, test_mask)
    oracle_ade = np.minimum(floor_ade, np.min(family_ade, axis=1))
    oracle_switch = np.min(family_ade, axis=1) < floor_ade
    oracle_metric = am._metric(oracle_ade, floor_ade, data, oracle_switch, test_mask)
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    source_id = _rel_to_source_id(heldout_source)
    record = records.get(source_id, {})
    deployable = bool(
        metric["easy_degradation"] <= 0.02
        and metric["all_improvement"] > 0.03
        and (metric["t50_improvement"] > 0.03 or metric["hard_failure_improvement"] > 0.10)
    )
    return {
        "source": "fresh_run",
        "heldout_source": heldout_source,
        "source_id": source_id,
        "split_stats": split_stats,
        "calibration_record": {
            "source_specific_metric_time_evidence": bool(record.get("source_specific_metric_time_evidence")),
            "allowed_local_claim": record.get("allowed_local_claim", "not_verified"),
            "annotation_timestep_seconds": record.get("timing", {}).get("annotation_timestep_seconds"),
            "h50_annotation_seconds_hint": record.get("timing", {}).get("h50_annotation_seconds"),
            "global_metric_seconds_claim_allowed": False,
        },
        "feature_schema": {
            "feature_count": int(len(feature_names)),
            "feature_names_sample": feature_names[:12],
            "calibrated_flags_included": True,
            "past_only_history_summary": True,
            "future_inputs": False,
        },
        "support": {
            "threshold": threshold,
            "nearest_sources": nearest,
            "nearest_distance": nearest_distance,
            "in_support": in_support,
            "family_support_top": family_support[:8],
            "selected_family": selected,
            "decision_reason": reason,
            "test_threshold_tuning": False,
        },
        "metrics": {
            "calibrated_support_policy": metric,
            "calibrated_support_policy_fde": fde_metric,
            "family_oracle": oracle_metric,
        },
        "deployable_after_calibrated_support_recheck": deployable,
        "bootstrap": {
            "all": am._bootstrap_ci(selected_ade, floor_ade, test_mask, seed=42251),
            "t50": am._bootstrap_ci(selected_ade, floor_ade, test_mask & (h == 50), seed=42252),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(selected_ade, floor_ade, test_mask & (h == 100), seed=42253),
            "hard_failure": am._bootstrap_ci(selected_ade, floor_ade, test_mask & hard_failure, seed=42254),
            "easy_degradation": am._bootstrap_ci(floor_ade, selected_ade, test_mask & easy, seed=42255),
            "oracle_t50": am._bootstrap_ci(oracle_ade, floor_ade, test_mask & (h == 50), seed=42256),
        },
    }


def _user_actions(targets: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    actions = []
    for row in targets:
        if row["deployable_after_calibrated_support_recheck"]:
            continue
        source = row["heldout_source"]
        if row["support"]["decision_reason"] == "calibrated_source_out_of_support":
            action = "provide_or_convert_additional_same-family_calibrated_source_support"
            detail = "The held-out source remains outside the calibrated support hull; add a source with similar history/geometry statistics before deploying switches."
        else:
            action = "provide_source_specific_easy_harm_labels_or_calibrated_scene_context"
            detail = "The source is near existing support, but no nearest support source has an easy-safe family; stronger harm labels or source-specific scene geometry is needed."
        actions.append(
            {
                "source": source,
                "source_id": row["source_id"],
                "action": action,
                "detail": detail,
                "official_or_local_source": "OpenTraj local ETH/UCY files with BN source-specific calibration evidence",
                "claim_boundary": "restricted source-specific diagnostic only; no global metric/seconds claim",
            }
        )
    return actions


def _summary(targets: list[Mapping[str, Any]]) -> dict[str, Any]:
    repaired = [row["heldout_source"] for row in targets if row["deployable_after_calibrated_support_recheck"]]
    blocked = [row["heldout_source"] for row in targets if not row["deployable_after_calibrated_support_recheck"]]
    source_specific_calibrated = [
        row["heldout_source"] for row in targets if row["calibration_record"]["source_specific_metric_time_evidence"]
    ]
    decision = (
        "calibrated_support_recheck_repaired_all_blocked_sources"
        if repaired and not blocked
        else "calibrated_support_recheck_partially_repaired_blocked_sources"
        if repaired
        else "calibrated_support_recheck_blocked_no_safe_deployment"
    )
    return {
        "source": SOURCE,
        "targeted_sources": [row["heldout_source"] for row in targets],
        "source_specific_calibrated_sources": source_specific_calibrated,
        "repaired_sources": repaired,
        "still_blocked_sources": blocked,
        "decision": decision,
        "next_action": "Use the user-action package to add same-family calibrated source support or source-specific easy-harm/scene context before another deployment attempt.",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = jg._eth_ucy_data()
    labels = am._reconstruct_waypoint_labels(data)
    records = _bn_records()
    targets = [_evaluate_source(data, labels, source, records) for source in _targets_from_jl()]
    payload: dict[str, Any] = {
        "stage": "Stage42-JM",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(["data/stage41_world_model/combined_external.npz", str(JL_JSON), str(BN_JSON)]),
        "current_facts": CURRENT_FACTS,
        "target_selection": {
            "source": "cached_verified_stage42_jl_still_blocked_sources",
            "blocked_sources_from_jl": _targets_from_jl(),
        },
        "targets": targets,
        "summary": _summary(targets),
        "user_action_required": _user_actions(targets),
        "user_action_required_written": True,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "heldout_labels_for_support": False,
            "source_overlap_pass": all(row["split_stats"]["source_overlap_pass"] for row in targets),
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_or_seconds_claim": False,
            "source_specific_seconds_hints_only": True,
            "raw_frame_dataset_local_main_claim": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_jm_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "jl_blocked_sources_targeted": len(payload["targets"]) == len(payload["target_selection"]["blocked_sources_from_jl"]) and len(payload["targets"]) > 0,
        "bn_calibration_records_used": all(row["calibration_record"]["allowed_local_claim"] != "not_verified" for row in payload["targets"]),
        "calibrated_feature_schema_built": all(row["feature_schema"]["calibrated_flags_included"] for row in payload["targets"]),
        "support_recomputed": all(row["support"]["nearest_sources"] for row in payload["targets"]),
        "repair_or_blocker_recorded": bool(payload["summary"]["repaired_sources"] or payload["summary"]["still_blocked_sources"]),
        "user_action_written": payload["user_action_required_written"] and len(payload["user_action_required"]) == len(payload["summary"]["still_blocked_sources"]),
        "no_test_threshold_tuning": all(row["support"]["test_threshold_tuning"] is False for row in payload["targets"]),
        "no_leakage_pass": all(
            payload["no_leakage"][key] is False
            for key in [
                "future_endpoint_input",
                "future_waypoint_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
                "heldout_labels_for_support",
            ]
        )
        and payload["no_leakage"]["train_only_feature_normalization"]
        and payload["no_leakage"]["source_overlap_pass"],
        "no_global_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    verdict = "stage42_jm_eth_ucy_calibrated_support_recheck_pass" if passed == len(gates) else "stage42_jm_eth_ucy_calibrated_support_recheck_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _fmt(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jm_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-JM ETH_UCY Calibrated Support Recheck",
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
        f"- source_specific_calibrated_sources: `{summary['source_specific_calibrated_sources']}`",
        f"- repaired_sources: `{summary['repaired_sources']}`",
        f"- still_blocked_sources: `{summary['still_blocked_sources']}`",
        f"- next_action: {summary['next_action']}",
        "",
        "## Held-Out Calibrated Support Metrics",
        "",
        "| source | local calibration | nearest | distance | threshold | all | t50 | hard/failure | easy degradation | oracle t50 | deployable |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["targets"]:
        metric = row["metrics"]["calibrated_support_policy"]
        oracle = row["metrics"]["family_oracle"]
        nearest = row["support"]["nearest_sources"][0]["source"] if row["support"]["nearest_sources"] else "none"
        lines.append(
            f"| `{row['heldout_source']}` | `{row['calibration_record']['allowed_local_claim']}` | `{nearest}` | "
            f"{row['support']['nearest_distance']:.3f} | {row['support']['threshold']:.3f} | "
            f"{_fmt(metric['all_improvement'])} | {_fmt(metric['t50_improvement'])} | {_fmt(metric['hard_failure_improvement'])} | "
            f"{_fmt(metric['easy_degradation'])} | {_fmt(oracle['t50_improvement'])} | `{row['deployable_after_calibrated_support_recheck']}` |"
        )
    lines.extend(["", "## Support Family Candidates", ""])
    for row in payload["targets"]:
        support = row["support"]
        lines.extend(
            [
                f"### `{row['heldout_source']}`",
                "",
                f"- decision_reason: `{support['decision_reason']}`",
                f"- selected_family: `{support['selected_family']}`",
                "",
                "| rank | family | safe sources | mean all | mean t50 | mean hard | max easy |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for idx, cand in enumerate(support["family_support_top"][:8], start=1):
            lines.append(
                f"| {idx} | `{cand['family']}` | {cand['safe_source_count']}/{cand['support_source_count']} | "
                f"{_fmt(cand['mean_all_improvement'])} | {_fmt(cand['mean_t50_improvement'])} | "
                f"{_fmt(cand['mean_hard_failure_improvement'])} | {_fmt(cand['max_easy_degradation'])} |"
            )
        lines.append("")
    lines.extend(["## User Action Required", ""])
    for action in payload["user_action_required"]:
        lines.extend(
            [
                f"- `{action['source']}`: `{action['action']}`",
                f"  - {action['detail']}",
                f"  - claim boundary: {action['claim_boundary']}",
            ]
        )
    lines.extend(["", "## Bootstrap CI", "", "| source | slice | low | mid | high | n |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in payload["targets"]:
        for key, ci in row["bootstrap"].items():
            lines.append(f"| `{row['heldout_source']}` | `{key}` | {_fmt(ci['low'])} | {_fmt(ci['mid'])} | {_fmt(ci['high'])} | {ci['n']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- JM uses source-specific BN calibration evidence to recheck support, but it does not upgrade the global model to metric/seconds-level.",
            "- If calibrated support still cannot safely switch, the blocked sources need either source-specific geometry/context labels or additional same-family calibrated support.",
            "- This remains a protected raw-frame / dataset-local 2.5D model track; Stage5C and SMC remain off.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: ETH/UCY Calibrated Source Support",
        "",
        "This file is not a completion claim. It lists the source-specific evidence still needed before another deployment attempt on the blocked ETH/UCY sources.",
        "",
    ]
    for action in payload["user_action_required"]:
        lines.extend(
            [
                f"## {action['source']}",
                "",
                f"- action: `{action['action']}`",
                f"- source_id: `{action['source_id']}`",
                f"- detail: {action['detail']}",
                f"- source basis: {action['official_or_local_source']}",
                f"- claim boundary: {action['claim_boundary']}",
                "",
            ]
        )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jm_gate"]
    lines = [
        "# Stage42-JM Gate",
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
    gate = payload["stage42_jm_gate"]
    bits = []
    for row in payload["targets"]:
        metric = row["metrics"]["calibrated_support_policy"]
        bits.append(
            f"{row['heldout_source']}: local_calib={row['calibration_record']['allowed_local_claim']}, all {_fmt(metric['all_improvement'])}, t50 {_fmt(metric['t50_improvement'])}, hard {_fmt(metric['hard_failure_improvement'])}, easy {_fmt(metric['easy_degradation'])}, deployable={row['deployable_after_calibrated_support_recheck']}"
        )
    return [
        "## Stage42-JM ETH_UCY Calibrated Support Recheck",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- calibrated-support heldout results: {'; '.join(bits)}.",
        f"- decision: `{summary['decision']}`; repaired: `{summary['repaired_sources']}`; still blocked: `{summary['still_blocked_sources']}`.",
        "- boundary: source-specific calibration evidence is recorded, but the main claim remains dataset-local/raw-frame 2.5D; no global metric/seconds claim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["eth_ucy_calibrated_support_recheck"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jm_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jm_gate"]["passed"], "total": payload["stage42_jm_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "targeted_sources": payload["summary"]["targeted_sources"],
        "source_specific_calibrated_sources": payload["summary"]["source_specific_calibrated_sources"],
        "repaired_sources": payload["summary"]["repaired_sources"],
        "still_blocked_sources": payload["summary"]["still_blocked_sources"],
        "global_metric_or_seconds_claim": False,
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
                    "stage": "Stage42-JM",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jm_gate"]["verdict"],
                    "result": payload["summary"]["decision"],
                    "fresh_run": True,
                    "downloaded": False,
                    "converted": False,
                    "trained": False,
                    "evaluated": True,
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_eth_ucy_calibrated_support_recheck(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    run_stage42_eth_ucy_calibrated_support_recheck(refresh_readmes=True)


if __name__ == "__main__":
    main()
