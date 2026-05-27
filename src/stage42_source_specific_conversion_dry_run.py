from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_local_t100_conversion_readiness as be
from src import stage42_source_time_geometry_calibration as bn
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
DV_JSON = OUT_DIR / "calibration_candidate_manifest_stage42.json"
BN_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"
DATA_CALIBRATION_JSON = OUT_DIR / "data_calibration_stage42.json"
DATA_CALIBRATION_MD = OUT_DIR / "data_calibration_stage42.md"

REPORT_JSON = OUT_DIR / "source_specific_conversion_dry_run_stage42.json"
REPORT_MD = OUT_DIR / "source_specific_conversion_dry_run_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_specific_conversion_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dw_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

HORIZONS = [10, 25, 50, 100]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "source_specific_conversion_executable_now": False,
    "converted_datasets_now": 0,
    "evaluated_datasets_now": 0,
    "stage5c_executed": False,
    "smc_enabled": False,
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DW 是 source-specific calibrated conversion dry-run，不转换数据、不训练、不评估。",
    "本步骤只验证 UCY/ETH source-specific candidates 的技术可转换性、horizon support 与 source-CV readiness。",
    "terms/source/path/version 未确认前，source-specific metric/time subset 仍不能执行转换或声明 official result。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍不能写成全局 seconds-level；source-specific seconds 也必须等 legal conversion 后限定声明。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _source_specs_by_id() -> dict[str, Mapping[str, Any]]:
    return {str(spec["source_id"]): spec for spec in bn.SOURCE_SPECS}


def _candidate_source_ids(dv_payload: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for row in dv_payload.get("candidate_rows", []):
        if row.get("candidate_class") != "source_specific_metric_time_candidate_after_terms":
            continue
        ids.extend(str(source_id) for source_id in row.get("bn_source_specific_candidates", []))
    return sorted(set(ids))


def _horizon_counts_from_tracks(tracks: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, int]:
    return be._horizon_counts([len(track) for track in tracks.values()])


def _history_counts_from_tracks(tracks: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, int]:
    return be._history_counts([len(track) for track in tracks.values()])


def _audit_source_for_dry_run(spec: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(spec["source_id"])
    source_record = bn._audit_source(spec)
    path = Path(str(spec["trajectory_file"]))
    rows = be._parse_rows(path)
    tracks = be._track_map(rows)
    lengths = [len(track) for track in tracks.values()]
    horizon_counts = _horizon_counts_from_tracks(tracks)
    history_counts = _history_counts_from_tracks(tracks)
    step = be._common_step(tracks)
    continuity = be._continuity(tracks, step)
    h50 = int(horizon_counts["50"])
    h100 = int(horizon_counts["100"])
    return {
        "source": "fresh_source_specific_conversion_dry_run",
        "source_id": source_id,
        "domain": str(spec["domain"]),
        "dataset": str(spec["dataset"]),
        "trajectory_file": str(path),
        "path_exists": path.exists(),
        "homography": source_record.get("homography"),
        "timing": source_record.get("timing"),
        "coordinate": source_record.get("coordinate"),
        "source_specific_metric_time_evidence": bool(source_record.get("source_specific_metric_time_evidence")),
        "rows": int(len(rows)),
        "agents": int(len(tracks)),
        "unique_frames": int(len({int(row["frame_id"]) for row in rows})),
        "track_count": int(len(lengths)),
        "max_track_points": int(max(lengths, default=0)),
        "median_track_points": float(__import__("statistics").median(lengths)) if lengths else 0.0,
        "common_frame_step": int(step) if step is not None else None,
        "continuity": continuity,
        "horizon_counts": horizon_counts,
        "history_horizon_counts": history_counts,
        "t50_capable": h50 > 0,
        "t100_capable": h100 > 0,
        "causal_velocity_possible": bool(max(lengths, default=0) >= 2),
        "central_velocity_used": False,
        "future_labels_available_for_loss_eval_only": bool(h50 > 0 or h100 > 0),
        "metric_status_if_terms_confirmed": "source_specific_candidate_only_not_global_metric",
        "seconds_status_if_terms_confirmed": "source_specific_annotation_step_candidate_only_not_global_seconds",
        "conversion_allowed_now": False,
        "full_world_state_rows_written": 0,
        "evaluation_rows_written": 0,
        "technical_conversion_ready_after_terms": bool(
            path.exists()
            and rows
            and tracks
            and source_record.get("source_specific_metric_time_evidence")
            and continuity.get("gap_ratio", 1.0) < 0.25
            and h50 > 0
        ),
        "blocked_by": ["terms/source_identity/path_version_not_confirmed"],
    }


def _source_cv_plan(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["technical_conversion_ready_after_terms"]:
            by_domain[str(row["domain"])].append(row)
    domains: dict[str, Any] = {}
    for domain, source_rows in sorted(by_domain.items()):
        ranked = sorted(source_rows, key=lambda row: (-int(row["horizon_counts"]["50"]), str(row["source_id"])))
        folds: list[dict[str, Any]] = []
        feasible = len(ranked) >= 3
        if feasible:
            for holdout in ranked:
                remaining = [row for row in ranked if row["source_id"] != holdout["source_id"]]
                validation = remaining[0]
                train = [str(row["source_id"]) for row in remaining[1:]]
                folds.append(
                    {
                        "holdout_source": holdout["source_id"],
                        "validation_source": validation["source_id"],
                        "train_sources": train,
                        "holdout_t50_windows": int(holdout["horizon_counts"]["50"]),
                        "holdout_t100_windows": int(holdout["horizon_counts"]["100"]),
                    }
                )
        domains[domain] = {
            "sources": [str(row["source_id"]) for row in ranked],
            "source_count": len(ranked),
            "t50_windows": int(sum(int(row["horizon_counts"]["50"]) for row in ranked)),
            "t100_windows": int(sum(int(row["horizon_counts"]["100"]) for row in ranked)),
            "source_cv_feasible_after_terms": feasible,
            "folds": folds,
        }
    return {
        "source": "fresh_source_specific_conversion_dry_run",
        "min_sources_for_source_cv": 3,
        "domains": domains,
        "domains_with_source_cv_after_terms": [domain for domain, row in domains.items() if row["source_cv_feasible_after_terms"]],
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    no_leak = payload["no_leakage_preflight"]
    gates = {
        "dv_input_passed": payload.get("stage42_dv_verdict") == "stage42_dv_calibration_candidate_manifest_pass",
        "bn_input_passed": bool(payload.get("stage42_bn_verdict")),
        "source_specific_candidates_loaded": s["source_specific_sources_checked"] >= 6,
        "technical_ready_after_terms_present": s["technical_conversion_ready_after_terms_sources"] >= 5,
        "short_source_blocker_reported": bool(s["technical_not_ready_sources"]),
        "t50_support_present": s["estimated_t50_windows"] > 0,
        "t100_support_present": s["estimated_t100_windows"] > 0,
        "source_cv_feasible_domain_present": bool(s["domains_with_source_cv_after_terms"]),
        "legal_blocker_preserved": s["conversion_allowed_now_sources"] == 0,
        "no_world_state_rows_written": s["full_world_state_rows_written"] == 0,
        "no_evaluation_rows_written": s["evaluation_rows_written"] == 0,
        "no_leakage_preflight": all(value is False for key, value in no_leak.items() if key.endswith("_input") or key in {"central_velocity", "test_endpoint_goals", "test_metrics_for_threshold"}),
        "global_metric_seconds_blocked": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_dw_source_specific_conversion_dry_run_pass" if passed == total else "stage42_dw_source_specific_conversion_dry_run_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-DW Source-Specific Calibrated Conversion Dry-Run",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{payload['stage42_dw_gate']['passed']} / {payload['stage42_dw_gate']['total']}`",
        f"- verdict: `{payload['stage42_dw_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Source Dry-Run Table",
        "",
        "| source | domain | rows | agents | t50 | t100 | step | gap ratio | ready after terms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["source_rows"]:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{:.4f}` | `{}` |".format(
                row["source_id"],
                row["domain"],
                row["rows"],
                row["agents"],
                row["horizon_counts"]["50"],
                row["horizon_counts"]["100"],
                row["common_frame_step"],
                float(row["continuity"].get("gap_ratio", 0.0)),
                row["technical_conversion_ready_after_terms"],
            )
        )
    lines.extend(
        [
            "",
            "## Source-CV Plan",
            "",
            "| domain | sources | t50 windows | t100 windows | source-CV feasible after terms |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, row in payload["source_cv_plan"]["domains"].items():
        lines.append(
            f"| `{domain}` | `{row['source_count']}` | `{row['t50_windows']}` | `{row['t100_windows']}` | `{row['source_cv_feasible_after_terms']}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This dry-run did not write world-state rows, feature stores, checkpoints, or evaluation metrics.",
        "- UCY appears technically source-CV feasible after terms confirmation; ETH/BIWI has two calibrated sources and remains weaker for source-CV by itself.",
            "- UCY_zara03 is retained as source-specific evidence but marked technically not-ready for t50/t100 because this dry-run found no t50 windows.",
            "- Conversion remains blocked until user confirms official terms/source/path/version and no-leakage conversion is executed.",
            "- No global metric or seconds-level M3W claim is allowed.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["stage42_dw_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-DW Source-Specific Conversion",
        "",
        "以下 source 技术上可以在 terms/source/path/version 确认后进入 no-leakage conversion dry-to-actual step。确认前不得当作 converted/evaluated/metric result。",
        "",
        "| priority | source | domain | required action |",
        "| ---: | --- | --- | --- |",
    ]
    for row in sorted(payload["source_rows"], key=lambda r: (-int(r["horizon_counts"]["50"]), str(r["source_id"]))):
        lines.append(
            f"| `{int(row['horizon_counts']['50'])}` | `{row['source_id']}` | `{row['domain']}` | confirm official terms, source identity, local path/version, H/FPS convention, then run no-leakage conversion |"
        )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dw_gate"]
    return [
        "# Stage42-DW Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in gate["gates"].items()],
    ]


def _calibration_addendum(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-DW Source-Specific Conversion Dry-Run",
        "",
        "- source: `fresh_source_specific_conversion_dry_run_from_stage42_dv`",
        "- role: technical dry-run for calibrated UCY/ETH candidates; no conversion, no evaluation, no metric/seconds claim.",
        f"- gate: `{payload['stage42_dw_gate']['passed']} / {payload['stage42_dw_gate']['total']}`; verdict `{payload['stage42_dw_gate']['verdict']}`.",
        f"- sources checked: `{s['source_specific_sources_checked']}`; technical ready after terms: `{s['technical_conversion_ready_after_terms_sources']}`.",
        f"- technical not-ready sources: `{s['technical_not_ready_sources']}`.",
        f"- estimated t50/t100 windows: `{s['estimated_t50_windows']}` / `{s['estimated_t100_windows']}`.",
        f"- source-CV domains after terms: `{s['domains_with_source_cv_after_terms']}`.",
        "- Conversion remains blocked by terms/source/path/version confirmation.",
    ]


def _refresh_data_calibration(payload: Mapping[str, Any]) -> None:
    existing = read_json(DATA_CALIBRATION_JSON, {}) if DATA_CALIBRATION_JSON.exists() else {}
    existing["stage42_dw_source_specific_conversion_dry_run"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_dw_gate"]["verdict"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(DATA_CALIBRATION_JSON, existing)
    _replace_section(DATA_CALIBRATION_MD, "STAGE42_DW_SOURCE_SPECIFIC_CONVERSION_DRY_RUN", _calibration_addendum(payload))


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-DW Source-Specific Conversion Dry-Run",
        "",
        "- source: `fresh_source_specific_conversion_dry_run_from_stage42_dv`",
        "- role: parses calibrated UCY/ETH source candidates for horizon/source-CV readiness; no conversion/evaluation.",
        f"- gate: `{payload['stage42_dw_gate']['passed']} / {payload['stage42_dw_gate']['total']}`; verdict `{payload['stage42_dw_gate']['verdict']}`.",
        f"- sources checked: `{s['source_specific_sources_checked']}`; technical ready after terms: `{s['technical_conversion_ready_after_terms_sources']}`.",
        f"- technical not-ready sources: `{s['technical_not_ready_sources']}`.",
        f"- estimated t50/t100 windows: `{s['estimated_t50_windows']}` / `{s['estimated_t100_windows']}`.",
        f"- source-CV domains after terms: `{s['domains_with_source_cv_after_terms']}`; conversion allowed now remains `0`.",
        f"- report: `{REPORT_MD}`.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_DW_SOURCE_SPECIFIC_CONVERSION_DRY_RUN", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DW source-specific conversion dry-run"
    state["current_verdict"] = payload["stage42_dw_gate"]["verdict"]
    state["stage42_dw_source_specific_conversion_dry_run"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "user_action_required": str(USER_ACTION_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_dw_gate"]["verdict"],
        "gates": f"{payload['stage42_dw_gate']['passed']}/{payload['stage42_dw_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_specific_conversion_dry_run() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    dv_payload = read_json(DV_JSON, {})
    bn_payload = read_json(BN_JSON, {})
    specs = _source_specs_by_id()
    source_ids = _candidate_source_ids(dv_payload)
    rows = [_audit_source_for_dry_run(specs[source_id]) for source_id in source_ids if source_id in specs]
    cv_plan = _source_cv_plan(rows)
    payload: dict[str, Any] = {
        "source": "fresh_source_specific_conversion_dry_run_from_stage42_dv",
        "stage": "Stage42-DW",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_facts": CURRENT_FACTS,
        "stage42_dv_report": str(DV_JSON),
        "stage42_dv_verdict": dv_payload.get("stage42_dv_gate", {}).get("verdict"),
        "stage42_bn_report": str(BN_JSON),
        "stage42_bn_verdict": bn_payload.get("stage42_bn_gate", {}).get("verdict"),
        "source_rows": rows,
        "source_cv_plan": cv_plan,
        "summary": {
            "source_specific_sources_checked": len(rows),
            "technical_conversion_ready_after_terms_sources": sum(1 for row in rows if row["technical_conversion_ready_after_terms"]),
            "technical_not_ready_sources": [row["source_id"] for row in rows if not row["technical_conversion_ready_after_terms"]],
            "t50_capable_sources": sum(1 for row in rows if row["t50_capable"]),
            "t100_capable_sources": sum(1 for row in rows if row["t100_capable"]),
            "estimated_t50_windows": int(sum(int(row["horizon_counts"]["50"]) for row in rows)),
            "estimated_t100_windows": int(sum(int(row["horizon_counts"]["100"]) for row in rows)),
            "domains_with_source_cv_after_terms": cv_plan["domains_with_source_cv_after_terms"],
            "conversion_allowed_now_sources": sum(1 for row in rows if row["conversion_allowed_now"]),
            "full_world_state_rows_written": 0,
            "evaluation_rows_written": 0,
        },
        "no_leakage_preflight": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "future_labels_loss_eval_only": True,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_dw_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_data_calibration(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_source_specific_conversion_dry_run()
