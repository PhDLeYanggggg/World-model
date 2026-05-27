from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_prefill_intake_bridge import _has_user_confirmation
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

INTAKE_JSON = OUT_DIR / "source_terms_confirmation_intake_template_stage42.json"
DU_JSON = OUT_DIR / "raw_source_time_geometry_hint_audit_stage42.json"

REPORT_JSON = OUT_DIR / "calibration_hint_intake_bridge_stage42.json"
REPORT_MD = OUT_DIR / "calibration_hint_intake_bridge_stage42.md"
SNAPSHOT_JSON = OUT_DIR / "source_terms_confirmation_intake_calibration_snapshot_stage42.json"
USER_ACTION_MD = OUT_DIR / "user_action_required_calibration_hint_intake_bridge_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gd_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gd_calibration_hint_intake_bridge"

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
    "calibration_hint_is_metric_claim": False,
    "calibration_hint_is_seconds_claim": False,
    "conversion_ready_claim": False,
}


def _hints_by_id(du: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("dataset_id", "")): row for row in du.get("target_rows", [])}


def _compact_hint(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "source": SOURCE,
            "hint_found": False,
            "h_matrix_hint_count": 0,
            "time_metadata_hint_count": 0,
            "frame_stride_hint_count": 0,
            "metric_time_subset_hint": False,
            "claim_allowed_now": False,
            "reason_claim_not_allowed": "no_du_hint_row_for_dataset",
            "selected_examples": {},
        }
    return {
        "source": SOURCE,
        "hint_found": bool(
            row.get("h_matrix_hint_count", 0)
            or row.get("time_metadata_hint_count", 0)
            or row.get("frame_stride_hint_count", 0)
        ),
        "h_matrix_hint_count": int(row.get("h_matrix_hint_count", 0) or 0),
        "time_metadata_hint_count": int(row.get("time_metadata_hint_count", 0) or 0),
        "frame_stride_hint_count": int(row.get("frame_stride_hint_count", 0) or 0),
        "metric_time_subset_hint": bool(row.get("metric_time_subset_hint", False)),
        "legal_conversion_ready": bool(row.get("legal_conversion_ready", False)),
        "claim_allowed_now": False,
        "reason_claim_not_allowed": row.get("reason_claim_not_allowed", "hints_only_and_terms_source_confirmation_missing"),
        "selected_examples": {
            "h_matrix": row.get("h_matrix_hints", [])[:2],
            "time_metadata": row.get("time_metadata_hints", [])[:2],
            "ndjson_fps": row.get("ndjson_fps_hints", [])[:2],
            "frame_stride": row.get("frame_stride_hints", [])[:3],
        },
        "safe_use": "Use only after legal terms/source identity confirmation and source-specific calibration validation; not a global metric/seconds claim.",
    }


def _merge_intake_with_calibration(intake: Mapping[str, Any], du: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(intake))
    hints = _hints_by_id(du)
    datasets: list[dict[str, Any]] = []
    for row in intake.get("datasets", []):
        new_row = deepcopy(dict(row))
        dataset_id = str(new_row.get("dataset_id", ""))
        new_row["calibration_prefill"] = _compact_hint(hints.get(dataset_id))
        new_row["conversion_ready_now"] = False
        new_row["converted_now"] = False
        new_row["evaluated_now"] = False
        datasets.append(new_row)
    merged["datasets"] = datasets
    merged["calibration_hint_bridge"] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "du_report": str(DU_JSON),
        "rules": [
            "calibration_prefill is metadata-only",
            "calibration hints do not permit conversion or metric/seconds claims",
            "user_confirmation remains the legal gate",
            "source-specific no-leakage and calibration validation remain required",
        ],
    }
    return merged


def _summary(merged: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(merged.get("datasets", []))
    return {
        "source": SOURCE,
        "intake_rows": len(rows),
        "rows_with_calibration_prefill": sum(1 for row in rows if "calibration_prefill" in row),
        "rows_with_any_calibration_hint": sum(1 for row in rows if row.get("calibration_prefill", {}).get("hint_found")),
        "rows_with_metric_time_subset_hint": sum(
            1 for row in rows if row.get("calibration_prefill", {}).get("metric_time_subset_hint")
        ),
        "rows_with_user_confirmation": sum(1 for row in rows if _has_user_confirmation(row)),
        "conversion_ready_now": sum(1 for row in rows if row.get("conversion_ready_now") is True),
        "converted_now": sum(1 for row in rows if row.get("converted_now") is True),
        "evaluated_now": sum(1 for row in rows if row.get("evaluated_now") is True),
        "metric_claim_allowed_now": False,
        "seconds_claim_allowed_now": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "du_loaded": payload.get("input_status", {}).get("du_exists") is True,
        "intake_loaded": payload.get("input_status", {}).get("intake_exists") is True,
        "intake_rows_preserved": s.get("intake_rows", 0) >= 5,
        "calibration_prefill_added": s.get("rows_with_calibration_prefill", 0) >= 5,
        "calibration_hints_present": s.get("rows_with_any_calibration_hint", 0) >= 3,
        "metric_time_hints_separated": s.get("rows_with_metric_time_subset_hint", 0) >= 2,
        "user_confirmation_not_auto_filled": s.get("rows_with_user_confirmation") == 0,
        "conversion_ready_zero": s.get("conversion_ready_now") == 0,
        "no_conversion_or_eval": s.get("converted_now") == 0 and s.get("evaluated_now") == 0,
        "snapshot_written": payload.get("snapshot_written") is True,
        "intake_template_updated": payload.get("intake_template_updated") is True,
        "user_action_written": payload.get("user_action_required_written") is True,
        "no_download_conversion_eval": claim.get("download_executed") is False
        and claim.get("conversion_executed") is False
        and claim.get("evaluation_executed") is False,
        "no_metric_seconds_overclaim": claim.get("global_metric_claim_allowed") is False
        and claim.get("global_seconds_claim_allowed") is False
        and s.get("metric_claim_allowed_now") is False
        and s.get("seconds_claim_allowed_now") is False,
        "no_true3d_foundation_overclaim": claim.get("true_3d") is False and claim.get("foundation_world_model") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    return {
        "passed": passed,
        "total": total,
        "gates": gates,
        "verdict": "stage42_gd_calibration_hint_intake_bridge_pass" if passed == total else "stage42_gd_calibration_hint_intake_bridge_partial",
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GD Calibration Hint -> Intake Bridge",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gd_gate']['passed']} / {payload['stage42_gd_gate']['total']}`",
        f"- verdict: `{payload['stage42_gd_gate']['verdict']}`",
        "",
        "## Role",
        "",
        "- This bridges DU metadata-only H/FPS/stride hints into the source terms intake as `calibration_prefill`.",
        "- It does not accept terms, convert, train, evaluate, or create metric/seconds claims.",
        "- Metric/time hints remain source-specific leads only until legal confirmation and calibration validation pass.",
        "",
        "## Summary",
        "",
        f"- intake_rows: `{s['intake_rows']}`",
        f"- rows_with_calibration_prefill: `{s['rows_with_calibration_prefill']}`",
        f"- rows_with_any_calibration_hint: `{s['rows_with_any_calibration_hint']}`",
        f"- rows_with_metric_time_subset_hint: `{s['rows_with_metric_time_subset_hint']}`",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`",
        f"- metric/seconds claim allowed now: `{s['metric_claim_allowed_now']}` / `{s['seconds_claim_allowed_now']}`",
        "",
        "## Intake Rows",
        "",
        "| dataset | H hints | time hints | stride hints | metric/time subset hint | claim allowed |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["merged_intake"]["datasets"]:
        hint = row.get("calibration_prefill", {})
        lines.append(
            f"| `{row.get('dataset_id')}` | {hint.get('h_matrix_hint_count', 0)} | "
            f"{hint.get('time_metadata_hint_count', 0)} | {hint.get('frame_stride_hint_count', 0)} | "
            f"{hint.get('metric_time_subset_hint', False)} | {hint.get('claim_allowed_now', False)} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Calibration hints are not source conversion readiness.",
            "- Calibration hints are not global metric or seconds-level evidence.",
            "- Current M3W remains protected dataset-local/raw-frame 2.5D; no true 3D, foundation, Stage5C, or SMC claim.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# User Action Required: Stage42-GD Calibration Hints",
        "",
        f"- Open `{INTAKE_JSON}`.",
        "- Inspect `calibration_prefill` for H/FPS/stride hints before choosing which source to confirm.",
        "- UCY and ETH rows have metric/time subset hints, but those hints are not claims until official terms/source identity and calibration validation pass.",
        "- After user confirmation, rerun:",
        "",
        "```bash",
        ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
        ".venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py",
        ".venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py",
        ".venv-pytorch/bin/python run_stage42_source_support_closure_audit.py",
        "```",
    ]


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-GD Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    write_md(GATE_MD, lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "## Stage42-GD Calibration Hint -> Intake Bridge",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_gd_gate']['passed']} / {payload['stage42_gd_gate']['total']}`; verdict `{payload['stage42_gd_gate']['verdict']}`.",
        "- role: adds DU metadata-only H/FPS/stride hints into the intake template as non-claim `calibration_prefill` leads.",
        f"- rows with hints: `{s['rows_with_any_calibration_hint']}`; metric/time subset hint rows `{s['rows_with_metric_time_subset_hint']}`.",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`; metric/seconds claim allowed now `{s['metric_claim_allowed_now']}` / `{s['seconds_claim_allowed_now']}`.",
        "- boundary: hints are not permission, not conversion readiness, and not global metric/seconds evidence; Stage5C/SMC remain false.",
    ]
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GD_CALIBRATION_HINT_INTAKE_BRIDGE", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-GD calibration hint intake bridge"
    state["current_verdict"] = payload["stage42_gd_gate"]["verdict"]
    state["stage42_gd_calibration_hint_intake_bridge"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "snapshot": str(SNAPSHOT_JSON),
        "gate": str(GATE_MD),
        "updated_at": payload["generated_at_utc"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_calibration_hint_intake_bridge() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    intake = read_json(INTAKE_JSON, {})
    du = read_json(DU_JSON, {})
    merged = _merge_intake_with_calibration(intake, du)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([INTAKE_JSON, DU_JSON]),
        "input_status": {
            "intake_exists": INTAKE_JSON.exists(),
            "du_exists": DU_JSON.exists(),
            "intake_source": intake.get("source", ""),
            "du_source": du.get("source", ""),
        },
        "merged_intake": merged,
        "summary": _summary(merged),
        "claim_boundary": CLAIM_BOUNDARY,
        "snapshot_written": True,
        "intake_template_updated": True,
        "user_action_required_written": True,
    }
    payload["stage42_gd_gate"] = _gate(payload)
    write_json(INTAKE_JSON, merged)
    write_json(SNAPSHOT_JSON, merged)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    _write_gate(payload["stage42_gd_gate"])
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


__all__ = [
    "run_stage42_calibration_hint_intake_bridge",
    "_merge_intake_with_calibration",
    "_compact_hint",
    "_gate",
]
