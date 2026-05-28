from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
JN_JSON = OUT_DIR / "local_calibrated_source_support_intake_stage42.json"
REPORT_JSON = OUT_DIR / "local_calibrated_source_guarded_conversion_preflight_stage42.json"
REPORT_MD = OUT_DIR / "local_calibrated_source_guarded_conversion_preflight_stage42.md"
TERMS_TEMPLATE_JSON = OUT_DIR / "local_calibrated_source_terms_template_stage42.json"
GATE_MD = OUT_DIR / "stage42_stage_jo_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_local_calibrated_conversion_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JO_LOCAL_CALIBRATED_SOURCE_GUARDED_CONVERSION_PREFLIGHT"
SOURCE = "fresh_stage42_jo_local_calibrated_source_guarded_conversion_preflight"


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JO consumes Stage42-JN local support-candidate evidence and builds a guarded conversion preflight.",
    "This stage does not convert Town-Center, Wild-Track, or PETS into the benchmark because dataset-specific terms are not confirmed.",
    "Future endpoints may appear only as future labels after legal conversion; they are not inference inputs in this preflight.",
    "Local calibration files are treated as projection hints, not permission for global metric/seconds claims.",
    "Stage5C and SMC remain disabled.",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _confirmation_template(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    datasets = []
    for row in records:
        datasets.append(
            {
                "dataset_name": row["dataset_name"],
                "local_path": row["root"],
                "official_url": "",
                "official_terms_url": "",
                "license_name": "",
                "terms_accepted_by_user": False,
                "accepted_by_user": "",
                "accepted_at_utc": "",
                "allowed_use": "",
                "commercial_use_allowed": None,
                "derived_data_allowed": None,
                "redistribution_allowed": None,
                "source_identity_confirmed": False,
                "conversion_scope_confirmed": False,
                "notes_from_user": "",
            }
        )
    return {
        "source": SOURCE,
        "purpose": "User-fillable confirmation template. The agent must not fill acceptance fields.",
        "terms_confirmation_is_currently_absent": True,
        "datasets": datasets,
    }


def _legal_blockers(confirmation: Mapping[str, Any]) -> list[str]:
    blockers = []
    if not confirmation.get("terms_accepted_by_user"):
        blockers.append("terms_not_accepted_by_user")
    if not confirmation.get("source_identity_confirmed"):
        blockers.append("source_identity_not_confirmed")
    if not confirmation.get("conversion_scope_confirmed"):
        blockers.append("conversion_scope_not_confirmed")
    if not str(confirmation.get("official_terms_url", "")).strip():
        blockers.append("official_terms_url_missing")
    if not str(confirmation.get("allowed_use", "")).strip():
        blockers.append("allowed_use_missing")
    return blockers


def _geometry_blockers(row: Mapping[str, Any]) -> list[str]:
    status = str(row.get("metric_status", "")).lower()
    unit = str(row.get("coordinate_unit", "")).lower()
    blockers = []
    if "projection_not_integrated" in status:
        blockers.append("world_projection_not_integrated")
    if "pixel" in unit:
        blockers.append("pixel_coordinate_requires_source_specific_projection_audit")
    if "ground_grid" in unit:
        blockers.append("ground_grid_requires_dataset_specific_geometry_audit_before_metric_claim")
    if not row.get("calibration_file_count", 0):
        blockers.append("no_calibration_file_for_metric_or_time_claim")
    return blockers


def _candidate_preflight(row: Mapping[str, Any], confirmation: Mapping[str, Any]) -> dict[str, Any]:
    stats = row.get("stats", {})
    technical_ready = bool(
        row.get("root_exists")
        and row.get("parseable")
        and stats.get("t50_rows", 0) > 0
        and stats.get("t100_rows", 0) > 0
        and row.get("calibration_file_count", 0) > 0
    )
    legal_blockers = _legal_blockers(confirmation)
    geometry_blockers = _geometry_blockers(row)
    conversion_allowed_now = bool(technical_ready and not legal_blockers and not geometry_blockers)
    after_terms_possible = bool(technical_ready and not row.get("legal_auto_convert_allowed", False))
    return {
        "dataset_name": row["dataset_name"],
        "source": "fresh_run_from_stage42_jn_cached_verified_input",
        "technical_ready_for_guarded_conversion_after_terms": technical_ready,
        "after_terms_possible": after_terms_possible,
        "conversion_allowed_now": conversion_allowed_now,
        "conversion_status": "not_run_user_terms_required" if legal_blockers else "not_run_geometry_projection_required" if geometry_blockers else "ready_after_explicit_confirmation",
        "local_path": row["root"],
        "coordinate_unit": row.get("coordinate_unit"),
        "metric_status": row.get("metric_status"),
        "calibration_file_count": row.get("calibration_file_count", 0),
        "t50_rows": int(stats.get("t50_rows", 0)),
        "t100_rows": int(stats.get("t100_rows", 0)),
        "agent_tracks": int(stats.get("agent_tracks", 0)),
        "legal_blockers": legal_blockers,
        "geometry_blockers": geometry_blockers,
        "required_before_conversion": [
            "user confirms dataset-specific terms and official source identity",
            "guarded converter maps rows into source-level feature schema",
            "causal velocity is computed from current/past only",
            "train/val/test or source-CV split is rebuilt without leakage",
            "future endpoints are stored only as labels/evaluation targets",
            "metric/time claims remain disabled until source-specific projection/timing audit passes",
        ],
        "forbidden_claims_now": [
            "converted_dataset",
            "deployable_external_support_source",
            "global_metric_coordinates",
            "seconds_level_t50_or_t100",
            "stage5c_or_smc_readiness",
        ],
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    jn_payload = read_json(JN_JSON, {})
    records = list(jn_payload.get("records", []))
    template = _confirmation_template(records)
    template_by_name = {row["dataset_name"]: row for row in template["datasets"]}
    preflights = [_candidate_preflight(row, template_by_name[row["dataset_name"]]) for row in records]
    payload: dict[str, Any] = {
        "stage": "Stage42-JO",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_report": str(JN_JSON),
        "input_hash": _combined_hash([str(JN_JSON)]),
        "current_facts": CURRENT_FACTS,
        "stage42_jn_summary": jn_payload.get("summary", {}),
        "confirmation_template": template,
        "candidate_preflights": preflights,
        "summary": {
            "candidate_count": len(preflights),
            "technical_ready_after_terms": [row["dataset_name"] for row in preflights if row["technical_ready_for_guarded_conversion_after_terms"]],
            "conversion_allowed_now": [row["dataset_name"] for row in preflights if row["conversion_allowed_now"]],
            "blocked_by_terms": [row["dataset_name"] for row in preflights if "terms_not_accepted_by_user" in row["legal_blockers"]],
            "blocked_by_geometry_audit": [row["dataset_name"] for row in preflights if row["geometry_blockers"]],
            "decision": "guarded_conversion_preflight_blocked_pending_user_terms",
            "next_action": "User confirms dataset-specific official terms/source identity, then rerun this preflight before guarded conversion.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "conversion_executed": False,
            "preflight_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_or_seconds_claim": False,
            "converted_external_support_source": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_jo_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidates = payload["candidate_preflights"]
    template = payload["confirmation_template"]
    gates = {
        "stage42_jn_input_loaded": Path(payload["input_report"]).exists() and len(payload["stage42_jn_summary"]) > 0,
        "candidate_preflights_built": len(candidates) >= 3,
        "technical_preflight_computed": all("technical_ready_for_guarded_conversion_after_terms" in row for row in candidates),
        "confirmation_template_written": len(template.get("datasets", [])) == len(candidates) and len(candidates) > 0,
        "conversion_blocked_without_terms": all(not row["conversion_allowed_now"] for row in candidates),
        "legal_blockers_explicit": all(row["legal_blockers"] for row in candidates),
        "geometry_or_metric_blockers_explicit": all(row["geometry_blockers"] for row in candidates),
        "no_candidate_marked_converted": all(row["conversion_status"] != "converted" for row in candidates),
        "user_action_required": len(payload["summary"]["blocked_by_terms"]) > 0,
        "no_leakage_pass": all(
            payload["no_leakage"][key] is False
            for key in ["future_endpoint_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning", "conversion_executed"]
        )
        and payload["no_leakage"]["preflight_only"],
        "no_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    verdict = "stage42_jo_local_calibrated_source_guarded_preflight_pass" if passed == len(gates) else "stage42_jo_local_calibrated_source_guarded_preflight_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jo_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-JO Local Calibrated Source Guarded Conversion Preflight",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_report: `{payload['input_report']}`",
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
        f"- technical_ready_after_terms: `{summary['technical_ready_after_terms']}`",
        f"- conversion_allowed_now: `{summary['conversion_allowed_now']}`",
        f"- blocked_by_terms: `{summary['blocked_by_terms']}`",
        f"- blocked_by_geometry_audit: `{summary['blocked_by_geometry_audit']}`",
        f"- next_action: {summary['next_action']}",
        "",
        "## Candidate Preflights",
        "",
        "| dataset | technical ready after terms | conversion now | t50 | t100 | legal blockers | geometry blockers | status |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["candidate_preflights"]:
        lines.append(
            f"| `{row['dataset_name']}` | `{row['technical_ready_for_guarded_conversion_after_terms']}` | "
            f"`{row['conversion_allowed_now']}` | {row['t50_rows']} | {row['t100_rows']} | "
            f"`{row['legal_blockers']}` | `{row['geometry_blockers']}` | `{row['conversion_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Conversion Contract",
            "",
            "- Do not convert any candidate until user-confirmed official terms/source identity and conversion scope are recorded.",
            "- After confirmation, rerun this preflight and then run a guarded converter, no-leakage audit, source-CV split, strongest baseline, and protected policy evaluation.",
            "- Future endpoint coordinates may be materialized only as supervised labels/evaluation labels, never as inference inputs.",
            "- Source-specific calibration can support restricted geometry audits only after conversion; it does not authorize global metric or seconds-level claims.",
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
        "# User Action Required: Local Calibrated Source Conversion",
        "",
        "The sources below are technically promising but are not converted. Please fill/confirm the terms template before any guarded conversion.",
        "",
        f"- template_json: `{TERMS_TEMPLATE_JSON}`",
        "",
    ]
    for row in payload["candidate_preflights"]:
        lines.extend(
            [
                f"## {row['dataset_name']}",
                "",
                f"- local_path: `{row['local_path']}`",
                f"- technical_ready_after_terms: `{row['technical_ready_for_guarded_conversion_after_terms']}`",
                f"- t50_rows: `{row['t50_rows']}`; t100_rows: `{row['t100_rows']}`",
                f"- legal_blockers: `{row['legal_blockers']}`",
                f"- geometry_blockers: `{row['geometry_blockers']}`",
                "- required user fields: official_url, official_terms_url, license_name, terms_accepted_by_user, accepted_by_user, allowed_use, source_identity_confirmed, conversion_scope_confirmed.",
                "",
            ]
        )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jo_gate"]
    lines = [
        "# Stage42-JO Gate",
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
    gate = payload["stage42_jo_gate"]
    summary = payload["summary"]
    return [
        "## Stage42-JO Local Calibrated Source Guarded Conversion Preflight",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- technical_ready_after_terms: `{summary['technical_ready_after_terms']}`; conversion_allowed_now: `{summary['conversion_allowed_now']}`.",
        f"- decision: `{summary['decision']}`; blocked_by_terms: `{summary['blocked_by_terms']}`.",
        "- boundary: preflight only; no conversion, no deployable source-support claim, no metric/seconds overclaim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["local_calibrated_source_guarded_conversion_preflight"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jo_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jo_gate"]["passed"], "total": payload["stage42_jo_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "technical_ready_after_terms": payload["summary"]["technical_ready_after_terms"],
        "conversion_allowed_now": payload["summary"]["conversion_allowed_now"],
        "blocked_by_terms": payload["summary"]["blocked_by_terms"],
        "converted": False,
        "global_metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    state["last_successful_command"] = "python run_stage42_local_calibrated_source_guarded_conversion_preflight.py"
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JO",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jo_gate"]["verdict"],
                    "result": payload["summary"]["decision"],
                    "fresh_run": True,
                    "downloaded": False,
                    "converted": False,
                    "trained": False,
                    "evaluated": False,
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_local_calibrated_source_guarded_conversion_preflight(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_json(TERMS_TEMPLATE_JSON, _jsonable(payload["confirmation_template"]))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    payload = run_stage42_local_calibrated_source_guarded_conversion_preflight(refresh_readmes=True)
    gate = payload["stage42_jo_gate"]
    print(f"Stage42-JO local calibrated source guarded conversion preflight: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
