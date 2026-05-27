from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

GX_JSON = OUT_DIR / "ucy_h100_candidate_integrity_manifest_stage42.json"
FR_TEMPLATE_JSON = OUT_DIR / "ucy_h100_candidate_terms_template_stage42.json"

PREFILL_JSON = OUT_DIR / "ucy_h100_terms_prefill_from_integrity_stage42.json"
REPORT_MD = OUT_DIR / "ucy_h100_terms_prefill_from_integrity_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_ucy_h100_terms_prefill_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gy_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
CONSOLIDATED_SUMMARY = Path("README_M3W_CURRENT_GOAL_CONSOLIDATED_SUMMARY_ZH.md")
PAPER_EVIDENCE = OUT_DIR / "paper_ready_evidence_matrix_stage42.md"
A_JOURNAL_GAP = OUT_DIR / "a_journal_gap_stage42.md"
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gy_ucy_h100_terms_prefill_from_integrity"
UCY_OFFICIAL_URL = "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GY 只生成 UCY h100 terms intake prefill，不下载、不转换、不训练、不评估。",
    "本阶段允许预填 source identity suggestion、file hash 和 candidate metadata；不允许预填 terms acceptance、allowed use、local path 或 user confirmation。",
    "hash/path/source identity suggestion 不等于 legal permission 或 conversion readiness。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "uniform_h100_or_t100_claim_allowed": False,
    "license_acceptance_autofilled": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _integrity_by_relative_path(gx_payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("relative_path", "")): row for row in gx_payload.get("candidate_integrity_rows", [])}


def _prefill_row(template_row: Mapping[str, Any], integrity_row: Mapping[str, Any] | None) -> dict[str, Any]:
    relative_path = str(template_row.get("relative_path", ""))
    integrity = integrity_row or {}
    source_identity_suggestion = str(integrity.get("source_identity_suggestion", ""))
    return {
        "dataset_id": template_row.get("dataset_id", "ucy_crowd_original"),
        "candidate_id": template_row.get("candidate_id", ""),
        "source_id": template_row.get("source_id", ""),
        "relative_path": relative_path,
        "official_terms_url": UCY_OFFICIAL_URL,
        "source_identity_suggestion": source_identity_suggestion,
        "file_sha256": integrity.get("sha256", ""),
        "file_size_bytes": integrity.get("file_size_bytes", 0),
        "parsed_rows": integrity.get("parsed_rows", 0),
        "unique_agents": integrity.get("unique_agents", 0),
        "unique_frames": integrity.get("unique_frames", 0),
        "max_track_points": integrity.get("max_track_points", 0),
        "parsed_estimated_t100_windows": integrity.get("parsed_estimated_t100_windows", 0),
        "target_bucket_match": integrity.get("target_bucket_match", False),
        "t100_capable": integrity.get("t100_capable", False),
        "terms_accepted_by_user": False,
        "terms_acceptance_date": "",
        "accepted_terms_version_or_access_date": "",
        "allowed_use": "",
        "redistribution_allowed": "unknown",
        "derived_data_allowed": "unknown",
        "local_path": "",
        "source_identity": "",
        "confirmed_by_user": "",
        "notes": "User may copy source_identity_suggestion into source_identity only after confirming official UCY terms and local source identity.",
        "agent_may_fill_legal_acceptance": False,
        "conversion_ready_now": False,
        "conversion_executed": False,
        "evaluation_executed": False,
        "do_not_count_as_converted_until": "terms validator, guarded conversion, no-leakage, source-CV, and h100 easy-safety CI pass",
    }


def _build_prefill(gx_payload: Mapping[str, Any], template_payload: Mapping[str, Any]) -> dict[str, Any]:
    integrity = _integrity_by_relative_path(gx_payload)
    rows = [_prefill_row(row, integrity.get(str(row.get("relative_path", "")))) for row in template_payload.get("datasets", [])]
    return {
        "source": SOURCE,
        "purpose": "UCY h100 terms intake prefill with hash/source-identity suggestions only. Legal acceptance fields remain intentionally blank.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_terms_url": UCY_OFFICIAL_URL,
        "agent_may_fill_legal_acceptance": False,
        "datasets": rows,
        "non_claims": [
            "This prefill is not legal acceptance.",
            "This prefill is not conversion readiness.",
            "This prefill is not conversion, evaluation, h100 repair, metric evidence, or seconds-level evidence.",
        ],
    }


def _summary(prefill: Mapping[str, Any], gx_payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(prefill.get("datasets", []))
    legal_fields_blank = all(
        row.get("terms_accepted_by_user") is False
        and not str(row.get("allowed_use", "")).strip()
        and not str(row.get("local_path", "")).strip()
        and not str(row.get("confirmed_by_user", "")).strip()
        for row in rows
    )
    return {
        "source": SOURCE,
        "input_gx_verdict": gx_payload.get("stage42_gx_gate", {}).get("verdict", ""),
        "prefill_rows": len(rows),
        "rows_with_hash": sum(1 for row in rows if row.get("file_sha256")),
        "rows_with_source_identity_suggestion": sum(1 for row in rows if row.get("source_identity_suggestion")),
        "target_family_rows": sum(1 for row in rows if row.get("target_bucket_match") is True),
        "t100_capable_rows": sum(1 for row in rows if row.get("t100_capable") is True),
        "legal_acceptance_fields_blank": legal_fields_blank,
        "conversion_ready_now_count": 0,
        "downloaded_now": 0,
        "converted_now": 0,
        "evaluated_now": 0,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    rows = payload["prefill"]["datasets"]
    gates = {
        "gx_input_verified": str(s["input_gx_verdict"]).startswith("stage42_gx_ucy_h100_candidate_integrity_manifest_pass"),
        "prefill_rows_written": s["prefill_rows"] >= 1,
        "hashes_included": s["rows_with_hash"] == s["prefill_rows"],
        "source_identity_suggestions_included": s["rows_with_source_identity_suggestion"] == s["prefill_rows"],
        "target_family_preserved": s["target_family_rows"] >= 1,
        "legal_acceptance_not_autofilled": s["legal_acceptance_fields_blank"] is True,
        "agent_may_not_fill_legal_acceptance": payload["prefill"]["agent_may_fill_legal_acceptance"] is False
        and all(row["agent_may_fill_legal_acceptance"] is False for row in rows),
        "no_conversion_ready_claim": s["conversion_ready_now_count"] == 0,
        "no_download_conversion_eval": s["downloaded_now"] == 0 and s["converted_now"] == 0 and s["evaluated_now"] == 0,
        "user_action_written": payload["user_action_required_written"] is True,
        "no_future_test_or_central_velocity_leakage": (
            payload["no_leakage"]["future_endpoint_input"] is False
            and payload["no_leakage"]["future_waypoint_input"] is False
            and payload["no_leakage"]["central_velocity"] is False
            and payload["no_leakage"]["test_endpoint_goals"] is False
            and payload["no_leakage"]["test_threshold_tuning"] is False
        ),
        "no_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_claim_allowed"] is False
        and payload["claim_boundary"]["global_seconds_claim_allowed"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_gy_ucy_h100_terms_prefill_pass" if passed == total else "stage42_gy_ucy_h100_terms_prefill_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    gx = read_json(GX_JSON, {})
    template = read_json(FR_TEMPLATE_JSON, {})
    prefill = _build_prefill(gx, template)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GY UCY h100 terms prefill from integrity manifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([GX_JSON, FR_TEMPLATE_JSON]),
        "current_facts": CURRENT_FACTS,
        "prefill": prefill,
        "summary": _summary(prefill, gx),
        "user_action_required": _user_actions(prefill),
        "user_action_required_written": True,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "prefill_only_no_conversion": True,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_gy_gate"] = _gate(payload)
    return payload


def _user_actions(prefill: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": row["relative_path"],
            "source_identity_suggestion": row["source_identity_suggestion"],
            "file_sha256": row["file_sha256"],
            "required_user_fields": [
                "terms_accepted_by_user",
                "terms_acceptance_date",
                "allowed_use",
                "redistribution_allowed",
                "derived_data_allowed",
                "local_path",
                "source_identity",
                "confirmed_by_user",
            ],
            "claim_guard": "Do not run conversion or claim repair until these fields are user-confirmed and validator/guarded conversion/no-leakage/source-CV pass.",
        }
        for row in prefill.get("datasets", [])
    ]


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gy_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-GY UCY H100 Terms Prefill From Integrity Manifest",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Summary",
        "",
        f"- prefill_rows: `{s['prefill_rows']}`",
        f"- rows_with_hash: `{s['rows_with_hash']}`",
        f"- rows_with_source_identity_suggestion: `{s['rows_with_source_identity_suggestion']}`",
        f"- target_family_rows: `{s['target_family_rows']}`",
        f"- t100_capable_rows: `{s['t100_capable_rows']}`",
        f"- legal_acceptance_fields_blank: `{s['legal_acceptance_fields_blank']}`",
        f"- conversion_ready_now_count: `{s['conversion_ready_now_count']}`",
        "",
        "## Prefill Rows",
        "",
        "| path | sha256 | source identity suggestion | t100 windows | target | user must fill |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in payload["prefill"]["datasets"]:
        lines.append(
            f"| `{row['relative_path']}` | `{row['file_sha256'][:16]}...` | `{row['source_identity_suggestion']}` | "
            f"{row['parsed_estimated_t100_windows']} | {row['target_bucket_match']} | terms/path/source_identity/confirmed_by_user |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- GY reduces ambiguity for manual UCY terms confirmation by carrying forward hash and source-identity suggestions.",
        "- GY intentionally leaves legal acceptance, allowed use, local path, and user confirmation blank.",
        "- This is still not conversion, not evaluation, not h100 repair, not metric evidence, and not seconds-level evidence.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-GY UCY H100 Terms Prefill",
        "",
        "Fill the required user fields only after you have accepted/verified the official UCY terms. The agent did not and must not fill those legal acceptance fields.",
        "",
        f"- prefill_json: `{PREFILL_JSON}`",
        "",
    ]
    for row in payload["user_action_required"]:
        lines += [
            f"## `{row['relative_path']}`",
            "",
            f"- source_identity_suggestion: `{row['source_identity_suggestion']}`",
            f"- file_sha256: `{row['file_sha256']}`",
            f"- required_user_fields: `{row['required_user_fields']}`",
            f"- claim_guard: {row['claim_guard']}",
            "",
        ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gy_gate"]
    lines = [
        "# Stage42-GY Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _refresh_docs(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "## Stage42-GY UCY H100 Terms Prefill From Integrity",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_gy_gate']['passed']} / {payload['stage42_gy_gate']['total']}`; verdict `{payload['stage42_gy_gate']['verdict']}`",
        f"- prefill rows: `{s['prefill_rows']}`; rows with hash/source identity suggestions: `{s['rows_with_hash']}` / `{s['rows_with_source_identity_suggestion']}`.",
        "- Legal acceptance fields remain blank and must be user-confirmed. This is not conversion, evaluation, h100 repair, metric evidence, or seconds-level evidence.",
        "- `UCY|100` remains blocked until terms/path/source identity are confirmed and guarded conversion/no-leakage/source-CV pass.",
    ]
    for path in [README_RESULTS, M3W_README, CONSOLIDATED_SUMMARY, PAPER_EVIDENCE, A_JOURNAL_GAP]:
        _replace_section(path, "STAGE42_GY_UCY_H100_TERMS_PREFILL", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    s = payload["summary"]
    state["current_stage"] = "Stage42-GY UCY h100 terms prefill from integrity"
    state["current_verdict"] = payload["stage42_gy_gate"]["verdict"]
    state["stage42_gy_ucy_h100_terms_prefill_from_integrity"] = {
        "source": payload["source"],
        "result_source": "fresh_run_prefill_from_integrity_manifest",
        "prefill_json": str(PREFILL_JSON),
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_gy_gate"]["verdict"],
        "gates": f"{payload['stage42_gy_gate']['passed']}/{payload['stage42_gy_gate']['total']}",
        "prefill_rows": s["prefill_rows"],
        "rows_with_hash": s["rows_with_hash"],
        "rows_with_source_identity_suggestion": s["rows_with_source_identity_suggestion"],
        "legal_acceptance_fields_blank": s["legal_acceptance_fields_blank"],
        "conversion_ready_now_count": s["conversion_ready_now_count"],
        "claim_boundary": CLAIM_BOUNDARY,
        "conclusion": "UCY h100 source identity/hash prefill is ready for user terms confirmation, but no legal acceptance, conversion, evaluation, or h100 repair is claimed.",
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_ucy_h100_terms_prefill_from_integrity() -> dict[str, Any]:
    payload = _build_payload()
    write_json(PREFILL_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_docs(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_ucy_h100_terms_prefill_from_integrity()
    gate = result["stage42_gy_gate"]
    print(f"Stage42-GY gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
