from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_local_calibrated_source_terms_prefill import (
    PREFILL_JSON,
    _git_commit,
    _jsonable,
)
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
TERMS_TEMPLATE_JSON = OUT_DIR / "local_calibrated_source_terms_template_stage42.json"
REPORT_JSON = OUT_DIR / "local_calibrated_source_terms_validation_stage42.json"
REPORT_MD = OUT_DIR / "local_calibrated_source_terms_validation_stage42.md"
QUEUE_JSON = OUT_DIR / "local_calibrated_source_conversion_queue_stage42.json"
QUEUE_MD = OUT_DIR / "local_calibrated_source_conversion_queue_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jq_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_local_calibrated_terms_validation_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JQ_LOCAL_CALIBRATED_SOURCE_TERMS_VALIDATION"
SOURCE = "fresh_stage42_jq_local_calibrated_source_terms_validator"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JQ validates user-filled local calibrated source terms; it does not accept terms for the user.",
    "Blank or partially filled terms templates block conversion.",
    "Conversion readiness requires explicit user terms acceptance, official/source URL confirmation, allowed use, source identity, scope confirmation, and local path existence.",
    "No download, conversion, training, evaluation, metric/seconds claim, Stage5C, or SMC is executed here.",
]

REQUIRED_FIELDS = [
    "official_url",
    "official_terms_url",
    "license_name",
    "terms_accepted_by_user",
    "accepted_by_user",
    "accepted_at_utc",
    "allowed_use",
    "source_identity_confirmed",
    "conversion_scope_confirmed",
]

SAFE_ALLOWED_USE_VALUES = {
    "research_only",
    "academic_research",
    "academic_noncommercial",
    "noncommercial_research",
    "research_and_education",
    "commercial_allowed",
}


def _prefill_by_dataset(prefill: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("dataset_name", "")): row for row in prefill.get("datasets", [])}


def _dataset_rows(template: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(template.get("datasets", []))


def _string_present(row: Mapping[str, Any], key: str) -> bool:
    return bool(str(row.get(key, "")).strip())


def _official_url_acceptable(row: Mapping[str, Any], prefill_row: Mapping[str, Any] | None) -> bool:
    official_url = str(row.get("official_url", "")).strip()
    if not official_url.startswith(("http://", "https://")):
        return False
    if not prefill_row:
        return True
    candidates = set(str(url).strip() for url in prefill_row.get("official_url_candidates", []) if str(url).strip())
    preferred = str(prefill_row.get("preferred_official_url", "")).strip()
    if preferred:
        candidates.add(preferred)
    # Allow a terms page on the same host as an official hint, but never a blank/random local path.
    if official_url in candidates:
        return True
    official_host = official_url.split("/")[2].lower() if "/" in official_url else ""
    return any(official_host and official_host == str(url).split("/")[2].lower() for url in candidates if url.startswith(("http://", "https://")))


def _validate_row(row: Mapping[str, Any], prefill_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    dataset_name = str(row.get("dataset_name", "")).strip()
    blockers: list[str] = []
    warnings: list[str] = []

    if not dataset_name:
        blockers.append("dataset_name_missing")
    for key in REQUIRED_FIELDS:
        if key == "terms_accepted_by_user":
            if row.get(key) is not True:
                blockers.append("terms_not_accepted_by_user")
        elif key in {"source_identity_confirmed", "conversion_scope_confirmed"}:
            if row.get(key) is not True:
                blockers.append(f"{key}_false")
        elif not _string_present(row, key):
            blockers.append(f"{key}_missing")

    if not _official_url_acceptable(row, prefill_row):
        blockers.append("official_url_not_confirmed_against_prefill")
    terms_url = str(row.get("official_terms_url", "")).strip()
    if terms_url and not terms_url.startswith(("http://", "https://")):
        blockers.append("official_terms_url_not_http")

    allowed_use = str(row.get("allowed_use", "")).strip().lower()
    if not allowed_use:
        if "allowed_use_missing" not in blockers:
            blockers.append("allowed_use_missing")
    elif allowed_use in {"unknown", "unspecified", "not_sure"}:
        blockers.append("allowed_use_unknown")
    elif allowed_use not in SAFE_ALLOWED_USE_VALUES:
        warnings.append("allowed_use_value_not_in_known_safe_set_manual_review_required")

    local_path = Path(str(row.get("local_path", "")).strip())
    if not str(row.get("local_path", "")).strip():
        blockers.append("local_path_missing")
    elif not local_path.exists():
        blockers.append("local_path_not_found")

    if row.get("commercial_use_allowed") is None:
        warnings.append("commercial_use_allowed_not_recorded")
    if row.get("derived_data_allowed") is None:
        warnings.append("derived_data_allowed_not_recorded")
    if row.get("redistribution_allowed") is None:
        warnings.append("redistribution_allowed_not_recorded")
    if prefill_row and prefill_row.get("source_confidence") == "low":
        warnings.append("low_source_confidence_requires_extra_manual_review")

    conversion_ready = not blockers
    return {
        "dataset_name": dataset_name,
        "result_source": "fresh_validation_from_user_terms_template_and_stage42_jp_prefill",
        "local_path": str(row.get("local_path", "")).strip(),
        "official_url": str(row.get("official_url", "")).strip(),
        "official_terms_url": terms_url,
        "license_name": str(row.get("license_name", "")).strip(),
        "terms_accepted_by_user": row.get("terms_accepted_by_user") is True,
        "accepted_by_user": str(row.get("accepted_by_user", "")).strip(),
        "allowed_use": allowed_use,
        "source_identity_confirmed": row.get("source_identity_confirmed") is True,
        "conversion_scope_confirmed": row.get("conversion_scope_confirmed") is True,
        "prefill_source_confidence": prefill_row.get("source_confidence", "missing_prefill") if prefill_row else "missing_prefill",
        "technical_ready_after_terms": bool(prefill_row and prefill_row.get("technical_ready_after_terms")),
        "t50_rows": int(prefill_row.get("t50_rows", 0)) if prefill_row else 0,
        "t100_rows": int(prefill_row.get("t100_rows", 0)) if prefill_row else 0,
        "coordinate_unit": prefill_row.get("coordinate_unit", "") if prefill_row else "",
        "metric_status": prefill_row.get("metric_status", "") if prefill_row else "",
        "blockers": blockers,
        "warnings": warnings,
        "conversion_ready": conversion_ready,
        "conversion_allowed_now": False,
        "converted_now": False,
        "evaluated_now": False,
        "next_action": "eligible_for_future_guarded_conversion_queue" if conversion_ready else "fill_missing_terms_or_fix_blockers_before_conversion",
    }


def _build_queue(validations: list[Mapping[str, Any]]) -> dict[str, Any]:
    ready = [dict(row) for row in validations if row["conversion_ready"]]
    blocked = [dict(row) for row in validations if not row["conversion_ready"]]
    return {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Queue for a future guarded conversion stage. This file itself does not convert data.",
        "conversion_executed": False,
        "evaluation_executed": False,
        "conversion_ready_datasets": ready,
        "blocked_datasets": blocked,
        "claim_boundary": {
            "queue_is_conversion_permission": False,
            "converted_now": False,
            "evaluated_now": False,
            "global_metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    gates = {
        "jp_prefill_loaded": payload["inputs"]["jp_prefill_exists"] is True,
        "terms_template_loaded": payload["inputs"]["terms_template_exists"] is True,
        "datasets_validated": s["datasets_validated"] >= 3,
        "all_rows_have_blocker_status": all(isinstance(row.get("blockers"), list) for row in payload["validations"]),
        "blank_template_blocks_conversion": s["terms_accepted_rows"] == 0 and s["conversion_ready_rows"] == 0,
        "conversion_queue_written": payload["outputs"]["queue_json"] == str(QUEUE_JSON),
        "conversion_allowed_count_zero": s["conversion_allowed_now_count"] == 0,
        "converted_zero": s["converted_now"] == 0,
        "evaluated_zero": s["evaluated_now"] == 0,
        "no_download_or_training": payload["no_leakage_and_execution"]["download_executed"] is False
        and payload["no_leakage_and_execution"]["training_executed"] is False,
        "no_future_or_test_leakage": payload["no_leakage_and_execution"]["future_endpoint_input"] is False
        and payload["no_leakage_and_execution"]["central_velocity"] is False
        and payload["no_leakage_and_execution"]["test_endpoint_goals"] is False,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    verdict = "stage42_jq_local_calibrated_source_terms_validation_pass" if passed == len(gates) else "stage42_jq_local_calibrated_source_terms_validation_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    prefill = read_json(PREFILL_JSON, {})
    template = read_json(TERMS_TEMPLATE_JSON, {})
    prefill_rows = _prefill_by_dataset(prefill)
    validations = [_validate_row(row, prefill_rows.get(str(row.get("dataset_name", "")))) for row in _dataset_rows(template)]
    queue = _build_queue(validations)
    summary = {
        "datasets_validated": len(validations),
        "terms_accepted_rows": sum(1 for row in validations if row["terms_accepted_by_user"]),
        "conversion_ready_rows": sum(1 for row in validations if row["conversion_ready"]),
        "conversion_allowed_now_count": 0,
        "converted_now": 0,
        "evaluated_now": 0,
        "blocked_rows": [row["dataset_name"] for row in validations if not row["conversion_ready"]],
        "ready_for_future_guarded_conversion": [row["dataset_name"] for row in validations if row["conversion_ready"]],
        "decision": "blocked_until_user_fills_terms_template" if not any(row["conversion_ready"] for row in validations) else "ready_rows_available_for_future_guarded_conversion_only",
    }
    payload: dict[str, Any] = {
        "stage": "Stage42-JQ",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(PREFILL_JSON), str(TERMS_TEMPLATE_JSON)]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "jp_prefill": str(PREFILL_JSON),
            "terms_template": str(TERMS_TEMPLATE_JSON),
            "jp_prefill_exists": PREFILL_JSON.exists(),
            "terms_template_exists": TERMS_TEMPLATE_JSON.exists(),
        },
        "outputs": {
            "report_json": str(REPORT_JSON),
            "report_md": str(REPORT_MD),
            "queue_json": str(QUEUE_JSON),
            "queue_md": str(QUEUE_MD),
            "gate_md": str(GATE_MD),
            "user_action_md": str(USER_ACTION_MD),
        },
        "validations": validations,
        "queue": queue,
        "summary": summary,
        "no_leakage_and_execution": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
        "claim_boundary": {
            "validator_is_permission": False,
            "queue_is_conversion": False,
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_jq_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_jq_gate"]
    lines = [
        "# Stage42-JQ Local Calibrated Source Terms Validation",
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
        f"- decision: `{s['decision']}`",
        f"- datasets_validated: `{s['datasets_validated']}`",
        f"- terms_accepted_rows: `{s['terms_accepted_rows']}`",
        f"- conversion_ready_rows: `{s['conversion_ready_rows']}`",
        f"- conversion_allowed_now_count: `{s['conversion_allowed_now_count']}`",
        f"- converted_now: `{s['converted_now']}`",
        f"- evaluated_now: `{s['evaluated_now']}`",
        f"- blocked_rows: `{s['blocked_rows']}`",
        f"- ready_for_future_guarded_conversion: `{s['ready_for_future_guarded_conversion']}`",
        "",
        "## Validation Table",
        "",
        "| dataset | accepted | ready | blockers | warnings | t50 | t100 |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for row in payload["validations"]:
        lines.append(
            f"| `{row['dataset_name']}` | `{row['terms_accepted_by_user']}` | `{row['conversion_ready']}` | "
            f"`{row['blockers']}` | `{row['warnings']}` | `{row['t50_rows']}` | `{row['t100_rows']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This stage is an intake validator only. It does not convert or evaluate any source.",
            "- A row marked `conversion_ready` would only be eligible for a later guarded conversion stage; it is still not converted here.",
            "- Blank templates correctly remain blocked.",
            "- Dataset-local calibration hints are not global metric or seconds-level claims.",
        ]
    )
    return lines


def _render_queue(queue: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-JQ Local Calibrated Source Conversion Queue",
        "",
        f"- source: `{queue['source']}`",
        f"- generated_at_utc: `{queue['generated_at_utc']}`",
        f"- conversion_executed: `{queue['conversion_executed']}`",
        f"- evaluation_executed: `{queue['evaluation_executed']}`",
        f"- ready_count: `{len(queue['conversion_ready_datasets'])}`",
        f"- blocked_count: `{len(queue['blocked_datasets'])}`",
        "",
        "This queue is only input for a future guarded conversion step. It is not permission and not conversion.",
        "",
        "## Ready Rows",
        "",
    ]
    if queue["conversion_ready_datasets"]:
        for row in queue["conversion_ready_datasets"]:
            lines.append(f"- `{row['dataset_name']}`: t50 `{row['t50_rows']}`, t100 `{row['t100_rows']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Blocked Rows", ""])
    for row in queue["blocked_datasets"]:
        lines.append(f"- `{row['dataset_name']}`: blockers `{row['blockers']}`")
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Local Calibrated Terms Validation",
        "",
        f"- terms_template: `{TERMS_TEMPLATE_JSON}`",
        f"- validation_report: `{REPORT_MD}`",
        "",
        "Fill the template manually only after checking official/source terms. The agent must not fill acceptance fields.",
        "",
    ]
    for row in payload["validations"]:
        if not row["conversion_ready"]:
            lines.extend(
                [
                    f"## {row['dataset_name']}",
                    "",
                    f"- blockers: `{row['blockers']}`",
                    f"- warnings: `{row['warnings']}`",
                    f"- local_path: `{row['local_path']}`",
                    f"- official_url: `{row['official_url']}`",
                    "- required: official URL, official terms URL, license name, accepted-by user/date, allowed use, source identity, conversion scope.",
                    "",
                ]
            )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jq_gate"]
    lines = [
        "# Stage42-JQ Gate",
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
    gate = payload["stage42_jq_gate"]
    s = payload["summary"]
    return [
        "## Stage42-JQ Local Calibrated Source Terms Validation",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- datasets_validated: `{s['datasets_validated']}`; terms_accepted_rows: `{s['terms_accepted_rows']}`; conversion_ready_rows: `{s['conversion_ready_rows']}`.",
        f"- blocked_rows: `{s['blocked_rows']}`; ready_for_future_guarded_conversion: `{s['ready_for_future_guarded_conversion']}`.",
        "- boundary: user terms validator only; no download, no conversion, no evaluation, no metric/seconds overclaim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["local_calibrated_source_terms_validation"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jq_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jq_gate"]["passed"], "total": payload["stage42_jq_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "datasets_validated": payload["summary"]["datasets_validated"],
        "conversion_ready_rows": payload["summary"]["conversion_ready_rows"],
        "converted": False,
        "evaluated": False,
        "global_metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    state["last_successful_command"] = "python run_stage42_local_calibrated_source_terms_validator.py"
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JQ",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jq_gate"]["verdict"],
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


def run_stage42_local_calibrated_source_terms_validator(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_json(QUEUE_JSON, _jsonable(payload["queue"]))
    write_md(REPORT_MD, _render_report(payload))
    write_md(QUEUE_MD, _render_queue(payload["queue"]))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    payload = run_stage42_local_calibrated_source_terms_validator(refresh_readmes=True)
    gate = payload["stage42_jq_gate"]
    print(f"Stage42-JQ local calibrated source terms validation: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
