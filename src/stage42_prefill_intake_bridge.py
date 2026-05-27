from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

PREFILL_JSON = OUT_DIR / "source_terms_confirmation_prefill_stage42.json"
INTAKE_JSON = OUT_DIR / "source_terms_confirmation_intake_template_stage42.json"

REPORT_JSON = OUT_DIR / "prefill_intake_bridge_stage42.json"
REPORT_MD = OUT_DIR / "prefill_intake_bridge_stage42.md"
SNAPSHOT_JSON = OUT_DIR / "source_terms_confirmation_intake_prefilled_snapshot_stage42.json"
USER_ACTION_MD = OUT_DIR / "user_action_required_prefill_intake_bridge_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gc_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gc_prefill_intake_bridge"

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
    "prefill_is_permission": False,
    "conversion_ready_claim": False,
}


def _prefill_by_id(prefill: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("dataset_id", "")): row for row in prefill.get("datasets", [])}


def _has_user_confirmation(row: Mapping[str, Any]) -> bool:
    user = row.get("user_confirmation", {})
    return bool(
        user.get("terms_accepted_by_user") is True
        or str(user.get("terms_acceptance_date", "")).strip()
        or str(user.get("allowed_use", "")).strip()
        or str(user.get("local_path", "")).strip()
        or str(user.get("source_identity", "")).strip()
        or str(user.get("confirmed_by_user", "")).strip()
    )


def _merge_intake_with_prefill(intake: Mapping[str, Any], prefill: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(intake))
    prefill_rows = _prefill_by_id(prefill)
    datasets: list[dict[str, Any]] = []
    for row in intake.get("datasets", []):
        new_row = deepcopy(dict(row))
        dataset_id = str(new_row.get("dataset_id", ""))
        suggestion = prefill_rows.get(dataset_id)
        if suggestion:
            new_row["prefill_suggestion"] = {
                "source": SOURCE,
                "suggested_local_path": suggestion.get("suggested_local_path", ""),
                "suggested_source_identity": suggestion.get("suggested_source_identity", ""),
                "local_path_candidates": suggestion.get("local_path_candidates", []),
                "safe_copy_instruction": suggestion.get("safe_copy_instruction", ""),
                "agent_may_copy_without_user_terms_confirmation": False,
            }
        else:
            new_row["prefill_suggestion"] = {
                "source": SOURCE,
                "suggested_local_path": "",
                "suggested_source_identity": "",
                "local_path_candidates": [],
                "safe_copy_instruction": "No prefill candidate found; user must provide official local path/source identity manually.",
                "agent_may_copy_without_user_terms_confirmation": False,
            }
        new_row["conversion_ready_now"] = False
        new_row["converted_now"] = False
        new_row["evaluated_now"] = False
        datasets.append(new_row)
    merged["datasets"] = datasets
    merged["prefill_bridge"] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prefill_file": str(PREFILL_JSON),
        "rules": [
            "prefill_suggestion is a convenience hint only",
            "user_confirmation remains the only place for accepted terms/path/source identity",
            "blank or suggested-only rows must remain blocked by the validator",
        ],
    }
    return merged


def _summary(merged: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(merged.get("datasets", []))
    return {
        "source": SOURCE,
        "intake_rows": len(rows),
        "rows_with_prefill_suggestion": sum(1 for row in rows if row.get("prefill_suggestion", {}).get("suggested_local_path")),
        "rows_with_user_confirmation": sum(1 for row in rows if _has_user_confirmation(row)),
        "conversion_ready_now": sum(1 for row in rows if row.get("conversion_ready_now") is True),
        "converted_now": sum(1 for row in rows if row.get("converted_now") is True),
        "evaluated_now": sum(1 for row in rows if row.get("evaluated_now") is True),
        "highest_priority_next_action": "User verifies official terms and copies/edits prefill_suggestion into user_confirmation only after confirmation.",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "prefill_loaded": payload.get("input_status", {}).get("prefill_exists") is True,
        "intake_loaded": payload.get("input_status", {}).get("intake_exists") is True,
        "intake_rows_preserved": s.get("intake_rows", 0) >= 5,
        "prefill_suggestions_added": s.get("rows_with_prefill_suggestion", 0) >= 5,
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
        and claim.get("global_seconds_claim_allowed") is False,
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
        "verdict": "stage42_gc_prefill_intake_bridge_pass" if passed == total else "stage42_gc_prefill_intake_bridge_partial",
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GC Prefill -> Intake Bridge",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gc_gate']['passed']} / {payload['stage42_gc_gate']['total']}`",
        f"- verdict: `{payload['stage42_gc_gate']['verdict']}`",
        "",
        "## Role",
        "",
        "- This bridges GB path/source-identity suggestions into the EH intake template as `prefill_suggestion` hints.",
        "- It does not fill `user_confirmation`, accept terms, download, convert, train, evaluate, or mark any dataset ready.",
        "- The validator must still block every row until the user manually confirms terms/path/source identity.",
        "",
        "## Summary",
        "",
        f"- intake_rows: `{s['intake_rows']}`",
        f"- rows_with_prefill_suggestion: `{s['rows_with_prefill_suggestion']}`",
        f"- rows_with_user_confirmation: `{s['rows_with_user_confirmation']}`",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`",
        f"- snapshot: `{SNAPSHOT_JSON}`",
        f"- updated intake template: `{INTAKE_JSON}`",
        "",
        "## Intake Rows",
        "",
        "| dataset | suggested local path | user confirmation filled | conversion ready |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in payload["merged_intake"]["datasets"]:
        suggestion = row.get("prefill_suggestion", {})
        lines.append(
            f"| `{row.get('dataset_id')}` | `{suggestion.get('suggested_local_path') or 'missing'}` | {_has_user_confirmation(row)} | {row.get('conversion_ready_now') is True} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Prefill suggestions are not legal permission and not source conversion readiness.",
            "- Current M3W remains protected dataset-local/raw-frame 2.5D; no true 3D, foundation, metric, seconds-level, Stage5C, or SMC claim.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# User Action Required: Stage42-GC Prefilled Intake",
        "",
        f"- Open `{INTAKE_JSON}`.",
        "- For any dataset you want to unblock, inspect the `prefill_suggestion` block.",
        "- Only after checking official terms, copy/edit the suggested path and source identity into `user_confirmation.local_path` and `user_confirmation.source_identity`.",
        "- Fill `terms_accepted_by_user`, `terms_acceptance_date`, `allowed_use`, `redistribution_allowed`, `derived_data_allowed`, and `confirmed_by_user` yourself.",
        "- Then rerun:",
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
        "# Stage42-GC Gate",
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
        "## Stage42-GC Prefill -> Intake Bridge",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_gc_gate']['passed']} / {payload['stage42_gc_gate']['total']}`; verdict `{payload['stage42_gc_gate']['verdict']}`.",
        "- role: adds GB local path/source identity suggestions into the EH intake template as non-permission `prefill_suggestion` hints.",
        f"- intake rows: `{s['intake_rows']}`; suggestions added `{s['rows_with_prefill_suggestion']}`; user-confirmed rows `{s['rows_with_user_confirmation']}`.",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`; updated intake template `{INTAKE_JSON}`.",
        "- boundary: user_confirmation is still blank; no download/conversion/training/evaluation; protected dataset-local/raw-frame 2.5D only.",
    ]
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GC_PREFILL_INTAKE_BRIDGE", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-GC prefill intake bridge"
    state["current_verdict"] = payload["stage42_gc_gate"]["verdict"]
    state["stage42_gc_prefill_intake_bridge"] = {
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


def run_stage42_prefill_intake_bridge() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    prefill = read_json(PREFILL_JSON, {})
    intake = read_json(INTAKE_JSON, {})
    merged = _merge_intake_with_prefill(intake, prefill)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([PREFILL_JSON, INTAKE_JSON]),
        "input_status": {
            "prefill_exists": PREFILL_JSON.exists(),
            "intake_exists": INTAKE_JSON.exists(),
            "prefill_source": prefill.get("source", ""),
            "intake_source": intake.get("source", ""),
        },
        "merged_intake": merged,
        "summary": _summary(merged),
        "claim_boundary": CLAIM_BOUNDARY,
        "snapshot_written": True,
        "intake_template_updated": True,
        "user_action_required_written": True,
    }
    payload["stage42_gc_gate"] = _gate(payload)
    write_json(INTAKE_JSON, merged)
    write_json(SNAPSHOT_JSON, merged)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    _write_gate(payload["stage42_gc_gate"])
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


__all__ = [
    "run_stage42_prefill_intake_bridge",
    "_merge_intake_with_prefill",
    "_has_user_confirmation",
    "_gate",
]
