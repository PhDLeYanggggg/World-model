from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section
from src.stage42_source_terms_confirmation_validator import run_stage42_source_terms_confirmation_validator


OUT_DIR = Path("outputs/stage42_long_research")
CG_JSON = OUT_DIR / "source_terms_validation_stage42.json"
EH_JSON = OUT_DIR / "source_terms_confirmation_intake_package_stage42.json"
EH_TEMPLATE_JSON = OUT_DIR / "source_terms_confirmation_intake_template_stage42.json"

REPORT_JSON = OUT_DIR / "source_terms_intake_validator_bridge_stage42.json"
REPORT_MD = OUT_DIR / "source_terms_intake_validator_bridge_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ei_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_validator_bridge_from_stage42_eh_intake"

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "converted_datasets_now": 0,
    "evaluated_datasets_now": 0,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _summary(cg: Mapping[str, Any], eh: Mapping[str, Any]) -> dict[str, Any]:
    input_reports = cg.get("input_reports", {})
    s = cg.get("summary", {})
    return {
        "source": SOURCE,
        "validator_template_source": input_reports.get("confirmation_template_source", ""),
        "validator_template_path": input_reports.get("confirmation_template_path", ""),
        "validator_template_format": input_reports.get("confirmation_template_format", ""),
        "eh_verdict": eh.get("stage42_eh_gate", {}).get("verdict", ""),
        "targets_validated": int(s.get("targets_validated", 0)),
        "terms_accepted_targets": int(s.get("terms_accepted_targets", 0)),
        "conversion_ready_targets": int(s.get("conversion_ready_targets", 0)),
        "converted_datasets_now": int(s.get("converted_datasets_now", 0)),
        "evaluated_datasets_now": int(s.get("evaluated_datasets_now", 0)),
        "user_action_required_count": len(cg.get("user_action_required", [])),
        "next_user_file": str(EH_TEMPLATE_JSON),
        "next_validator_command": ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    gates = {
        "eh_input_passed": s["eh_verdict"] == "stage42_eh_source_terms_confirmation_intake_pass",
        "validator_reads_eh_intake": s["validator_template_format"] == "stage42_eh_intake",
        "validator_path_is_eh_template": s["validator_template_path"] == str(EH_TEMPLATE_JSON),
        "targets_validated": s["targets_validated"] >= 5,
        "blank_intake_still_blocks_conversion": s["conversion_ready_targets"] == 0,
        "user_action_preserved": s["user_action_required_count"] >= 5,
        "no_conversion_or_eval_claim": s["converted_datasets_now"] == 0 and s["evaluated_datasets_now"] == 0,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_claim_allowed"] is False
        and payload["claim_boundary"]["global_seconds_claim_allowed"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_ei_intake_validator_bridge_pass" if passed == total else "stage42_ei_intake_validator_bridge_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "# Stage42-EI Source Terms Intake -> Validator Bridge",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ei_gate']['passed']} / {payload['stage42_ei_gate']['total']}`",
        f"- verdict: `{payload['stage42_ei_gate']['verdict']}`",
        "",
        "## Summary",
        "",
        f"- validator_template_source: `{s['validator_template_source']}`",
        f"- validator_template_format: `{s['validator_template_format']}`",
        f"- validator_template_path: `{s['validator_template_path']}`",
        f"- targets_validated: `{s['targets_validated']}`",
        f"- terms_accepted_targets: `{s['terms_accepted_targets']}`",
        f"- conversion_ready_targets: `{s['conversion_ready_targets']}`",
        f"- converted/evaluated now: `{s['converted_datasets_now']}` / `{s['evaluated_datasets_now']}`",
        f"- next_user_file: `{s['next_user_file']}`",
        f"- next_validator_command: `{s['next_validator_command']}`",
        "",
        "## Interpretation",
        "",
        "- The Stage42-CG validator now reads the Stage42-EH intake template path and nested intake format.",
        "- The current blank intake still blocks all conversion, as intended.",
        "- This bridge does not download, convert, train, evaluate, or make metric/seconds-level claims.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_ei_gate"]["gates"].items()],
    ]


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ei_gate"]
    return [
        "# Stage42-EI Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-EI Source Terms Intake Validator Bridge",
        "",
        "- source: `fresh_validator_bridge_from_stage42_eh_intake`",
        "- role: verifies that the CG validator now consumes the EH intake template and nested confirmation schema.",
        f"- gate: `{payload['stage42_ei_gate']['passed']} / {payload['stage42_ei_gate']['total']}`; verdict `{payload['stage42_ei_gate']['verdict']}`.",
        f"- validator_template_format: `{s['validator_template_format']}`; path `{s['validator_template_path']}`.",
        f"- conversion_ready_targets remains `{s['conversion_ready_targets']}`; converted/evaluated now `{s['converted_datasets_now']}` / `{s['evaluated_datasets_now']}`.",
        "- This fixes the EH->CG workflow bridge while preserving legal blocker, no metric/seconds claim, no Stage5C, and no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_EI_SOURCE_TERMS_INTAKE_VALIDATOR_BRIDGE", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EI source terms intake validator bridge"
    state["current_verdict"] = payload["stage42_ei_gate"]["verdict"]
    state["stage42_ei_source_terms_intake_validator_bridge"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ei_gate"]["verdict"],
        "gates": f"{payload['stage42_ei_gate']['passed']}/{payload['stage42_ei_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_terms_intake_validator_bridge(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cg = run_stage42_source_terms_confirmation_validator()
    eh = read_json(EH_JSON, {})
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EI",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([CG_JSON, EH_JSON, EH_TEMPLATE_JSON]),
        "summary": _summary(cg, eh),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_ei_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_source_terms_intake_validator_bridge()
