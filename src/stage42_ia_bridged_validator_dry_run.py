from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section
from src.stage42_source_terms_confirmation_validator import (
    CF_JSON,
    _confirmation_by_id,
    _load_json,
    _validate_confirmation,
)


OUT_DIR = Path("outputs/stage42_long_research")
IA_BRIDGED_TEMPLATE_JSON = OUT_DIR / "source_terms_confirmation_intake_from_hz_stage42.json"

REPORT_JSON = OUT_DIR / "source_terms_ia_bridged_validator_dry_run_stage42.json"
REPORT_MD = OUT_DIR / "source_terms_ia_bridged_validator_dry_run_stage42.md"
MANIFEST_JSON = OUT_DIR / "source_terms_ia_bridged_readiness_manifest_stage42.json"
MANIFEST_MD = OUT_DIR / "source_terms_ia_bridged_readiness_manifest_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_ia_bridged_validator_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ib_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_SUMMARY = Path("README_M3W_CURRENT_MASTER_SUMMARY_ZH.md")
ROUTES_SUMMARY = Path("README_M3W_RESEARCH_ROUTES_FAILURES_SUCCESSES_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_MATRIX = OUT_DIR / "paper_ready_evidence_matrix_stage42.md"

SECTION = "STAGE42_IB_IA_BRIDGED_VALIDATOR_DRY_RUN"
SOURCE = "fresh_stage42_ib_ia_bridged_validator_dry_run"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IB explicitly validates the IA bridged intake in dry-run mode; it does not activate conversion.",
    "The canonical CG readiness manifest is not overwritten by this dry-run.",
    "local path found 不等于 legal terms accepted，不等于 official source identity confirmed。",
    "future endpoints / waypoints 只作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cf = _load_json(CF_JSON)
    template = _load_json(IA_BRIDGED_TEMPLATE_JSON)
    confirmations = _confirmation_by_id(template)
    validations = [_validate_confirmation(row, confirmations.get(row["id"])) for row in cf["target_decisions"]]
    manifest = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run_only": True,
        "active_validator_input": bool(template.get("active_validator_input", False)),
        "conversion_ready_targets": [row for row in validations if row["conversion_ready"]],
        "blocked_targets": [row for row in validations if not row["conversion_ready"]],
        "conversion_executed": False,
        "evaluation_executed": False,
    }
    return {
        "stage": "Stage42-IB",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([CF_JSON, IA_BRIDGED_TEMPLATE_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "stage42_cf_verdict": cf.get("stage42_cf_gate", {}).get("verdict", ""),
            "bridged_template_source": template.get("source", ""),
            "bridged_template_path": str(IA_BRIDGED_TEMPLATE_JSON),
            "active_validator_input": bool(template.get("active_validator_input", False)),
        },
        "summary": {
            "targets_validated": len(validations),
            "terms_accepted_targets": sum(1 for row in validations if row["terms_accepted_by_user"]),
            "conversion_ready_targets": sum(1 for row in validations if row["conversion_ready"]),
            "blocked_targets": sum(1 for row in validations if not row["conversion_ready"]),
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "dry_run_only": True,
            "canonical_manifest_overwritten": False,
        },
        "validations": validations,
        "manifest": manifest,
        "actions": {"downloaded": False, "converted": False, "trained": False, "evaluated": False},
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    gates = {
        "cf_input_verified": payload["input_reports"]["stage42_cf_verdict"] == "stage42_cf_source_conversion_legal_gate_pass",
        "ia_bridged_template_loaded": IA_BRIDGED_TEMPLATE_JSON.exists(),
        "ia_template_source_recognized": payload["input_reports"]["bridged_template_source"] == "fresh_stage42_ia_hz_to_cg_intake_bridge",
        "bridged_template_inactive": payload["input_reports"]["active_validator_input"] is False,
        "all_targets_validated": s["targets_validated"] >= 5,
        "dry_run_manifest_written": MANIFEST_JSON.exists() and MANIFEST_MD.exists(),
        "blank_hz_blocks_conversion": s["conversion_ready_targets"] == 0,
        "no_conversion_claim": s["converted_datasets_now"] == 0,
        "no_evaluation_claim": s["evaluated_datasets_now"] == 0,
        "canonical_manifest_not_overwritten": s["canonical_manifest_overwritten"] is False,
        "user_action_written": USER_ACTION_MD.exists(),
        "readmes_updated": bool(payload.get("readme_updates", {}).get("readmes_updated", False)),
        "paper_matrix_updated": bool(payload.get("readme_updates", {}).get("paper_matrix_updated", False)),
        "no_metric_seconds_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_ib_ia_bridged_validator_dry_run_pass" if passed == total else "stage42_ib_ia_bridged_validator_dry_run_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _validation_table(validations: list[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| dataset | terms accepted | conversion ready | CF blockers | confirmation blockers |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in validations:
        lines.append(
            f"| `{row['dataset_id']}` | `{row['terms_accepted_by_user']}` | `{row['conversion_ready']}` | "
            f"{', '.join(row['cf_blockers']) or 'none'} | {', '.join(row['confirmation_blockers']) or 'none'} |"
        )
    return lines


def _write_reports(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    payload_for_json = dict(payload)
    payload_for_json["stage42_ib_gate"] = gate
    write_json(REPORT_JSON, payload_for_json)
    write_json(MANIFEST_JSON, {"source": SOURCE, "manifest": payload["manifest"], "stage42_ib_gate": gate})
    lines = [
        "# Stage42-IB IA Bridged Validator Dry Run",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate.get('passed', 'pending')} / {gate.get('total', 'pending')}`",
        f"- verdict: `{gate.get('verdict', 'pending')}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Validation Table",
        "",
        *_validation_table(payload["validations"]),
        "",
        "## Interpretation",
        "",
        "- IB confirms the IA bridged intake is structurally compatible with CG validation semantics.",
        "- It keeps conversion-ready targets at zero because the HZ confirmation fields remain blank.",
        "- This dry-run writes a separate manifest and does not overwrite the canonical CG manifest.",
    ]
    write_md(REPORT_MD, lines)
    manifest = payload["manifest"]
    write_md(
        MANIFEST_MD,
        [
            "# Stage42-IB IA Bridged Readiness Manifest",
            "",
            f"- source: `{SOURCE}`",
            f"- dry_run_only: `{manifest['dry_run_only']}`",
            f"- active_validator_input: `{manifest['active_validator_input']}`",
            f"- conversion_ready_targets: `{len(manifest['conversion_ready_targets'])}`",
            f"- blocked_targets: `{len(manifest['blocked_targets'])}`",
            f"- conversion_executed: `{manifest['conversion_executed']}`",
            f"- evaluation_executed: `{manifest['evaluation_executed']}`",
            "",
            "This is a dry-run readiness manifest only. It is not a converted dataset manifest.",
        ],
    )
    write_md(
        USER_ACTION_MD,
        [
            "# User Action Required: Stage42-IB IA Bridged Validator",
            "",
            "The IA bridged intake validates structurally, but no source is conversion-ready.",
            "",
            f"- bridged intake: `{IA_BRIDGED_TEMPLATE_JSON}`",
            f"- dry-run manifest: `{MANIFEST_MD}`",
            "",
            "Fill and confirm HZ source/terms fields first. Then rerun IA and this IB dry-run. Only after a nonzero ready target and a future guarded conversion/no-leakage/source-CV pass may any converted/evaluated claim be made.",
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage42-IB Gate",
            "",
            f"- verdict: `{gate.get('verdict', 'pending')}`",
            f"- passed: `{gate.get('passed', 'pending')} / {gate.get('total', 'pending')}`",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate.get("gates", {}).items()],
        ],
    )


def _refresh_lines(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> list[str]:
    return [
        "## Stage42-IB IA Bridged Validator Dry Run",
        "",
        f"- source: `{payload['source']}`",
        "- role: dry-run validate the IA bridged intake using CG source-terms semantics without activating conversion.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- targets validated: `{payload['summary']['targets_validated']}`; ready targets: `{payload['summary']['conversion_ready_targets']}`.",
        "- Current result is correctly blocked because HZ user-confirmation fields remain blank.",
        "- Claim boundary: no download, no conversion, no training, no evaluation, no metric/seconds-level claim, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, bool]:
    lines = _refresh_lines(payload, gate)
    readme_paths = [README_RESULTS, M3W_README, MASTER_SUMMARY, ROUTES_SUMMARY]
    for path in readme_paths:
        _replace_section(path, SECTION, lines)
    matrix_lines = [
        "## Stage42-IB IA Bridged Validator Dry Run",
        "",
        "- IB validates the IA bridged intake against CG source-terms semantics in dry-run mode.",
        f"- gate: `{gate['passed']} / {gate['total']}`.",
        f"- ready targets: `{payload['summary']['conversion_ready_targets']}`.",
        "- No conversion/evaluation occurred; source/legal blockers remain until user confirmation and guarded conversion pass.",
    ]
    _replace_section(PAPER_MATRIX, SECTION, matrix_lines)
    return {
        "readmes_updated": all(SECTION in path.read_text(encoding="utf-8") for path in readme_paths),
        "paper_matrix_updated": SECTION in PAPER_MATRIX.read_text(encoding="utf-8"),
    }


def _refresh_state(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-IB IA bridged validator dry run"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_ib_ia_bridged_validator_dry_run"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "manifest": str(MANIFEST_MD),
        "manifest_json": str(MANIFEST_JSON),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "gates": f"{gate['passed']}/{gate['total']}",
        "verdict": gate["verdict"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, MANIFEST_MD, MANIFEST_JSON, GATE_MD, USER_ACTION_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, state)


def run_stage42_ia_bridged_validator_dry_run() -> dict[str, Any]:
    payload = _build_payload()
    pending = {"passed": "pending", "total": "pending", "verdict": "pending", "gates": {}}
    _write_reports(payload, pending)
    readme_updates = _refresh_readmes(payload, pending | {"passed": "pending", "total": "pending", "verdict": "pending"})
    payload["readme_updates"] = readme_updates
    gate = _gate(payload)
    _write_reports(payload, gate)
    readme_updates = _refresh_readmes(payload, gate)
    payload["readme_updates"] = readme_updates
    gate = _gate(payload)
    payload["stage42_ib_gate"] = gate
    write_json(REPORT_JSON, payload)
    _refresh_state(payload, gate)
    return payload


if __name__ == "__main__":
    result = run_stage42_ia_bridged_validator_dry_run()
    gate = result["stage42_ib_gate"]
    print(f"Stage42-IB IA bridged validator dry run: {gate['verdict']} ({gate['passed']}/{gate['total']})")
