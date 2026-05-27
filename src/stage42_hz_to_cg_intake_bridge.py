from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
HZ_TEMPLATE_JSON = OUT_DIR / "source_terms_confirmation_template_stage42.json"
CG_INTAKE_JSON = OUT_DIR / "source_terms_confirmation_intake_template_stage42.json"

BRIDGE_JSON = OUT_DIR / "source_terms_hz_to_cg_intake_bridge_stage42.json"
BRIDGED_TEMPLATE_JSON = OUT_DIR / "source_terms_confirmation_intake_from_hz_stage42.json"
BRIDGE_MD = OUT_DIR / "source_terms_hz_to_cg_intake_bridge_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_hz_to_cg_intake_bridge_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ia_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_SUMMARY = Path("README_M3W_CURRENT_MASTER_SUMMARY_ZH.md")
ROUTES_SUMMARY = Path("README_M3W_RESEARCH_ROUTES_FAILURES_SUCCESSES_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_MATRIX = OUT_DIR / "paper_ready_evidence_matrix_stage42.md"

SECTION = "STAGE42_IA_HZ_TO_CG_INTAKE_BRIDGE"
SOURCE = "fresh_stage42_ia_hz_to_cg_intake_bridge"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IA 只把 HZ confirmation template 映射成 CG validator 兼容 intake 文件，不下载、不转换、不训练、不评估。",
    "Bridge output is not automatically active validator input; user must intentionally review/copy or a future guarded runner must explicitly select it.",
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


def _hz_by_id(hz_template: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("dataset_id", "")): row for row in hz_template.get("confirmations", [])}


def _hz_to_user_confirmation(hz_row: Mapping[str, Any] | None, fallback_url: str = "") -> dict[str, Any]:
    if not hz_row:
        return {
            "terms_accepted_by_user": False,
            "terms_acceptance_date": "",
            "official_terms_url": fallback_url,
            "accepted_terms_version_or_access_date": "",
            "allowed_use": "",
            "redistribution_allowed": "unknown",
            "derived_data_allowed": "unknown",
            "local_path": "",
            "source_identity": "",
            "confirmed_by_user": "",
            "notes": "No HZ row found for this dataset; remains blocked.",
        }
    redistribution = hz_row.get("redistribution_allowed")
    derived = hz_row.get("derived_data_allowed")
    return {
        "terms_accepted_by_user": bool(hz_row.get("terms_accepted_by_user") is True),
        "terms_acceptance_date": str(hz_row.get("terms_acceptance_date", "")),
        "official_terms_url": str(hz_row.get("official_url") or fallback_url),
        "accepted_terms_version_or_access_date": str(hz_row.get("terms_acceptance_date", "")),
        "allowed_use": str(hz_row.get("allowed_use", "")),
        "redistribution_allowed": "unknown" if redistribution is None else redistribution,
        "derived_data_allowed": "unknown" if derived is None else derived,
        "local_path": str(hz_row.get("local_path", "")),
        "source_identity": str(hz_row.get("source_identity", "")),
        "confirmed_by_user": str(hz_row.get("confirmed_by_user", "")),
        "notes": str(hz_row.get("notes", "")),
        "hz_positive_confirmations": {
            "official_source_url_confirmed": bool(hz_row.get("official_source_url_confirmed") is True),
            "local_path_confirmed": bool(hz_row.get("local_path_confirmed") is True),
            "source_identity_confirmed": bool(hz_row.get("source_identity_confirmed") is True),
        },
    }


def _confirmation_missing(user_confirmation: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    if user_confirmation.get("terms_accepted_by_user") is not True:
        missing.append("terms_accepted_by_user")
    for field in ["terms_acceptance_date", "allowed_use", "local_path", "source_identity", "confirmed_by_user"]:
        if not str(user_confirmation.get(field, "")).strip():
            missing.append(field)
    hz_flags = user_confirmation.get("hz_positive_confirmations", {})
    for field in ["official_source_url_confirmed", "local_path_confirmed", "source_identity_confirmed"]:
        if hz_flags.get(field) is not True:
            missing.append(field)
    return missing


def _bridge_templates(hz_template: Mapping[str, Any], cg_intake: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hz = _hz_by_id(hz_template)
    bridged_datasets = []
    bridge_rows = []
    for row in cg_intake.get("datasets", []):
        dataset_id = str(row.get("dataset_id", ""))
        hz_row = hz.get(dataset_id)
        user_confirmation = _hz_to_user_confirmation(hz_row, row.get("official_url_from_prior_audit", ""))
        bridged = dict(row)
        bridged["user_confirmation"] = user_confirmation
        bridged["conversion_ready_now"] = False
        bridged["converted_now"] = False
        bridged["evaluated_now"] = False
        bridged["stage42_ia_bridge"] = {
            "source": SOURCE,
            "active_validator_input": False,
            "source_hz_template": str(HZ_TEMPLATE_JSON),
            "requires_user_review_before_activation": True,
        }
        bridged_datasets.append(bridged)
        missing = _confirmation_missing(user_confirmation)
        bridge_rows.append(
            {
                "dataset_id": dataset_id,
                "domain": row.get("domain", ""),
                "hz_row_found": hz_row is not None,
                "local_path": user_confirmation.get("local_path", ""),
                "terms_accepted_by_user": user_confirmation.get("terms_accepted_by_user") is True,
                "missing_confirmation_fields": missing,
                "ready_if_activated_now": not missing,
                "active_validator_input": False,
            }
        )
    return (
        {
            "source": SOURCE,
            "purpose": "CG-compatible intake candidate produced from HZ confirmation template. Review before activating.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "agent_may_fill_legal_acceptance": False,
            "active_validator_input": False,
            "datasets": bridged_datasets,
        },
        bridge_rows,
    )


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload["bridge_rows"]
    gates = {
        "hz_template_loaded": HZ_TEMPLATE_JSON.exists(),
        "cg_intake_loaded": CG_INTAKE_JSON.exists(),
        "bridge_output_written": BRIDGE_JSON.exists() and BRIDGED_TEMPLATE_JSON.exists(),
        "all_cg_rows_mapped": len(rows) >= 5,
        "all_rows_have_active_false": all(row["active_validator_input"] is False for row in rows),
        "blank_hz_blocks_ready": payload["summary"]["ready_if_activated_now"] == 0,
        "user_confirmation_preserved": payload["summary"]["user_terms_accepted_count"] == 0,
        "no_download": payload["actions"]["downloaded"] is False,
        "no_conversion": payload["actions"]["converted"] is False,
        "no_training": payload["actions"]["trained"] is False,
        "no_evaluation": payload["actions"]["evaluated"] is False,
        "user_action_written": USER_ACTION_MD.exists(),
        "readmes_updated": bool(payload.get("readme_updates", {}).get("readmes_updated", False)),
        "paper_matrix_updated": bool(payload.get("readme_updates", {}).get("paper_matrix_updated", False)),
        "no_metric_seconds_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_ia_hz_to_cg_intake_bridge_pass" if passed == total else "stage42_ia_hz_to_cg_intake_bridge_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    hz_template = read_json(HZ_TEMPLATE_JSON, {})
    cg_intake = read_json(CG_INTAKE_JSON, {})
    bridged_template, bridge_rows = _bridge_templates(hz_template, cg_intake)
    summary = {
        "hz_rows": len(hz_template.get("confirmations", [])),
        "cg_rows": len(cg_intake.get("datasets", [])),
        "bridge_rows": len(bridge_rows),
        "ready_if_activated_now": sum(1 for row in bridge_rows if row["ready_if_activated_now"]),
        "user_terms_accepted_count": sum(1 for row in bridge_rows if row["terms_accepted_by_user"]),
        "active_validator_input": False,
        "converted_now": 0,
        "evaluated_now": 0,
    }
    return {
        "stage": "Stage42-IA",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([HZ_TEMPLATE_JSON, CG_INTAKE_JSON]),
        "current_facts": CURRENT_FACTS,
        "bridged_template": bridged_template,
        "bridge_rows": bridge_rows,
        "summary": summary,
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


def _rows_table(rows: list[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| dataset | domain | HZ row | local path | ready if activated | missing confirmation |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['dataset_id']}` | `{row.get('domain', '')}` | `{row['hz_row_found']}` | "
            f"`{row.get('local_path', '') or 'blank'}` | `{row['ready_if_activated_now']}` | "
            f"{', '.join(row['missing_confirmation_fields']) or 'none'} |"
        )
    return lines


def _write_reports(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    write_json(BRIDGED_TEMPLATE_JSON, payload["bridged_template"])
    payload_for_json = dict(payload)
    payload_for_json["stage42_ia_gate"] = gate
    write_json(BRIDGE_JSON, payload_for_json)
    lines = [
        "# Stage42-IA HZ to CG Intake Bridge",
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
        "## Bridge Rows",
        "",
        *_rows_table(payload["bridge_rows"]),
        "",
        "## Interpretation",
        "",
        "- IA prevents HZ and CG from drifting into incompatible source-terms schemas.",
        "- The bridged template is deliberately inactive. It is not used by the validator unless a future command or user action explicitly selects/copies it.",
        "- Current ready-if-activated count is zero because HZ confirmation fields remain blank.",
    ]
    write_md(BRIDGE_MD, lines)
    write_md(
        USER_ACTION_MD,
        [
            "# User Action Required: Stage42-IA HZ to CG Intake Bridge",
            "",
            f"Bridged candidate intake: `{BRIDGED_TEMPLATE_JSON}`",
            "",
            "This bridge did not activate conversion. After confirming official source terms in the HZ template, rerun:",
            "",
            "`.venv-pytorch/bin/python run_stage42_hz_to_cg_intake_bridge.py`",
            "",
            "Then either explicitly select the bridged intake in a future guarded runner or copy reviewed confirmations into the canonical CG intake template. Do not convert/evaluate until validator, guarded conversion, no-leakage, and source-CV all pass.",
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage42-IA Gate",
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
        "## Stage42-IA HZ to CG Intake Bridge",
        "",
        f"- source: `{payload['source']}`",
        "- role: bridge the new HZ confirmation packet into the older CG validator intake schema without activating conversion.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- mapped rows: `{payload['summary']['bridge_rows']}`; ready if activated now: `{payload['summary']['ready_if_activated_now']}`.",
        "- Remaining blocker: HZ confirmation fields are blank and require user-confirmed terms/source identity before guarded conversion.",
        "- Claim boundary: no download, no conversion, no training, no evaluation, no metric/seconds-level claim, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, bool]:
    lines = _refresh_lines(payload, gate)
    readme_paths = [README_RESULTS, M3W_README, MASTER_SUMMARY, ROUTES_SUMMARY]
    for path in readme_paths:
        _replace_section(path, SECTION, lines)
    matrix_lines = [
        "## Stage42-IA HZ to CG Intake Bridge",
        "",
        "- IA maps HZ confirmation rows into a CG-compatible inactive intake candidate.",
        f"- gate: `{gate['passed']} / {gate['total']}`.",
        f"- ready if activated now: `{payload['summary']['ready_if_activated_now']}`.",
        "- No conversion/evaluation occurred; source/legal blockers remain until user confirmation and guarded conversion pass.",
    ]
    _replace_section(PAPER_MATRIX, SECTION, matrix_lines)
    return {
        "readmes_updated": all(SECTION in path.read_text(encoding="utf-8") for path in readme_paths),
        "paper_matrix_updated": SECTION in PAPER_MATRIX.read_text(encoding="utf-8"),
    }


def _refresh_state(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-IA HZ to CG intake bridge"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_ia_hz_to_cg_intake_bridge"] = {
        "source": payload["source"],
        "report": str(BRIDGE_MD),
        "json": str(BRIDGE_JSON),
        "bridged_template": str(BRIDGED_TEMPLATE_JSON),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "gates": f"{gate['passed']}/{gate['total']}",
        "verdict": gate["verdict"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [BRIDGE_MD, BRIDGE_JSON, BRIDGED_TEMPLATE_JSON, GATE_MD, USER_ACTION_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, state)


def run_stage42_hz_to_cg_intake_bridge() -> dict[str, Any]:
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
    payload["stage42_ia_gate"] = gate
    write_json(BRIDGE_JSON, payload)
    _refresh_state(payload, gate)
    return payload


if __name__ == "__main__":
    result = run_stage42_hz_to_cg_intake_bridge()
    gate = result["stage42_ia_gate"]
    print(f"Stage42-IA HZ to CG intake bridge: {gate['verdict']} ({gate['passed']}/{gate['total']})")
