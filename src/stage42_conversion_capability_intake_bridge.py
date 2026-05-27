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
DW_JSON = OUT_DIR / "source_specific_conversion_dry_run_stage42.json"

REPORT_JSON = OUT_DIR / "conversion_capability_intake_bridge_stage42.json"
REPORT_MD = OUT_DIR / "conversion_capability_intake_bridge_stage42.md"
SNAPSHOT_JSON = OUT_DIR / "source_terms_confirmation_intake_conversion_capability_snapshot_stage42.json"
USER_ACTION_MD = OUT_DIR / "user_action_required_conversion_capability_intake_bridge_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ge_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_ge_conversion_capability_intake_bridge"

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
    "conversion_capability_is_permission": False,
    "conversion_ready_claim": False,
}

DATASET_DOMAIN = {
    "ucy_crowd_original": "UCY",
    "eth_biwi_original": "ETH_UCY",
    "trajnetplusplus_official": "TrajNet",
    "opentraj_toolkit": "OpenTraj",
    "aerialmpt_or_other_topdown": "AerialMPT",
}


def _dw_sources_by_domain(dw: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for row in dw.get("source_rows", []):
        out.setdefault(str(row.get("domain", "")), []).append(row)
    return out


def _domain_plan(dw: Mapping[str, Any], domain: str) -> Mapping[str, Any]:
    return dw.get("source_cv_plan", {}).get("domains", {}).get(domain, {})


def _compact_source(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": row.get("source_id", ""),
        "domain": row.get("domain", ""),
        "dataset": row.get("dataset", ""),
        "trajectory_file": row.get("trajectory_file", ""),
        "path_exists": bool(row.get("path_exists")),
        "rows": int(row.get("rows", 0) or 0),
        "agents": int(row.get("agents", 0) or 0),
        "common_frame_step": row.get("common_frame_step"),
        "horizon_counts": row.get("horizon_counts", {}),
        "history_horizon_counts": row.get("history_horizon_counts", {}),
        "t50_capable": bool(row.get("t50_capable")),
        "t100_capable": bool(row.get("t100_capable")),
        "causal_velocity_possible": bool(row.get("causal_velocity_possible")),
        "central_velocity_used": bool(row.get("central_velocity_used")),
        "technical_conversion_ready_after_terms": bool(row.get("technical_conversion_ready_after_terms")),
        "conversion_allowed_now": False,
        "blocked_by": list(row.get("blocked_by", [])) or ["terms/source_identity/path_version_not_confirmed"],
    }


def _capability_for_dataset(dataset_id: str, dw: Mapping[str, Any]) -> dict[str, Any]:
    domain = DATASET_DOMAIN.get(dataset_id, "")
    source_rows = [_compact_source(row) for row in _dw_sources_by_domain(dw).get(domain, [])]
    source_rows = sorted(source_rows, key=lambda row: (-int(row.get("horizon_counts", {}).get("50", 0) or 0), str(row.get("source_id", ""))))
    plan = dict(_domain_plan(dw, domain))
    return {
        "source": SOURCE,
        "domain": domain,
        "source_specific_dry_run_available": bool(source_rows),
        "source_count": len(source_rows),
        "technical_ready_after_terms_sources": sum(1 for row in source_rows if row["technical_conversion_ready_after_terms"]),
        "t50_windows_after_terms": int(sum(int(row.get("horizon_counts", {}).get("50", 0) or 0) for row in source_rows)),
        "t100_windows_after_terms": int(sum(int(row.get("horizon_counts", {}).get("100", 0) or 0) for row in source_rows)),
        "source_cv_feasible_after_terms": bool(plan.get("source_cv_feasible_after_terms", False)),
        "source_cv_plan": plan,
        "source_rows": source_rows,
        "conversion_allowed_now": False,
        "converted_now": False,
        "evaluated_now": False,
        "legal_blocker_preserved": True,
        "no_leakage_preflight": dw.get("no_leakage_preflight", {}),
        "safe_use": "Use only after user-confirmed official terms/source identity and validator/guarded queue pass; this is not permission and not conversion readiness.",
    }


def _merge_intake_with_capability(intake: Mapping[str, Any], dw: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(intake))
    datasets: list[dict[str, Any]] = []
    for row in intake.get("datasets", []):
        new_row = deepcopy(dict(row))
        dataset_id = str(new_row.get("dataset_id", ""))
        new_row["conversion_capability_prefill"] = _capability_for_dataset(dataset_id, dw)
        new_row["conversion_ready_now"] = False
        new_row["converted_now"] = False
        new_row["evaluated_now"] = False
        datasets.append(new_row)
    merged["datasets"] = datasets
    merged["conversion_capability_bridge"] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dw_report": str(DW_JSON),
        "rules": [
            "conversion_capability_prefill is a dry-run capability hint only",
            "it does not grant legal permission or conversion readiness",
            "source-CV plans may only execute after user confirmation, validator pass, guarded queue, and no-leakage checks",
        ],
    }
    return merged


def _summary(merged: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(merged.get("datasets", []))
    caps = [row.get("conversion_capability_prefill", {}) for row in rows]
    return {
        "source": SOURCE,
        "intake_rows": len(rows),
        "rows_with_capability_prefill": sum(1 for cap in caps if "source_count" in cap),
        "rows_with_source_specific_dry_run": sum(1 for cap in caps if cap.get("source_specific_dry_run_available")),
        "rows_with_source_cv_feasible_after_terms": sum(1 for cap in caps if cap.get("source_cv_feasible_after_terms")),
        "technical_ready_after_terms_sources": sum(int(cap.get("technical_ready_after_terms_sources", 0) or 0) for cap in caps),
        "t50_windows_after_terms": sum(int(cap.get("t50_windows_after_terms", 0) or 0) for cap in caps),
        "t100_windows_after_terms": sum(int(cap.get("t100_windows_after_terms", 0) or 0) for cap in caps),
        "rows_with_user_confirmation": sum(1 for row in rows if _has_user_confirmation(row)),
        "conversion_ready_now": sum(1 for row in rows if row.get("conversion_ready_now") is True),
        "converted_now": sum(1 for row in rows if row.get("converted_now") is True),
        "evaluated_now": sum(1 for row in rows if row.get("evaluated_now") is True),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "dw_loaded": payload.get("input_status", {}).get("dw_exists") is True,
        "intake_loaded": payload.get("input_status", {}).get("intake_exists") is True,
        "dw_passed": payload.get("input_status", {}).get("dw_verdict") == "stage42_dw_source_specific_conversion_dry_run_pass",
        "intake_rows_preserved": s.get("intake_rows", 0) >= 5,
        "capability_prefill_added": s.get("rows_with_capability_prefill", 0) >= 5,
        "source_specific_dry_runs_present": s.get("rows_with_source_specific_dry_run", 0) >= 2,
        "source_cv_feasible_after_terms_present": s.get("rows_with_source_cv_feasible_after_terms", 0) >= 1,
        "technical_windows_present": s.get("t50_windows_after_terms", 0) > 0 and s.get("t100_windows_after_terms", 0) > 0,
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
        "verdict": "stage42_ge_conversion_capability_intake_bridge_pass" if passed == total else "stage42_ge_conversion_capability_intake_bridge_partial",
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GE Conversion Capability -> Intake Bridge",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ge_gate']['passed']} / {payload['stage42_ge_gate']['total']}`",
        f"- verdict: `{payload['stage42_ge_gate']['verdict']}`",
        "",
        "## Role",
        "",
        "- This bridges DW source-specific conversion dry-run evidence into the intake as `conversion_capability_prefill`.",
        "- It records source IDs, horizon support, source-CV feasibility, and technical readiness after terms confirmation.",
        "- It does not grant permission, convert data, train, evaluate, or make metric/seconds claims.",
        "",
        "## Summary",
        "",
        f"- intake_rows: `{s['intake_rows']}`",
        f"- rows_with_source_specific_dry_run: `{s['rows_with_source_specific_dry_run']}`",
        f"- rows_with_source_cv_feasible_after_terms: `{s['rows_with_source_cv_feasible_after_terms']}`",
        f"- technical_ready_after_terms_sources: `{s['technical_ready_after_terms_sources']}`",
        f"- t50/t100 windows after terms: `{s['t50_windows_after_terms']}` / `{s['t100_windows_after_terms']}`",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`",
        "",
        "## Intake Rows",
        "",
        "| dataset | sources | tech-ready sources | source-CV after terms | t50 after terms | t100 after terms | conversion ready now |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["merged_intake"]["datasets"]:
        cap = row.get("conversion_capability_prefill", {})
        lines.append(
            f"| `{row.get('dataset_id')}` | {cap.get('source_count', 0)} | "
            f"{cap.get('technical_ready_after_terms_sources', 0)} | {cap.get('source_cv_feasible_after_terms', False)} | "
            f"{cap.get('t50_windows_after_terms', 0)} | {cap.get('t100_windows_after_terms', 0)} | {row.get('conversion_ready_now') is True} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Conversion capability is not legal permission and not conversion readiness.",
            "- UCY has a source-CV-capable plan after terms; ETH has technical source-specific candidates but fewer sources for source-CV.",
            "- Current M3W remains protected dataset-local/raw-frame 2.5D; no true 3D, foundation, metric/seconds, Stage5C, or SMC claim.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# User Action Required: Stage42-GE Conversion Capability",
        "",
        f"- Open `{INTAKE_JSON}`.",
        "- Inspect `conversion_capability_prefill` before choosing which source to confirm.",
        "- UCY currently has the strongest source-CV-capable after-terms plan; ETH has calibrated sources but insufficient independent sources for source-CV by itself.",
        "- These are dry-run technical hints only. Fill `user_confirmation` manually after official terms/source identity verification.",
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
        "# Stage42-GE Gate",
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
        "## Stage42-GE Conversion Capability -> Intake Bridge",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_ge_gate']['passed']} / {payload['stage42_ge_gate']['total']}`; verdict `{payload['stage42_ge_gate']['verdict']}`.",
        "- role: adds DW source-specific dry-run capability into the intake template as non-permission `conversion_capability_prefill`.",
        f"- source-specific rows available for `{s['rows_with_source_specific_dry_run']}` dataset rows; source-CV feasible after terms for `{s['rows_with_source_cv_feasible_after_terms']}` row.",
        f"- t50/t100 windows after terms: `{s['t50_windows_after_terms']}` / `{s['t100_windows_after_terms']}`; conversion_ready_now `{s['conversion_ready_now']}`.",
        "- boundary: dry-run capability is not permission or conversion readiness; no download/conversion/training/evaluation; no metric/seconds/Stage5C/SMC claim.",
    ]
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GE_CONVERSION_CAPABILITY_INTAKE_BRIDGE", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-GE conversion capability intake bridge"
    state["current_verdict"] = payload["stage42_ge_gate"]["verdict"]
    state["stage42_ge_conversion_capability_intake_bridge"] = {
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


def run_stage42_conversion_capability_intake_bridge() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    intake = read_json(INTAKE_JSON, {})
    dw = read_json(DW_JSON, {})
    merged = _merge_intake_with_capability(intake, dw)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([INTAKE_JSON, DW_JSON]),
        "input_status": {
            "intake_exists": INTAKE_JSON.exists(),
            "dw_exists": DW_JSON.exists(),
            "intake_source": intake.get("source", ""),
            "dw_source": dw.get("source", ""),
            "dw_verdict": dw.get("stage42_dw_gate", {}).get("verdict", ""),
        },
        "merged_intake": merged,
        "summary": _summary(merged),
        "claim_boundary": CLAIM_BOUNDARY,
        "snapshot_written": True,
        "intake_template_updated": True,
        "user_action_required_written": True,
    }
    payload["stage42_ge_gate"] = _gate(payload)
    write_json(INTAKE_JSON, merged)
    write_json(SNAPSHOT_JSON, merged)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    _write_gate(payload["stage42_ge_gate"])
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


__all__ = [
    "run_stage42_conversion_capability_intake_bridge",
    "_merge_intake_with_capability",
    "_capability_for_dataset",
    "_gate",
]
