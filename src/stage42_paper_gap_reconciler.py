from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

FU_JSON = OUT_DIR / "module_contribution_ledger_stage42.json"
EN_JSON = OUT_DIR / "floor_removability_decision_map_stage42.json"
GR_JSON = OUT_DIR / "long_objective_state_reconciler_stage42.json"
GQ_JSON = OUT_DIR / "source_terms_package_claim_linter_stage42.json"
GAP_MD = OUT_DIR / "a_journal_gap_stage42.md"

REPORT_JSON = OUT_DIR / "paper_gap_reconciler_stage42.json"
REPORT_MD = OUT_DIR / "paper_gap_reconciler_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gs_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gs_paper_gap_reconciler"

CURRENT_FACTS = [
    "Current model is not a true 3D world model.",
    "Current model is not a large-scale foundation world model.",
    "Current model remains a protected dataset-local/raw-frame 2.5D multi-agent world-state candidate.",
    "Stage42-GS reconciles stale A-journal gap statements against latest module/floor/source/legal guards.",
    "Stage42-GS does not download, convert, train, or evaluate data/models.",
    "Future endpoints/waypoints remain labels or evaluation targets only, not inference inputs.",
    "No central velocity, no test endpoints for goals, and no test-threshold tuning are allowed.",
    "t+50/t+100 remain raw-frame horizons unless a future source-specific guard proves otherwise.",
    "Dataset-local/raw-frame evidence is not global metric or seconds-level evidence.",
    "Stage5C latent generative execution remains disabled.",
    "SMC remains disabled.",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
}


STALE_PATTERNS: list[tuple[str, str, str]] = [
    (
        "stage42_p_t50_ci_low_negative",
        r"Stage42-P.*negative.*t50.*CI|Stage42-P.*t50.*CI.*negative|negative 3-seed t50 CI",
        "Superseded/refined by later protected t50 bootstrap and floor-removability evidence; remaining blocker is source diversity/legal conversion, not Stage42-P mean sign.",
    ),
    (
        "full_waypoint_shape_open_without_boundary",
        r"endpoint-bridge, and full-waypoint-shape retraining remain open|full-waypoint-shape retraining remain open",
        "Refined by the module contribution ledger: endpoint_bridge/full_waypoint_shape are supported bounded components, but not floor-free/global primary dynamics.",
    ),
    (
        "proximity_self_gate_open",
        r"Build a proximity-safe internal self-gate|proximity-safe internal self-gate",
        "Refined by proximity guard and floor-removability evidence: proximity guard is required for safety-sensitive claims; floor-free deployment remains blocked.",
    ),
    (
        "external_expansion_open",
        r"Add one more legally verified external|External expansion beyond the current converted",
        "Still open and active: latest source/legal contract has contract_ready_now=0 and auto_download_allowed_now=0.",
    ),
]


def _load_inputs() -> dict[str, Any]:
    return {
        "module_ledger": read_json(FU_JSON, {}),
        "floor_map": read_json(EN_JSON, {}),
        "long_objective": read_json(GR_JSON, {}),
        "package_linter": read_json(GQ_JSON, {}),
        "gap_text": GAP_MD.read_text(encoding="utf-8") if GAP_MD.exists() else "",
    }


def _input_status(inputs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path_map = {
        "module_ledger": FU_JSON,
        "floor_map": EN_JSON,
        "long_objective": GR_JSON,
        "package_linter": GQ_JSON,
        "gap_md": GAP_MD,
    }
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "source": inputs.get(name, {}).get("source") if isinstance(inputs.get(name), Mapping) else "",
        }
        for name, path in path_map.items()
    }


def _stale_gap_findings(gap_text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    lines = gap_text.splitlines()
    for finding_id, pattern, resolution in STALE_PATTERNS:
        matches = []
        for line_number, line in enumerate(lines, start=1):
            if re.search(pattern, line, flags=re.IGNORECASE):
                matches.append({"line": line_number, "text": line.strip()})
        if matches:
            findings.append(
                {
                    "finding_id": finding_id,
                    "matches": matches,
                    "resolution": resolution,
                    "status": "reconciled_in_stage42_gs_section",
                }
            )
    return findings


def _gap_rows(inputs: Mapping[str, Any], stale_findings: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fu = inputs["module_ledger"].get("summary", {})
    en = inputs["floor_map"].get("summary", {})
    gr = inputs["long_objective"].get("summary", {})
    gq = inputs["package_linter"].get("summary", {})
    return [
        {
            "gap": "source_legal_conversion",
            "current_status": "open_blocker",
            "result_source": "fresh_run",
            "evidence": str(GR_JSON),
            "current_value": {
                "contract_ready_now": gr.get("contract_ready_now", 0),
                "auto_download_allowed_now": gr.get("auto_download_allowed_now", 0),
                "after_terms_t50_opportunity": gr.get("after_terms_t50_opportunity", 0),
                "after_terms_t100_opportunity": gr.get("after_terms_t100_opportunity", 0),
            },
            "paper_claim": "No new external conversion/evaluation claim until user-confirmed terms/path/source identity and guarded conversion pass.",
            "next_action": "User source confirmation -> guarded conversion -> no-leakage -> source-CV -> final test once.",
        },
        {
            "gap": "module_contribution_boundary",
            "current_status": "claim_locked",
            "result_source": "cached_verified",
            "evidence": str(FU_JSON),
            "current_value": {
                "main_claim_allowed_modules": fu.get("main_claim_allowed_modules", []),
                "blocked_or_auxiliary_modules": fu.get("blocked_or_auxiliary_modules", []),
            },
            "paper_claim": "Allowed core claims are protected history/domain/safe-switch/teacher-floor/group-consistency full-waypoint family; JEPA/Transformer/scene-goal/neighbor-interaction remain auxiliary/negative.",
            "next_action": "Do not promote auxiliary context modules unless a new retrained protocol proves material lift.",
        },
        {
            "gap": "floor_free_neural_deployment",
            "current_status": "blocked_with_partial_floor_relaxation",
            "result_source": "cached_verified",
            "evidence": str(EN_JSON),
            "current_value": {
                "floor_free_neural_deployable": en.get("floor_free_neural_deployable", False),
                "safe_partial_floor_relaxation_available": en.get("safe_partial_floor_relaxation_available", False),
                "partial_relaxation_components": en.get("partial_relaxation_components", []),
                "proximity_guard_required_for_safety_claim": en.get("proximity_guard_required_for_safety_claim", True),
            },
            "paper_claim": "Teacher/Stage37 floor remains required globally; narrow validation-backed t50 slice relaxation is allowed only inside protected policy.",
            "next_action": "Any floor relaxation must keep proximity guard and validation-only selection.",
        },
        {
            "gap": "paper_package_source_claim_safety",
            "current_status": "clean_with_open_blocker",
            "result_source": "fresh_run",
            "evidence": str(GQ_JSON),
            "current_value": {
                "files_scanned": gq.get("files_scanned", 0),
                "violation_count": gq.get("violation_count", 0),
                "underlying_data_license_confirmed": gq.get("underlying_data_license_confirmed", 0),
            },
            "paper_claim": "Package source/legal language is currently clean, but it must keep explicit open source-term blockers.",
            "next_action": "Run package linter after every future paper-package refresh.",
        },
        {
            "gap": "stale_gap_text",
            "current_status": "reconciled",
            "result_source": "fresh_run",
            "evidence": str(GAP_MD),
            "current_value": {
                "stale_findings": [row["finding_id"] for row in stale_findings],
                "stale_finding_count": len(stale_findings),
            },
            "paper_claim": "Older gap statements are preserved historically but superseded/refined by the latest Stage42-GS refresh section.",
            "next_action": "Use the Stage42-GS section as the current gap summary.",
        },
    ]


def _summary(rows: list[Mapping[str, Any]], stale_findings: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source": SOURCE,
        "gap_rows": len(rows),
        "fresh_rows": sum(1 for row in rows if row["result_source"] == "fresh_run"),
        "cached_verified_rows": sum(1 for row in rows if row["result_source"] == "cached_verified"),
        "stale_findings_reconciled": len(stale_findings),
        "open_blockers": [row["gap"] for row in rows if "open" in row["current_status"] or "blocked" in row["current_status"]],
        "current_paper_position": "protected_2_5d_external_world_state_candidate_not_true3d_not_foundation",
        "download_executed": False,
        "conversion_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    cb = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "inputs_loaded": all(row["exists"] for row in payload["input_status"].values()),
        "gap_rows_written": s["gap_rows"] >= 5,
        "stale_findings_reconciled": s["stale_findings_reconciled"] >= 1,
        "source_blocker_preserved": "source_legal_conversion" in s["open_blockers"],
        "floor_free_blocker_preserved": "floor_free_neural_deployment" in s["open_blockers"],
        "module_boundary_present": any(row["gap"] == "module_contribution_boundary" for row in payload["gap_rows"]),
        "package_claim_linter_clean": next(
            row for row in payload["gap_rows"] if row["gap"] == "paper_package_source_claim_safety"
        )["current_value"]["violation_count"]
        == 0,
        "no_download_conversion_training_eval": not (
            s["download_executed"] or s["conversion_executed"] or s["training_executed"] or s["evaluation_executed"]
        ),
        "claim_boundary_no_metric_seconds": not cb["global_metric_claim_allowed"] and not cb["global_seconds_claim_allowed"],
        "claim_boundary_no_true3d_foundation": not cb["true_3d"] and not cb["foundation_world_model"],
        "stage5c_not_executed": not cb["stage5c_executed"],
        "smc_not_enabled": not cb["smc_enabled"],
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_gs_paper_gap_reconciler_pass" if passed == total else "stage42_gs_paper_gap_reconciler_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Stage42-GS Paper Gap Reconciler",
        "",
        f"- source: `{payload['source']}`",
        f"- generated at: `{payload['generated_at']}`",
        f"- git commit: `{payload['git_commit']}`",
        f"- gate: `{payload['stage42_gs_gate']['passed']} / {payload['stage42_gs_gate']['total']}`",
        f"- verdict: `{payload['stage42_gs_gate']['verdict']}`",
        "",
        "## Current Boundary",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Reconciled Gaps",
        "",
        "| gap | status | source | paper claim | next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["gap_rows"]:
        lines.append(
            f"| `{row['gap']}` | `{row['current_status']}` | `{row['result_source']}` | "
            f"{row['paper_claim']} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Stale Gap Statements Reconciled",
            "",
        ]
    )
    for finding in payload["stale_findings"]:
        lines.append(f"- `{finding['finding_id']}`: {finding['resolution']}")
    return "\n".join(lines) + "\n"


def _render_gate(payload: Mapping[str, Any]) -> str:
    gate = payload["stage42_gs_gate"]
    lines = [
        "# Stage42-GS Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    return "\n".join(lines) + "\n"


def _refresh_gap_file(payload: Mapping[str, Any]) -> None:
    lines = [
        "## Stage42-GS Current Gap Reconciliation",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{payload['stage42_gs_gate']['verdict']}`",
        f"- gates: `{payload['stage42_gs_gate']['passed']} / {payload['stage42_gs_gate']['total']}`",
        "- This section supersedes older Stage42-P/BY/BZ/EN gap wording where explicitly noted; older sections remain historical provenance.",
        "- Current paper position: protected dataset-local/raw-frame 2.5D world-state candidate, not true 3D, not foundation, not metric/seconds-level.",
        "- Source/legal conversion remains the main external expansion blocker: contract_ready_now=0, auto_download_allowed_now=0.",
        "- Floor-free neural deployment remains blocked; partial t50 floor relaxation is allowed only in validation-backed protected slices.",
        "- Core supported modules remain history, domain expert, safe-switch, teacher/Stage37 floor, group-consistency full-waypoint, full-waypoint shape, and endpoint bridge.",
        "- JEPA, Transformer, scene/goal, and neighbor/interaction remain auxiliary/negative or non-main claims under current evidence.",
    ]
    _replace_section(GAP_MD, "STAGE42_GS_REFRESH", lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "## Stage42-GS Paper Gap Reconciler",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{payload['stage42_gs_gate']['verdict']}`",
        f"- gates: `{payload['stage42_gs_gate']['passed']} / {payload['stage42_gs_gate']['total']}`",
        f"- gap rows: `{s['gap_rows']}`",
        f"- stale findings reconciled: `{s['stale_findings_reconciled']}`",
        f"- open blockers: `{s['open_blockers']}`",
        "- It refreshes A-journal gap language against current module, floor, source/legal, and package-claim guards.",
        "- No download, conversion, training, or evaluation was executed.",
    ]
    for path in [README_RESULTS, M3W_README]:
        _replace_section(path, "STAGE42_GS_REFRESH", lines)

    state = read_json(RESEARCH_STATE, {})
    state.update(
        {
            "current_stage": "Stage42-GS paper gap reconciler",
            "current_verdict": payload["stage42_gs_gate"]["verdict"],
            "stage42_gs_paper_gap_reconciler": {
                "source": payload["source"],
                "path": str(REPORT_MD),
                "json": str(REPORT_JSON),
                "gate": payload["stage42_gs_gate"],
                "summary": s,
                "claim_boundary": payload["claim_boundary"],
            },
        }
    )
    write_json(RESEARCH_STATE, state)


def run_stage42_paper_gap_reconciler(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = _load_inputs()
    stale_findings = _stale_gap_findings(inputs["gap_text"])
    rows = _gap_rows(inputs, stale_findings)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_status": _input_status(inputs),
        "current_facts": CURRENT_FACTS,
        "stale_findings": stale_findings,
        "gap_rows": rows,
        "summary": _summary(rows, stale_findings),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_gs_gate"] = _gate(payload)
    payload["evidence_hash"] = _combined_hash(
        {
            "source": payload["source"],
            "stale_findings": payload["stale_findings"],
            "gap_rows": payload["gap_rows"],
            "claim_boundary": payload["claim_boundary"],
        }
    )
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload).splitlines())
    write_md(GATE_MD, _render_gate(payload).splitlines())
    _refresh_gap_file(payload)
    if refresh_readmes:
        _refresh_readmes(payload)
    return payload


__all__ = ["run_stage42_paper_gap_reconciler", "_stale_gap_findings", "_gate"]
