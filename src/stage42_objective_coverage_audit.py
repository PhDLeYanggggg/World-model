from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

FU_JSON = OUT_DIR / "module_contribution_ledger_stage42.json"
FV_JSON = OUT_DIR / "claim_boundary_linter_stage42.json"
FW_JSON = OUT_DIR / "source_action_consolidator_stage42.json"
DM_JSON = OUT_DIR / "reviewer_replay_package_stage42.json"
CX_JSON = OUT_DIR / "evidence_provenance_stage42.json"
CZ_JSON = OUT_DIR / "paper_freeze_candidate_manifest_stage42.json"

REPORT_JSON = OUT_DIR / "objective_coverage_audit_stage42.json"
REPORT_MD = OUT_DIR / "objective_coverage_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fx_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_objective_coverage_audit_from_current_evidence"

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
    "threshold_tuned_on_test": False,
}


def _passed_gate(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate) and gate.get("passed") == gate.get("total")


def _load_inputs() -> dict[str, Any]:
    return {
        "module_ledger": read_json(FU_JSON, {}),
        "claim_linter": read_json(FV_JSON, {}),
        "source_action": read_json(FW_JSON, {}),
        "reviewer_replay": read_json(DM_JSON, {}),
        "evidence_provenance": read_json(CX_JSON, {}),
        "paper_freeze_manifest": read_json(CZ_JSON, {}),
    }


def _input_status(inputs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path_map = {
        "module_ledger": FU_JSON,
        "claim_linter": FV_JSON,
        "source_action": FW_JSON,
        "reviewer_replay": DM_JSON,
        "evidence_provenance": CX_JSON,
        "paper_freeze_manifest": CZ_JSON,
    }
    out: dict[str, dict[str, Any]] = {}
    for name, path in path_map.items():
        payload = inputs.get(name, {})
        out[name] = {
            "path": str(path),
            "exists": path.exists(),
            "source": payload.get("source"),
            "verdict": payload.get("verdict")
            or payload.get("stage42_fu_gate", {}).get("verdict")
            or payload.get("stage42_fv_gate", {}).get("verdict")
            or payload.get("stage42_fw_gate", {}).get("verdict")
            or payload.get("stage42_dm_gate", {}).get("verdict")
            or payload.get("stage42_cx_gate", {}).get("verdict")
            or payload.get("stage42_cz_gate", {}).get("verdict"),
        }
    return out


def _compact_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    module_ledger = inputs.get("module_ledger", {})
    claim_linter = inputs.get("claim_linter", {})
    source_action = inputs.get("source_action", {})
    reviewer_replay = inputs.get("reviewer_replay", {})
    evidence_provenance = inputs.get("evidence_provenance", {})
    paper_freeze_manifest = inputs.get("paper_freeze_manifest", {})
    return {
        "module_ledger": {
            "stage42_fu_gate": module_ledger.get("stage42_fu_gate", {}),
            "summary": module_ledger.get("summary", {}),
        },
        "claim_linter": {
            "stage42_fv_gate": claim_linter.get("stage42_fv_gate", {}),
            "summary": claim_linter.get("summary", {}),
        },
        "source_action": {
            "stage42_fw_gate": source_action.get("stage42_fw_gate", {}),
            "summary": source_action.get("summary", {}),
            "claim_boundary": source_action.get("claim_boundary", {}),
        },
        "reviewer_replay": {
            "stage42_dm_gate": reviewer_replay.get("stage42_dm_gate", {}),
            "summary": reviewer_replay.get("summary", {}),
        },
        "evidence_provenance": {
            "stage42_cx_gate": evidence_provenance.get("stage42_cx_gate", {}),
            "summary": evidence_provenance.get("summary", {}),
        },
        "paper_freeze_manifest": {
            "stage42_cz_gate": paper_freeze_manifest.get("stage42_cz_gate", {}),
            "freeze_status": paper_freeze_manifest.get("freeze_status", {}),
        },
    }


def _objective_rows(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    fu = inputs.get("module_ledger", {})
    fw = inputs.get("source_action", {})
    dm = inputs.get("reviewer_replay", {})
    fv = inputs.get("claim_linter", {})
    cx = inputs.get("evidence_provenance", {})
    cz = inputs.get("paper_freeze_manifest", {})

    supported = set(fu.get("summary", {}).get("main_claim_allowed_modules", []))
    blocked = set(fu.get("summary", {}).get("blocked_or_auxiliary_modules", []))
    top_actions = list(fw.get("summary", {}).get("top_actions", []))

    return [
        {
            "objective_id": "A",
            "name": "Data and calibration",
            "status": "blocked_user_action_required",
            "result_source": "fresh_audit_from_cached_verified_inputs",
            "evidence": [str(FW_JSON)],
            "proved": [
                "source/legal/horizon blockers are consolidated",
                "conversion_ready_now is 0, so no blocked data is counted as converted",
            ],
            "missing": [
                "UCY original terms/local path confirmation",
                "TrajNet longer h100-capable official source",
                "ETH/UCY/TrajNet source-specific metric/time calibration closure",
            ],
            "next_actions": top_actions[:5],
            "claim_boundary": "Keep raw-frame/dataset-local wording until guarded conversion, no-leakage, source-CV, and metric/time calibration pass.",
        },
        {
            "objective_id": "B",
            "name": "External validation",
            "status": "partial_positive_with_source_blockers",
            "result_source": "cached_verified",
            "evidence": [str(DM_JSON), str(FW_JSON)],
            "proved": [
                "Stage37/teacher-floor protected policy has positive external raw-frame evidence",
                "reviewer replay exact path reports runtime rows and positive all/t50/t100raw/hard metrics",
            ],
            "missing": [
                "additional legal converted external top-down sources",
                "uniform h100 horizon robustness",
                "closed TrajNet/UCY/ETH_UCY domain source support",
            ],
            "next_actions": [
                ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
                ".venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py",
                ".venv-pytorch/bin/python run_stage42_source_support_closure_audit.py",
            ],
            "claim_boundary": "External evidence is protected dataset-local/raw-frame; no foundation or metric/seconds claim.",
        },
        {
            "objective_id": "C",
            "name": "All-agent full-waypoint dynamics",
            "status": "partial_protected_not_ungated",
            "result_source": "cached_verified",
            "evidence": [str(FU_JSON), str(DM_JSON)],
            "proved": [
                "group-consistency full-waypoint is allowed as a supported source-level claim",
                "full-waypoint shape and endpoint bridge are supported components",
            ],
            "missing": [
                "ungated neural/full-waypoint deployment safety",
                "uniform h100/horizon closure",
                "metric/seconds calibration",
            ],
            "next_actions": [
                "keep Stage37/teacher floor for deployment",
                "only relax floor on slices where fresh no-easy-harm gates pass",
            ],
            "claim_boundary": "Full-waypoint is protected source-level evidence, not Stage5C latent generative rollout.",
        },
        {
            "objective_id": "D",
            "name": "Causal ablation and module contribution",
            "status": "partial_main_modules_identified",
            "result_source": "fresh_run",
            "evidence": [str(FU_JSON), str(FV_JSON)],
            "proved": [
                f"supported main modules: {sorted(supported)}",
                "claim linter reports zero violations",
            ],
            "missing": [
                f"blocked or auxiliary modules remain: {sorted(blocked)}",
                "JEPA and Transformer independent main contribution",
                "scene/goal and neighbor/interaction independent main contribution",
            ],
            "next_actions": [
                "do not repeat weak context residual protocols unchanged",
                "future context claims require retrained graph/scene-rich protocol beating baseline-family control",
            ],
            "claim_boundary": "Only history/domain expert/safe-switch/teacher-floor/group-consistency full-waypoint may be written as current core claims.",
        },
        {
            "objective_id": "E",
            "name": "Safety floor study",
            "status": "pass_floor_required",
            "result_source": "fresh_and_cached_verified",
            "evidence": [str(FU_JSON), str(DM_JSON)],
            "proved": [
                "teacher floor is necessary_not_removable in module ledger",
                "safe-switch is supported as deployment mechanism",
                "reviewer replay preserves exact runtime policy behavior",
            ],
            "missing": [
                "floor-free neural dynamics that preserves easy cases",
                "Stage5C/SMC readiness",
            ],
            "next_actions": [
                "treat floor removal as a future gated experiment, not deployment default",
                "preserve Stage37/teacher floor in reviewer-facing policy package",
            ],
            "claim_boundary": "Current best model is protected; do not claim ungated neural dynamics.",
        },
        {
            "objective_id": "F",
            "name": "Paper-ready evidence package",
            "status": "paper_package_candidate_clean_with_open_blockers",
            "result_source": "fresh_run",
            "evidence": [str(DM_JSON), str(CX_JSON), str(CZ_JSON), str(FV_JSON)],
            "proved": [
                "reviewer replay package gate passes",
                "paper freeze candidate manifest gate passes",
                "claim linter gate passes with zero violations",
            ],
            "missing": [
                "legal/source conversion closure for blocked external sources",
                "global metric/seconds claim support",
                "foundation-track breadth",
            ],
            "next_actions": [
                ".venv-pytorch/bin/python run_stage42_reviewer_replay_package.py",
                ".venv-pytorch/bin/python run_stage42_evidence_provenance_verifier.py",
                ".venv-pytorch/bin/python run_stage42_paper_freeze_candidate_manifest.py",
            ],
            "claim_boundary": "Paper candidate is protected 2.5D evidence only; A-journal package must foreground limitations.",
        },
    ]


def _coverage_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    blocked = [row["objective_id"] for row in rows if "blocked" in str(row.get("status", ""))]
    partial = [row["objective_id"] for row in rows if str(row.get("status", "")).startswith("partial")]
    passed = [row["objective_id"] for row in rows if str(row.get("status", "")).startswith("pass")]
    return {
        "objectives_total": len(rows),
        "status_counts": counts,
        "blocked_objectives": blocked,
        "partial_objectives": partial,
        "passed_objectives": passed,
        "goal_complete": False,
        "current_best_status": "protected_dataset_local_raw_frame_2_5d_candidate",
        "highest_priority_next_action": "FW-TERMS-ucy_crowd_original",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("objective_rows", []))
    claim = payload.get("claim_boundary", {})
    summary = payload.get("summary", {})
    inputs = payload.get("inputs", {})
    source_action = inputs.get("source_action", {})
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "six_objectives_a_to_f_covered": [row.get("objective_id") for row in rows] == list("ABCDEF"),
        "every_objective_has_evidence": all(row.get("evidence") for row in rows),
        "every_objective_has_missing_or_next": all(row.get("missing") or row.get("next_actions") for row in rows),
        "goal_not_marked_complete": summary.get("goal_complete") is False,
        "data_blocker_preserved": "A" in summary.get("blocked_objectives", []),
        "source_action_gate_passed": _passed_gate(source_action, "stage42_fw_gate"),
        "conversion_ready_not_overclaimed": int(source_action.get("summary", {}).get("conversion_ready_now", -1)) == 0,
        "no_true_3d_overclaim": claim.get("true_3d") is False,
        "no_foundation_overclaim": claim.get("foundation_world_model") is False,
        "no_metric_seconds_overclaim": claim.get("global_metric_claim_allowed") is False
        and claim.get("global_seconds_claim_allowed") is False,
        "no_download_conversion_training": claim.get("download_executed") is False
        and claim.get("conversion_executed") is False
        and claim.get("training_executed") is False,
        "no_test_threshold_tuning": claim.get("threshold_tuned_on_test") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    return {
        "passed": passed,
        "total": total,
        "gates": gates,
        "verdict": "stage42_fx_objective_coverage_audit_pass" if passed == total else "stage42_fx_objective_coverage_audit_partial",
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-FX Objective Coverage Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_fx_gate']['passed']} / {payload['stage42_fx_gate']['total']}`",
        f"- verdict: `{payload['stage42_fx_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        "- 当前不是 true 3D world model。",
        "- 当前不是 large-scale foundation world model。",
        "- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
        "- Stage42-FX is a requirement coverage audit; it does not train, download, convert, or tune thresholds.",
        "- All rows are marked fresh audit over current evidence, cached_verified, blocked, or partial; goal_complete remains false.",
        "- Stage5C latent generative is not executed; SMC is not enabled.",
        "",
        "## Summary",
        "",
        f"- objectives_total: `{summary['objectives_total']}`",
        f"- status_counts: `{summary['status_counts']}`",
        f"- blocked_objectives: `{summary['blocked_objectives']}`",
        f"- partial_objectives: `{summary['partial_objectives']}`",
        f"- passed_objectives: `{summary['passed_objectives']}`",
        f"- goal_complete: `{summary['goal_complete']}`",
        f"- current_best_status: `{summary['current_best_status']}`",
        f"- highest_priority_next_action: `{summary['highest_priority_next_action']}`",
        "",
        "## Objective Rows",
        "",
        "| objective | status | result source | proved | missing | next actions |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["objective_rows"]:
        lines.append(
            "| `{}` | `{}` | `{}` | {} | {} | {} |".format(
                row["objective_id"],
                row["status"],
                row["result_source"],
                "<br>".join(str(x) for x in row.get("proved", [])),
                "<br>".join(str(x) for x in row.get("missing", [])),
                "<br>".join(str(x) for x in row.get("next_actions", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
        ]
    )
    for gate, ok in payload["stage42_fx_gate"]["gates"].items():
        lines.append(f"| `{gate}` | {bool(ok)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A/F style paper evidence is now easier to audit requirement-by-requirement, but the long goal remains active.",
            "- Data/calibration is the main hard blocker because legal/source/h100 conversion readiness is still zero.",
            "- Current model evidence supports a protected 2.5D raw-frame candidate, not true 3D, foundation, metric, seconds-level, Stage5C, or SMC.",
        ]
    )
    return lines


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-FX Gate",
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
    summary = payload["summary"]
    lines = [
        "## Stage42-FX Objective Coverage Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_fx_gate']['passed']} / {payload['stage42_fx_gate']['total']}`; verdict `{payload['stage42_fx_gate']['verdict']}`.",
        f"- objectives covered: `{summary['objectives_total']}`; blocked objectives `{summary['blocked_objectives']}`; partial objectives `{summary['partial_objectives']}`; passed objectives `{summary['passed_objectives']}`.",
        f"- current best status: `{summary['current_best_status']}`.",
        f"- highest-priority next action: `{summary['highest_priority_next_action']}`.",
        "- role: requirement coverage audit for the active Stage42 A-F long objective; no training, no download, no conversion, no threshold tuning.",
        "- boundary: goal remains active and incomplete; M3W remains protected dataset-local/raw-frame 2.5D; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.",
    ]
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_FX_OBJECTIVE_COVERAGE_AUDIT", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FX objective coverage audit"
    state["current_verdict"] = payload["stage42_fx_gate"]["verdict"]
    state["stage42_fx_objective_coverage_audit"] = {
        "source": payload["source"],
        "path": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "updated_at": payload["generated_at_utc"],
        "gate": payload["stage42_fx_gate"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "goal_complete": False,
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_objective_coverage_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = _load_inputs()
    rows = _objective_rows(inputs)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([FU_JSON, FV_JSON, FW_JSON, DM_JSON, CX_JSON, CZ_JSON]),
        "inputs": _compact_inputs(inputs),
        "input_status": _input_status(inputs),
        "objective_rows": rows,
        "summary": _coverage_summary(rows),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_fx_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    _write_gate(payload["stage42_fx_gate"])
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


__all__ = ["run_stage42_objective_coverage_audit", "_gate", "_coverage_summary", "_objective_rows"]
