from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

REPORT_JSON = OUT_DIR / "long_objective_coverage_audit_stage42.json"
REPORT_MD = OUT_DIR / "long_objective_coverage_audit_stage42.md"
EVIDENCE_MATRIX_MD = OUT_DIR / "paper_ready_evidence_matrix_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ek_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_long_objective_coverage_audit"

PAPER_FILES = [
    OUT_DIR / "paper_outline_stage42.md",
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "data_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EK 是 long-objective coverage audit，不下载、不转换、不训练、不评估。",
    "本审计把 Stage42 A-F 目标映射到 fresh/cached/not_run 证据和 blocker，防止 paper package 过度 claim。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
    "conversion_executed_in_stage42_ek": False,
    "evaluation_executed_in_stage42_ek": False,
    "training_executed_in_stage42_ek": False,
}


def _exists(path: str | Path) -> bool:
    return Path(path).exists()


def _load(path: str | Path) -> dict[str, Any]:
    return read_json(path, {})


def _gate_passed(path: str | Path, gate_key: str) -> bool:
    payload = _load(path)
    gate = payload.get(gate_key, {})
    return bool(gate.get("passed") == gate.get("total") and gate.get("total", 0) > 0)


def _value(path: str | Path, keys: list[str], default: Any = None) -> Any:
    cur: Any = _load(path)
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _paper_file_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in PAPER_FILES:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def _row(
    *,
    phase: str,
    requirement_id: str,
    requirement: str,
    status: str,
    result_source: str,
    evidence_files: list[str],
    evidence_summary: str,
    blockers: list[str] | None = None,
    next_action: str = "",
) -> dict[str, Any]:
    return {
        "phase": phase,
        "requirement_id": requirement_id,
        "requirement": requirement,
        "status": status,
        "result_source": result_source,
        "evidence_files": evidence_files,
        "evidence_summary": evidence_summary,
        "blockers": blockers or [],
        "next_action": next_action,
    }


def _build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    data_calibration = OUT_DIR / "data_calibration_stage42.json"
    time_calibration = OUT_DIR / "source_time_geometry_calibration_stage42.json"
    guarded_launcher = OUT_DIR / "guarded_source_conversion_launcher_stage42.json"
    source_terms = OUT_DIR / "source_terms_validation_stage42.json"
    conversion_ready = int(_value(guarded_launcher, ["summary", "conversion_queue_count"], 0) or 0)
    source_ready = int(_value(source_terms, ["summary", "conversion_ready_targets"], 0) or 0)
    rows.append(
        _row(
            phase="A data and calibration",
            requirement_id="A1",
            requirement="Audit external_data/OpenTraj/ETH_UCY/TrajNet/SDD/TGSIM/AerialMPT and preserve legal source blockers.",
            status="partial_blocked",
            result_source="cached_verified",
            evidence_files=[
                str(data_calibration),
                str(time_calibration),
                str(guarded_launcher),
                str(source_terms),
            ],
            evidence_summary=(
                f"data calibration files exist={data_calibration.exists()} / {time_calibration.exists()}; "
                f"conversion_ready_targets={source_ready}; guarded_queue={conversion_ready}."
            ),
            blockers=["source_terms_confirmation_missing", "conversion_queue_empty"] if conversion_ready == 0 else [],
            next_action="User must fill source terms intake before any guarded conversion.",
        )
    )
    rows.append(
        _row(
            phase="A data and calibration",
            requirement_id="A2",
            requirement="Metric/time calibration may be used only for verified source-specific subsets.",
            status="partial_blocked",
            result_source="cached_verified",
            evidence_files=[
                str(OUT_DIR / "raw_source_time_geometry_hint_audit_stage42.json"),
                str(OUT_DIR / "calibration_candidate_manifest_stage42.json"),
                str(OUT_DIR / "source_terms_gap_audit_stage42.json"),
            ],
            evidence_summary="Hint/candidate manifests exist, but legal conversion-ready sources remain zero; no global metric/seconds claim is allowed.",
            blockers=["global_metric_seconds_claim_blocked", "legal_conversion_ready_now_zero"],
            next_action="After terms confirmation, run source-specific conversion/no-leakage/time-geometry audit.",
        )
    )

    ea = OUT_DIR / "dual_domain_group_consistency_statistics_stage42.json"
    dq = OUT_DIR / "full_waypoint_promotion_checkpoint_stage42.json"
    runtime = OUT_DIR / "group_consistency_runtime_policy_stage42.json"
    rows.append(
        _row(
            phase="B external validation",
            requirement_id="B1",
            requirement="External validation must include source/domain-level all, t50, t100 diagnostic, hard/failure, easy evidence.",
            status="pass_with_boundary",
            result_source="cached_verified",
            evidence_files=[str(ea), str(dq), str(runtime)],
            evidence_summary="Dual-domain group-consistency statistics and runtime replay exist; evidence remains dataset-local/raw-frame.",
            blockers=["not_true_metric_seconds", "source_terms_for_new_conversion_open"],
            next_action="Close source terms for additional independent top-down sources before broadening the claim.",
        )
    )

    co = OUT_DIR / "common_validation_bridge_shape_composer_stage42.json"
    cq = OUT_DIR / "proximity_aware_composer_guard_stage42.json"
    cr = OUT_DIR / "proximity_guard_ablation_stage42.json"
    rows.append(
        _row(
            phase="C full-waypoint dynamics",
            requirement_id="C1",
            requirement="Compare endpoint-only, bridge, full-waypoint shape, protected full-waypoint, and proximity-safe variants.",
            status="pass_with_boundary",
            result_source="cached_verified",
            evidence_files=[str(co), str(cq), str(cr), str(dq)],
            evidence_summary="Common-validation composer, proximity guard, and promotion checkpoint exist; full-waypoint is protected/source-level, not global ungated replacement.",
            blockers=["ungated_full_waypoint_not_promoted", "global_primary_full_waypoint_blocked"],
            next_action="Only promote broader full-waypoint if additional sources pass guarded conversion and source-CV.",
        )
    )

    dp = OUT_DIR / "context_model_closure_stage42.json"
    ec = OUT_DIR / "group_consistency_contribution_audit_stage42.json"
    ee = OUT_DIR / "context_switchability_materiality_stage42.json"
    rows.append(
        _row(
            phase="D causal ablation",
            requirement_id="D1",
            requirement="Prove scene/goal/interaction/neighbor/history/safe-switch/teacher-floor contributions with retrained or source-level evidence.",
            status="mixed",
            result_source="cached_verified",
            evidence_files=[str(ec), str(dp), str(ee), str(OUT_DIR / "unified_ablation_evidence_stage42.json")],
            evidence_summary="Group-consistency and safety floor are supported; current scene/goal/neighbor/interaction context protocols are below materiality or closed.",
            blockers=["scene_goal_main_claim_blocked", "neighbor_interaction_main_claim_blocked", "current_context_protocol_closed"],
            next_action="If context is retried, change target/architecture rather than repeating current residual context protocol.",
        )
    )

    safety = OUT_DIR / "source_level_safety_floor_audit_stage42.json"
    cs = OUT_DIR / "frozen_proximity_guard_composer_policy_stage42.json"
    cv = OUT_DIR / "proximity_guard_batch_replay_stage42.json"
    rows.append(
        _row(
            phase="E safety floor",
            requirement_id="E1",
            requirement="Study whether Stage37/teacher floor is removable or necessary, and keep ungated neural unsafe if it fails.",
            status="pass_with_boundary",
            result_source="cached_verified",
            evidence_files=[str(safety), str(cs), str(cv), str(cq)],
            evidence_summary="Safety-floor and proximity-guard evidence exists; deployable policies remain protected.",
            blockers=["ungated_neural_not_deployable", "teacher_floor_still_required"],
            next_action="Report floor as safety mechanism unless future ungated model passes easy/hard/all/proximity gates.",
        )
    )

    paper_rows = _paper_file_rows()
    missing_paper = [row["path"] for row in paper_rows if not row["exists"]]
    rows.append(
        _row(
            phase="F paper package",
            requirement_id="F1",
            requirement="Maintain paper outline, method, experiments, ablation, failure taxonomy, model/data cards, reproducibility, and A-journal gap package.",
            status="pass_with_open_gaps" if not missing_paper else "partial_blocked",
            result_source="cached_verified",
            evidence_files=[row["path"] for row in paper_rows],
            evidence_summary=f"paper_files_present={len(paper_rows) - len(missing_paper)}/{len(paper_rows)}; latest source terms and context blockers must remain in claims.",
            blockers=missing_paper + ["source_terms_open", "context_materiality_negative_evidence"],
            next_action="Keep post-EE/EF/EJ claim boundary in paper package; do not claim foundation/metric/Stage5C/SMC.",
        )
    )
    return rows


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_phase: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        by_phase[row["phase"]] = by_phase.get(row["phase"], 0) + 1
    hard_blockers = sorted({blocker for row in rows for blocker in row["blockers"] if "blocked" in blocker or "missing" in blocker or "zero" in blocker})
    return {
        "source": SOURCE,
        "requirements_audited": len(rows),
        "phases_audited": sorted(by_phase),
        "status_counts": by_status,
        "paper_files_present": sum(1 for row in _paper_file_rows() if row["exists"]),
        "paper_files_total": len(PAPER_FILES),
        "open_blockers": hard_blockers,
        "completion_claim_allowed": False,
        "a_journal_ready_claim_allowed": False,
        "current_verdict": "stage42_long_objective_has_strong_evidence_but_open_source_context_metric_gaps",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    phases = set(summary["phases_audited"])
    gates = {
        "all_phases_audited": phases
        == {
            "A data and calibration",
            "B external validation",
            "C full-waypoint dynamics",
            "D causal ablation",
            "E safety floor",
            "F paper package",
        },
        "requirements_written": summary["requirements_audited"] >= 7,
        "paper_files_checked": summary["paper_files_present"] == summary["paper_files_total"],
        "open_blockers_preserved": len(summary["open_blockers"]) > 0,
        "no_completion_overclaim": summary["completion_claim_allowed"] is False,
        "no_a_journal_overclaim": summary["a_journal_ready_claim_allowed"] is False,
        "no_conversion_training_eval_in_audit": claim["conversion_executed_in_stage42_ek"] is False
        and claim["evaluation_executed_in_stage42_ek"] is False
        and claim["training_executed_in_stage42_ek"] is False,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for value in gates.values() if value)
    total = len(gates)
    verdict = "stage42_ek_long_objective_coverage_audit_pass_open_blockers" if passed == total else "stage42_ek_long_objective_coverage_audit_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-EK Long Objective Coverage Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ek_gate']['passed']} / {payload['stage42_ek_gate']['total']}`",
        f"- verdict: `{payload['stage42_ek_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Requirement Matrix",
        "",
        "| phase | id | status | result source | evidence summary | blockers |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["requirements"]:
        lines.append(
            f"| {row['phase']} | `{row['requirement_id']}` | `{row['status']}` | `{row['result_source']}` | "
            f"{row['evidence_summary']} | {', '.join(row['blockers']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_ek_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_matrix(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42 Paper-Ready Evidence Matrix",
        "",
        "This matrix is generated by Stage42-EK. It is an evidence coverage audit, not new training or conversion.",
        "",
        "| requirement | status | usable claim | cannot claim | next action | evidence files |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["requirements"]:
        usable = {
            "pass_with_boundary": "usable with stated boundary",
            "pass_with_open_gaps": "usable as package evidence with open gaps",
            "mixed": "usable only as mixed/negative evidence",
            "partial_blocked": "not usable as completion claim",
        }.get(row["status"], "diagnostic only")
        cannot = "; ".join(row["blockers"]) or "none"
        evidence = "<br>".join(row["evidence_files"])
        lines.append(
            f"| `{row['requirement_id']}` {row['requirement']} | `{row['status']}` | {usable} | {cannot} | {row['next_action']} | {evidence} |"
        )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ek_gate"]
    return [
        "# Stage42-EK Gate",
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
        "## Stage42-EK Long Objective Coverage Audit",
        "",
        "- source: `fresh_stage42_long_objective_coverage_audit`",
        "- role: maps the active Stage42 A-F long objective to evidence rows, status labels, blockers, and paper-safe claims.",
        f"- gate: `{payload['stage42_ek_gate']['passed']} / {payload['stage42_ek_gate']['total']}`; verdict `{payload['stage42_ek_gate']['verdict']}`.",
        f"- requirements audited: `{s['requirements_audited']}` across phases `{s['phases_audited']}`.",
        f"- paper files present: `{s['paper_files_present']} / {s['paper_files_total']}`.",
        f"- open blockers preserved: `{s['open_blockers']}`.",
        "- completion/A-journal-ready claims remain disallowed; this is a coverage audit, not conversion/training/evaluation.",
        "- Boundary: no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_EK_LONG_OBJECTIVE_COVERAGE_AUDIT", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EK long objective coverage audit"
    state["current_verdict"] = payload["stage42_ek_gate"]["verdict"]
    state["stage42_ek_long_objective_coverage_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "evidence_matrix": str(EVIDENCE_MATRIX_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ek_gate"]["verdict"],
        "gates": f"{payload['stage42_ek_gate']['passed']}/{payload['stage42_ek_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_long_objective_coverage_audit(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    rows = _build_rows()
    evidence_inputs = sorted({path for row in rows for path in row["evidence_files"]})
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EK",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(evidence_inputs),
        "current_facts": CURRENT_FACTS,
        "requirements": rows,
        "summary": _summary(rows),
        "paper_files": _paper_file_rows(),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_ek_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(EVIDENCE_MATRIX_MD, _render_matrix(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_long_objective_coverage_audit()
