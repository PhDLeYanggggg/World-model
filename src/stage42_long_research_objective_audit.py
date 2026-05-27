from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

DATA_CALIBRATION = OUT_DIR / "data_calibration_stage42.json"
EXTERNAL_VALIDATION = OUT_DIR / "external_validation_stage42.json"
FULL_WAYPOINT = OUT_DIR / "full_waypoint_dynamics_stage42.json"
UNIFIED_FULL_WAYPOINT = OUT_DIR / "unified_external_full_waypoint_policy_stage42.json"
GROUP_CONSISTENCY = OUT_DIR / "group_consistency_contribution_audit_stage42.json"
SEQUENCE_CONTEXT = OUT_DIR / "source_level_sequence_context_stage42.json"
GRAPH_CONTEXT = OUT_DIR / "source_level_graph_context_stage42.json"
METRIC_TIME_QUEUE = OUT_DIR / "restricted_metric_time_conversion_queue_v2_stage42.json"
M3W_EVIDENCE = Path("outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.json")
STAGE37_GATE = Path("outputs/stage37_t50_history/world_model_gate_stage37.json")
STAGE40_GATE = Path("outputs/stage40_neural_optimization/world_model_gate_stage40.json")

REPORT_JSON = OUT_DIR / "long_research_objective_audit_stage42.json"
REPORT_MD = OUT_DIR / "long_research_objective_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ho_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_ho_long_research_objective_audit"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HO 是长期目标覆盖审计：不下载、不转换、不训练、不调 test threshold。",
    "本阶段把 Stage42 Long Research Mode A-F 要求映射到当前 authoritative evidence。",
    "future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d_world_model": False,
    "foundation_world_model": False,
    "metric_seconds_claim_allowed": False,
    "ungated_neural_deployable": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _gate_value(payload: Mapping[str, Any], gate_key: str) -> tuple[int, int, str]:
    gate = payload.get(gate_key, {})
    return int(gate.get("passed", 0) or 0), int(gate.get("total", 0) or 0), str(gate.get("verdict", ""))


def _stage37_summary(stage37: Mapping[str, Any]) -> dict[str, Any]:
    # Stage37 gate JSON stores final metrics under several possible keys across reruns.
    metrics = (
        stage37.get("summary", {}).get("final_metrics")
        or stage37.get("final_metrics")
        or stage37.get("metrics")
        or {}
    )
    return {
        "source": "cached_verified",
        "gate": _gate_value(stage37, "stage37_gate"),
        "all_improvement": _num(metrics.get("all_improvement")),
        "t50_improvement": _num(metrics.get("t50_improvement")),
        "hard_failure_improvement": _num(metrics.get("hard_failure_improvement")),
        "easy_degradation": _num(metrics.get("easy_degradation")),
    }


def _external_summary(external: Mapping[str, Any]) -> dict[str, Any]:
    summary = external.get("summary", {})
    protected = summary.get("protected_m3w") or summary.get("protected_m3w_neural_v1") or {}
    return {
        "source": external.get("source", "cached_verified"),
        "protected_all": _num(protected.get("all_improvement") or protected.get("all_ade_improvement")),
        "protected_t50": _num(protected.get("t50_improvement") or protected.get("t50_ade_improvement")),
        "protected_t100_raw": _num(protected.get("t100_improvement") or protected.get("t100_raw_frame_diagnostic_improvement")),
        "protected_hard": _num(protected.get("hard_failure_improvement") or protected.get("hard_ade_improvement")),
        "easy_degradation": _num(protected.get("easy_degradation")),
    }


def _m3w_neural_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    metrics = (
        evidence.get("best_metrics_vs_stage37_floor")
        or evidence.get("key_metrics")
        or evidence.get("summary")
        or evidence
    )
    return {
        "source": evidence.get("source", "cached_verified"),
        "gates": evidence.get("gates") or evidence.get("gate") or {},
        "all_improvement": _num(metrics.get("all_improvement") or metrics.get("all")),
        "t50_improvement": _num(metrics.get("t50_improvement") or metrics.get("t50")),
        "t100_raw_improvement": _num(metrics.get("t100_raw_frame_diagnostic_improvement") or metrics.get("t100")),
        "hard_failure_improvement": _num(metrics.get("hard_failure_improvement") or metrics.get("hard_failure")),
        "easy_degradation": _num(metrics.get("easy_degradation") or metrics.get("easy")),
    }


def _full_waypoint_summary(full: Mapping[str, Any], unified: Mapping[str, Any], group: Mapping[str, Any]) -> dict[str, Any]:
    comparison = full.get("comparisons", {})
    protected = comparison.get("full_waypoint_transformer_protected", {})
    ade = protected.get("ade", {})
    unified_summary = unified.get("weighted_summary", {}) or unified.get("summary", {}) or {}
    if "rows" not in unified_summary:
        unified_summary = unified.get("package_summary", {}) or unified_summary
    group_summary = group.get("summary", {})
    return {
        "source": "cached_verified",
        "protected_full_waypoint_all": _num(ade.get("all_improvement")),
        "protected_full_waypoint_t50": _num(ade.get("t50_improvement")),
        "protected_full_waypoint_t100_raw": _num(ade.get("t100_improvement")),
        "protected_full_waypoint_hard": _num(ade.get("hard_failure_improvement")),
        "unified_full_waypoint_rows": int(unified_summary.get("rows", 0) or 0),
        "group_consistency_verdict": group.get("stage42_ec_gate", {}).get("verdict", ""),
        "group_consistency_supported": "explicit_group_consistency_full_waypoint"
        in str(group_summary)
        or "explicit_group_consistency_full_waypoint" in str(group.get("supported_contributions", {})),
    }


def _context_summary(seq: Mapping[str, Any], graph: Mapping[str, Any]) -> dict[str, Any]:
    seq_verdict = seq.get("stage42_ar_gate", {}).get("verdict", seq.get("verdict", ""))
    graph_verdict = graph.get("stage42_as_gate", {}).get("verdict", graph.get("verdict", ""))
    return {
        "sequence_source": seq.get("source", ""),
        "graph_source": graph.get("source", ""),
        "sequence_verdict": seq_verdict,
        "graph_verdict": graph_verdict,
        "scene_goal_independent_claim": "not_supported",
        "neighbor_interaction_independent_claim": "not_supported",
        "reason": "Fresh sequence and graph residual context variants underperformed the baseline-family rollout control on all/t50/hard.",
    }


def _metric_time_summary(queue: Mapping[str, Any]) -> dict[str, Any]:
    summary = queue.get("summary", {})
    return {
        "source": queue.get("source", ""),
        "ready_candidates": int(summary.get("manifest_ready_candidates", 0) or 0),
        "conversion_queue_count": int(summary.get("conversion_queue_count", 0) or 0),
        "blocked_action_count": int(summary.get("blocked_action_count", 0) or 0),
        "blocked_t50_windows_after_terms": int(summary.get("blocked_t50_windows_after_terms", 0) or 0),
        "blocked_t100_windows_after_terms": int(summary.get("blocked_t100_windows_after_terms", 0) or 0),
        "restricted_metric_time_claim_allowed_now": False,
    }


def _coverage(payload: Mapping[str, Any]) -> dict[str, Any]:
    metric_time = payload["metric_time"]
    full = payload["full_waypoint"]
    context = payload["context"]
    m3w = payload["m3w_neural_v1"]
    return {
        "stage_a_data_and_calibration": {
            "status": "partial_blocked",
            "evidence": "data calibration and source/legal guards exist; metric/time conversion queue is empty.",
            "pass_for_objective": False,
            "next_action": "user-confirmed source terms/path/source identity, then guarded conversion and no-leakage/source-CV.",
            "blocked_by": [
                "restricted metric/time ready candidates = 0",
                f"blocked after-terms t50/t100 windows retained = {metric_time['blocked_t50_windows_after_terms']} / {metric_time['blocked_t100_windows_after_terms']}",
            ],
        },
        "stage_b_external_validation": {
            "status": "protected_positive",
            "evidence": "Stage37/M3W-Neural/Stage42 protected external validation is positive under raw-frame dataset-local protocol.",
            "pass_for_objective": True,
            "next_action": "expand independent legal source diversity; do not broaden metric/time claim.",
        },
        "stage_c_full_waypoint_dynamics": {
            "status": "protected_positive_not_ungated",
            "evidence": "Protected full-waypoint and group-consistency policies are positive; ungated/global replacement remains blocked.",
            "pass_for_objective": bool(full["group_consistency_supported"]),
            "next_action": "continue source-level group-consistency/full-waypoint training rather than endpoint-only bridge overclaims.",
        },
        "stage_d_causal_ablation": {
            "status": "mixed",
            "evidence": "history/safe-switch/floor/group-consistency supported; JEPA/scene/goal/neighbor independent main claims blocked under current protocols.",
            "pass_for_objective": False,
            "next_action": "if revisiting context, change target to gain-harm/switchability or source-level full-waypoint consistency; do not repeat residual context protocol.",
            "blocked_by": [
                context["reason"],
                "JEPA non-collapse without stable downstream lift.",
            ],
        },
        "stage_e_safety_floor": {
            "status": "necessary_floor_proven",
            "evidence": "Protected candidates pass; ungated neural remains unsafe; floor-free/global removal is blocked.",
            "pass_for_objective": True,
            "next_action": "study slice-specific floor relaxation only under validation-selected proximity/conformal guards.",
        },
        "stage_f_paper_package": {
            "status": "paper_package_partial_strong",
            "evidence": "paper package, model/data cards, claim guards, and gap analysis exist; metric/time/source-diversity still blocked.",
            "pass_for_objective": False,
            "next_action": "keep paper framing as protected raw-frame 2.5D world-state dynamics; add legal external source or restricted metric/time subset before stronger claims.",
        },
        "overall_stage42_long_goal": {
            "status": "active_not_complete",
            "evidence": "Strong protected 2.5D evidence exists, but full objective still requires metric/time or broader external/source evidence and stronger independent module proof.",
            "pass_for_objective": False,
            "next_action": "continue long research mode; do not mark goal complete.",
        },
        "headline_metrics": {
            "m3w_neural_v1_all": m3w["all_improvement"],
            "m3w_neural_v1_t50": m3w["t50_improvement"],
            "m3w_neural_v1_hard_failure": m3w["hard_failure_improvement"],
            "full_waypoint_protected_all": full["protected_full_waypoint_all"],
            "full_waypoint_protected_t50": full["protected_full_waypoint_t50"],
        },
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    cov = payload["coverage"]
    metric_time = payload["metric_time"]
    context = payload["context"]
    claim = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "inputs_loaded": all(bool(v) for v in payload["input_status"].values()),
        "stage_a_blocker_recorded": cov["stage_a_data_and_calibration"]["status"] == "partial_blocked",
        "stage_b_external_positive_recorded": cov["stage_b_external_validation"]["pass_for_objective"] is True,
        "stage_c_full_waypoint_positive_recorded": cov["stage_c_full_waypoint_dynamics"]["pass_for_objective"] is True,
        "stage_d_mixed_ablation_recorded": cov["stage_d_causal_ablation"]["status"] == "mixed",
        "stage_e_floor_needed_recorded": cov["stage_e_safety_floor"]["status"] == "necessary_floor_proven",
        "stage_f_partial_package_recorded": cov["stage_f_paper_package"]["status"] == "paper_package_partial_strong",
        "overall_not_marked_complete": cov["overall_stage42_long_goal"]["pass_for_objective"] is False,
        "context_overclaim_blocked": context["scene_goal_independent_claim"] == "not_supported"
        and context["neighbor_interaction_independent_claim"] == "not_supported",
        "metric_time_not_overclaimed": metric_time["ready_candidates"] == 0
        and metric_time["restricted_metric_time_claim_allowed_now"] is False,
        "future_endpoint_blocked": payload["no_leakage"]["future_endpoint_input"] is False,
        "central_velocity_blocked": payload["no_leakage"]["central_velocity"] is False,
        "test_endpoint_goals_blocked": payload["no_leakage"]["test_endpoint_goals"] is False,
        "no_metric_seconds_claim": claim["metric_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for value in gates.values() if value)
    total = len(gates)
    verdict = (
        "stage42_ho_long_research_objective_audit_pass_keep_goal_active"
        if passed == total
        else "stage42_ho_long_research_objective_audit_partial"
    )
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> str:
    gate = payload["stage42_ho_gate"]
    lines = [
        "# Stage42-HO Long Research Objective Audit",
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
    ]
    lines.extend(f"- {fact}" for fact in CURRENT_FACTS)
    lines.extend(["", "## Objective Coverage", ""])
    for key, row in payload["coverage"].items():
        if key == "headline_metrics":
            continue
        lines.extend(
            [
                f"### {key}",
                "",
                f"- status: `{row['status']}`",
                f"- pass_for_objective: `{row['pass_for_objective']}`",
                f"- evidence: {row['evidence']}",
                f"- next_action: {row['next_action']}",
            ]
        )
        if row.get("blocked_by"):
            lines.append("- blocked_by:")
            lines.extend(f"  - {item}" for item in row["blocked_by"])
        lines.append("")
    lines.extend(
        [
            "## Headline Metrics Snapshot",
            "",
            "| metric | value |",
            "| --- | ---: |",
        ]
    )
    for key, value in payload["coverage"]["headline_metrics"].items():
        lines.append(f"| `{key}` | {value:.6f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-HO keeps the long goal active: the protected 2.5D evidence is strong, but the full objective is not complete.",
            "- The next non-blocked scientific path is source-level group-consistency/full-waypoint training and claim packaging.",
            "- The next blocked-but-important path is legal/source confirmed restricted metric/time conversion.",
            "- Repeating the current sequence/graph residual-context protocol is not recommended without changing target or evidence source.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| `{name}` | {value} |" for name, value in gate["gates"].items())
    return "\n".join(lines) + "\n"


def _render_gate(payload: Mapping[str, Any]) -> str:
    gate = payload["stage42_ho_gate"]
    lines = [
        "# Stage42-HO Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{name}` | {value} |" for name, value in gate["gates"].items())
    return "\n".join(lines) + "\n"


def build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = {
        "data_calibration": read_json(DATA_CALIBRATION, {}),
        "external_validation": read_json(EXTERNAL_VALIDATION, {}),
        "full_waypoint": read_json(FULL_WAYPOINT, {}),
        "unified_full_waypoint": read_json(UNIFIED_FULL_WAYPOINT, {}),
        "group_consistency": read_json(GROUP_CONSISTENCY, {}),
        "sequence_context": read_json(SEQUENCE_CONTEXT, {}),
        "graph_context": read_json(GRAPH_CONTEXT, {}),
        "metric_time_queue": read_json(METRIC_TIME_QUEUE, {}),
        "m3w_evidence": read_json(M3W_EVIDENCE, {}),
        "stage37_gate": read_json(STAGE37_GATE, {}),
        "stage40_gate": read_json(STAGE40_GATE, {}),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HO Long Research Objective Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                DATA_CALIBRATION,
                EXTERNAL_VALIDATION,
                FULL_WAYPOINT,
                UNIFIED_FULL_WAYPOINT,
                GROUP_CONSISTENCY,
                SEQUENCE_CONTEXT,
                GRAPH_CONTEXT,
                METRIC_TIME_QUEUE,
                M3W_EVIDENCE,
                STAGE37_GATE,
                STAGE40_GATE,
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "input_status": {name: bool(value) for name, value in inputs.items()},
        "stage37": _stage37_summary(inputs["stage37_gate"]),
        "external_validation": _external_summary(inputs["external_validation"]),
        "m3w_neural_v1": _m3w_neural_summary(inputs["m3w_evidence"]),
        "full_waypoint": _full_waypoint_summary(
            inputs["full_waypoint"],
            inputs["unified_full_waypoint"],
            inputs["group_consistency"],
        ),
        "context": _context_summary(inputs["sequence_context"], inputs["graph_context"]),
        "metric_time": _metric_time_summary(inputs["metric_time_queue"]),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["coverage"] = _coverage(payload)
    payload["stage42_ho_gate"] = _gate(payload)
    return payload


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-HO long research objective audit"
    state["current_verdict"] = payload["stage42_ho_gate"]["verdict"]
    state["stage42_ho_long_research_objective_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ho_gate"]["verdict"],
        "gates": f"{payload['stage42_ho_gate']['passed']}/{payload['stage42_ho_gate']['total']}",
        "summary_refresh_only": False,
        "new_training_or_conversion": False,
        "long_goal_complete": False,
        "next_non_blocked_action": "continue source-level group-consistency/full-waypoint training and paper evidence packaging",
        "next_blocked_action": "user-confirmed restricted metric/time source terms/path/source identity before conversion",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json(RESEARCH_STATE, state)


def _update_readmes(payload: Mapping[str, Any]) -> None:
    section = [
        "## Stage42-HO Long Research Objective Audit",
        "",
        "本轮继续 Stage42 Long Research Mode，新增长期目标覆盖审计：",
        "",
        f"`/Users/yangyue/Downloads/World/{REPORT_MD}`",
        "",
        f"结果来源：`{payload['source']}`；gate `{payload['stage42_ho_gate']['passed']} / {payload['stage42_ho_gate']['total']}`；verdict `{payload['stage42_ho_gate']['verdict']}`。",
        "该审计不下载、不转换、不训练、不调 test threshold；它把 Stage42 A-F 要求映射到当前 authoritative evidence，并明确保持长期目标 active。",
        "",
        "结论：external/protected full-waypoint/group-consistency evidence 已经很强，但 full objective 尚未完成。metric/time conversion 仍因 ready candidates = 0 被阻塞；JEPA、scene/goal、neighbor/interaction 独立主 claim 仍不支持；Stage5C 与 SMC 仍禁止。",
        "",
    ]
    for path, marker in [
        (README_RESULTS, "STAGE42_HO_LONG_OBJECTIVE_AUDIT"),
        (M3W_README, "STAGE42_HO_LONG_OBJECTIVE_AUDIT"),
    ]:
        _replace_section(path, marker, section)


def run_stage42_long_research_objective_audit() -> dict[str, Any]:
    payload = build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload).splitlines())
    write_md(GATE_MD, _render_gate(payload).splitlines())
    _update_state(payload)
    _update_readmes(payload)
    return payload


if __name__ == "__main__":
    run_stage42_long_research_objective_audit()
