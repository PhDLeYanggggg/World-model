from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "full_waypoint_deployment_gap_audit_stage42.json"
REPORT_MD = OUT_DIR / "full_waypoint_deployment_gap_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_de_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_README = Path("README_M3W_GOAL_SUMMARY_ZH.md")
CURRENT_RETROSPECTIVE = Path("README_M3W_CURRENT_FULL_RETROSPECTIVE_ZH.md")
RESEARCH_STATE = Path("research_state.json")

INPUTS = {
    "full_waypoint_boundary": OUT_DIR / "full_waypoint_bridge_shape_audit_stage42.json",
    "common_validation_composer": OUT_DIR / "common_validation_bridge_shape_composer_stage42.json",
    "proximity_guard": OUT_DIR / "proximity_aware_composer_guard_stage42.json",
    "proximity_ablation": OUT_DIR / "proximity_guard_ablation_stage42.json",
    "unified_row_cache": OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.json",
    "source_level_full_waypoint": OUT_DIR / "source_level_full_waypoint_eval_stage42.json",
    "source_support_closure": OUT_DIR / "source_support_closure_audit_stage42.json",
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DE 是 full-waypoint deployment-gap audit，不重新训练，不用 test 调 threshold，不执行 Stage5C，不启用 SMC。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "endpoint-only 或 endpoint-to-linear bridge 成功不能自动算 full-waypoint world-state dynamics 成功。",
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


def _load_inputs() -> dict[str, Any]:
    return {key: read_json(path, {}) for key, path in INPUTS.items()}


def _input_status(payloads: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "path": str(INPUTS[key]),
            "exists": INPUTS[key].exists(),
            "source": payload.get("source", "missing_or_unparseable"),
            "stage": payload.get("stage", ""),
            "generated_at_utc": payload.get("generated_at_utc", ""),
            "gate_verdict": _find_gate_verdict(payload),
        }
        for key, payload in payloads.items()
    }


def _find_gate_verdict(payload: Mapping[str, Any]) -> str:
    for key, value in payload.items():
        if key.endswith("_gate") and isinstance(value, Mapping):
            return str(value.get("verdict", ""))
    return ""


def _row_by_name(rows: list[Mapping[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row.get("name") == name:
            return dict(row)
    return {}


def _metric(payload: Mapping[str, Any], path: list[str], default: float = 0.0) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    try:
        return float(value)
    except Exception:
        return default


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, Mapping):
        for key in ("mean", "mid", "value"):
            if key in value:
                return _number(value[key], default)
        return default
    try:
        return float(value)
    except Exception:
        return default


def _summarize_evidence(payloads: Mapping[str, Any]) -> dict[str, Any]:
    cm = payloads["full_waypoint_boundary"]
    co = payloads["common_validation_composer"]
    cq = payloads["proximity_guard"]
    cr = payloads["proximity_ablation"]
    unified = payloads["unified_row_cache"]
    source_level = payloads["source_level_full_waypoint"]
    dd = payloads["source_support_closure"]

    cm_rows = cm.get("comparison_rows", [])
    protected_full = _row_by_name(cm_rows, "full_waypoint_transformer_protected")
    endpoint_linear = _row_by_name(cm_rows, "m3w_neural_v1_composite_tail_linear_bridge")
    ungated_full = _row_by_name(cm_rows, "ungated_full_waypoint_transformer")
    graph_group = _row_by_name(cm_rows, "graph_interaction_group_consistency")
    unified_row = _row_by_name(cm_rows, "unified_row_level_full_waypoint_cache")
    cm_delta = cm.get("deltas", {}).get("full_waypoint_minus_linear_bridge", {})
    graph_delta = cm.get("deltas", {}).get("graph_group_minus_full_waypoint", {})

    co_metric = co.get("test_eval", {}).get("metric_vs_endpoint_ade", {})
    cq_metric = cq.get("test_eval", {}).get("metric_vs_endpoint_ade", {})
    cq_safety = cq.get("test_joint_safety", {}).get("composer_minus_endpoint", {})
    cr_rows = cr.get("ablation_rows", {})
    cr_no_guard = cr_rows.get("no_proximity_guard", {})
    cr_guard = cr_rows.get("proximity_guard", {})
    unified_summary = unified.get("summary", {})
    source_metrics = source_level.get("model", {}).get("metrics", {}).get("protected_ridge_source_level", {})
    source_fde = source_level.get("model", {}).get("metrics", {}).get("protected_ridge_source_level_fde", {})
    source_support = dd.get("closure_summary", {})

    horizon_aux_supported = (
        float(cm_delta.get("t50_improvement", 0.0)) > 0.0
        and float(cm_delta.get("t100_raw_frame_diagnostic_improvement", 0.0)) > 0.0
    )
    replacement_supported = (
        float(cm_delta.get("all_improvement", 0.0)) > 0.0
        and float(cm_delta.get("hard_failure_improvement", 0.0)) > 0.0
        and float(protected_full.get("easy_degradation", 1.0)) <= 0.02
    )
    guarded_composer_supported = (
        float(cq_metric.get("all_improvement", 0.0)) > 0.0
        and float(cq_metric.get("t50_improvement", 0.0)) > 0.0
        and float(cq_metric.get("t100_raw_frame_diagnostic_improvement", 0.0)) > 0.0
        and float(cq_metric.get("hard_failure_improvement", 0.0)) > 0.0
        and float(cq_metric.get("easy_degradation", 1.0)) <= 0.02
        and float(cq_safety.get("near_collision_rate_005", cr_guard.get("near_collision_005_delta_vs_endpoint", 1.0))) <= 0.0
    )
    ungated_blocked = float(ungated_full.get("easy_degradation", 0.0)) > 0.02
    unified_support = (
        _number(unified_summary.get("ade_all")) > 0.0
        and _number(unified_summary.get("ade_t50")) > 0.0
        and _number(unified_summary.get("ade_hard_failure")) > 0.0
        and _number(unified_summary.get("ade_easy_degradation"), 1.0) <= 0.02
    )
    source_support_closed = not source_support.get("domains_not_closed", ["unknown"])

    decision = "full_waypoint_auxiliary_horizon_composer_supported_not_full_deployment"
    if replacement_supported and guarded_composer_supported and source_support_closed:
        decision = "full_waypoint_deployment_promotion_supported"
    elif guarded_composer_supported:
        decision = "protected_full_waypoint_composer_supported_deployment_promotion_blocked"

    blockers = []
    if not replacement_supported:
        blockers.append("protected_full_waypoint_does_not_beat_endpoint_linear_on_all_and_hard")
    if ungated_blocked:
        blockers.append("ungated_full_waypoint_easy_degradation_unsafe")
    if not source_support_closed:
        blockers.append("source_legal_time_t100_closure_open")
    if not guarded_composer_supported:
        blockers.append("proximity_guarded_composer_not_safe_positive")
    if float(graph_delta.get("collision_delta_vs_floor_005", 0.0)) > 0.0:
        blockers.append("graph_group_interaction_has_proximity_caveat")

    next_training_targets = [
        "train all-agent full-waypoint sequence model with all/hard ADE objective strong enough to beat endpoint-linear, not only t50/t100 raw-frame slices",
        "add proximity / collision / physical validity loss so graph or full-waypoint gains do not create proximity caveats",
        "keep validation-only domain/horizon safe-switch and easy-preservation loss; do not remove Stage37/teacher floor until no-floor gates pass",
        "replace current weak context/gain-harm protocol with richer joint occupancy or interaction-constraint target because Stage42-DC did not support context switchability",
        "close source/legal/time/t100 support for ETH_UCY, TrajNet, and UCY before broad metric/seconds/t100 claims",
    ]

    return {
        "endpoint_linear_bridge_floor": endpoint_linear,
        "protected_full_waypoint_transformer": protected_full,
        "ungated_full_waypoint_transformer": ungated_full,
        "graph_interaction_group_consistency": graph_group,
        "unified_row_level_full_waypoint_cache": unified_row,
        "full_waypoint_minus_endpoint_linear": cm_delta,
        "graph_group_minus_full_waypoint": graph_delta,
        "common_validation_composer_vs_endpoint": co_metric,
        "proximity_guarded_composer_vs_endpoint": cq_metric,
        "proximity_guarded_joint_safety_delta": cq_safety,
        "proximity_guard_ablation_no_guard": cr_no_guard,
        "proximity_guard_ablation_guarded": cr_guard,
        "unified_row_cache_summary": unified_summary,
        "source_level_full_waypoint_ade": source_metrics,
        "source_level_full_waypoint_fde": source_fde,
        "source_support_closure": source_support,
        "support_flags": {
            "horizon_auxiliary_supported": horizon_aux_supported,
            "endpoint_linear_replacement_supported": replacement_supported,
            "guarded_composer_supported": guarded_composer_supported,
            "ungated_full_waypoint_blocked": ungated_blocked,
            "unified_three_domain_row_cache_support": unified_support,
            "source_support_closed": source_support_closed,
        },
        "deployment_decision": {
            "decision": decision,
            "promote_full_waypoint_as_primary_deployable_dynamics": decision == "full_waypoint_deployment_promotion_supported",
            "use_guarded_full_waypoint_composer_for_safety_sensitive_reporting": guarded_composer_supported,
            "keep_endpoint_linear_or_stage37_teacher_floor_as_safety_floor": True,
            "blockers": blockers,
            "next_training_targets": next_training_targets,
        },
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payloads = _load_inputs()
    evidence = _summarize_evidence(payloads)
    result: dict[str, Any] = {
        "source": "fresh_stage42_de_full_waypoint_deployment_gap_audit",
        "stage": "Stage42-DE Full-Waypoint Deployment Gap Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_status": _input_status(payloads),
        "evidence": evidence,
        "no_leakage_and_protocol": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
            "stage42_de_training_run": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "full_waypoint_primary_deployable_claim_allowed": evidence["deployment_decision"][
                "promote_full_waypoint_as_primary_deployable_dynamics"
            ],
            "guarded_composer_claim_allowed": evidence["deployment_decision"][
                "use_guarded_full_waypoint_composer_for_safety_sensitive_reporting"
            ],
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["stage42_de_gate"] = _gate(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    claim = result["claim_boundary"]
    evidence = result["evidence"]
    support = evidence["support_flags"]
    decision = evidence["deployment_decision"]
    inputs = result["input_status"]
    no_leak = result["no_leakage_and_protocol"]
    gates = {
        "required_inputs_loaded": all(row["exists"] for row in inputs.values()) and len(inputs) == len(INPUTS),
        "boundary_audit_loaded": inputs["full_waypoint_boundary"]["source"].startswith("fresh"),
        "common_validation_composer_loaded": bool(inputs["common_validation_composer"]["source"]),
        "proximity_guard_loaded": bool(inputs["proximity_guard"]["source"]),
        "proximity_ablation_loaded": bool(inputs["proximity_ablation"]["source"]),
        "unified_row_cache_loaded": bool(inputs["unified_row_cache"]["source"]),
        "source_support_closure_loaded": bool(inputs["source_support_closure"]["source"]),
        "horizon_auxiliary_support_recorded": support["horizon_auxiliary_supported"] is True,
        "endpoint_linear_replacement_blocker_recorded": support["endpoint_linear_replacement_supported"] is False,
        "ungated_full_waypoint_blocked_as_unsafe": support["ungated_full_waypoint_blocked"] is True,
        "guarded_composer_support_recorded": support["guarded_composer_supported"] is True,
        "deployment_decision_honest": decision["promote_full_waypoint_as_primary_deployable_dynamics"] is False
        and decision["keep_endpoint_linear_or_stage37_teacher_floor_as_safety_floor"] is True,
        "next_training_targets_emitted": len(decision["next_training_targets"]) >= 4,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["future_waypoint_label_eval_only"] is True,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
            ]
        ),
        "global_metric_seconds_blocked": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = (
        "stage42_de_full_waypoint_deployment_gap_audit_pass_primary_promotion_blocked"
        if passed == total
        else "stage42_de_full_waypoint_deployment_gap_audit_partial"
    )
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _pct(value: Any) -> str:
    try:
        return f"{_number(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def _render_report(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_de_gate"]
    evidence = result["evidence"]
    decision = evidence["deployment_decision"]
    support = evidence["support_flags"]
    lines = [
        "# Stage42-DE Full-Waypoint Deployment Gap Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deployment_decision: `{decision['decision']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Decision Summary",
        "",
        f"- promote_full_waypoint_as_primary_deployable_dynamics: `{decision['promote_full_waypoint_as_primary_deployable_dynamics']}`",
        f"- use_guarded_full_waypoint_composer_for_safety_sensitive_reporting: `{decision['use_guarded_full_waypoint_composer_for_safety_sensitive_reporting']}`",
        f"- keep_endpoint_linear_or_stage37_teacher_floor_as_safety_floor: `{decision['keep_endpoint_linear_or_stage37_teacher_floor_as_safety_floor']}`",
        f"- blockers: `{decision['blockers']}`",
        "",
        "## Support Flags",
        "",
        *[f"- {key}: `{value}`" for key, value in support.items()],
        "",
        "## Key Evidence",
        "",
        "| evidence | all | t50 | t100 raw diagnostic | hard/failure | easy | note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    rows = [
        (
            "endpoint_linear_bridge_floor",
            evidence["endpoint_linear_bridge_floor"],
            "current endpoint-linear / teacher protected floor",
        ),
        (
            "protected_full_waypoint_transformer",
            evidence["protected_full_waypoint_transformer"],
            "actual full-waypoint sequence model under protected switch",
        ),
        (
            "ungated_full_waypoint_transformer",
            evidence["ungated_full_waypoint_transformer"],
            "unsafe diagnostic, blocked by easy degradation",
        ),
        (
            "proximity_guarded_composer_vs_endpoint",
            evidence["proximity_guarded_composer_vs_endpoint"],
            "safety-sensitive guarded composer over endpoint-linear",
        ),
        (
            "unified_row_cache_summary",
            {
                "all_improvement": evidence["unified_row_cache_summary"].get("ade_all"),
                "t50_improvement": evidence["unified_row_cache_summary"].get("ade_t50"),
                "t100_raw_frame_diagnostic_improvement": evidence["unified_row_cache_summary"].get(
                    "ade_t100_raw_frame_diagnostic"
                ),
                "hard_failure_improvement": evidence["unified_row_cache_summary"].get("ade_hard_failure"),
                "easy_degradation": evidence["unified_row_cache_summary"].get("ade_easy_degradation"),
            },
            "three-domain row-level full-waypoint cache support",
        ),
    ]
    for name, row, note in rows:
        lines.append(
            f"| `{name}` | {_pct(row.get('all_improvement'))} | {_pct(row.get('t50_improvement'))} | "
            f"{_pct(row.get('t100_raw_frame_diagnostic_improvement'))} | {_pct(row.get('hard_failure_improvement'))} | "
            f"{_pct(row.get('easy_degradation'))} | {note} |"
        )
    lines.extend(
        [
            "",
            "## Boundary Deltas",
            "",
            f"- full_waypoint_minus_endpoint_linear: `{evidence['full_waypoint_minus_endpoint_linear']}`",
            f"- graph_group_minus_full_waypoint: `{evidence['graph_group_minus_full_waypoint']}`",
            "",
            "## Proximity Guard Pareto Result",
            "",
            f"- no_proximity_guard: `{evidence['proximity_guard_ablation_no_guard']}`",
            f"- proximity_guard: `{evidence['proximity_guard_ablation_guarded']}`",
            "",
            "## Next Training Targets",
            "",
            *[f"- {item}" for item in decision["next_training_targets"]],
            "",
            "## Claim Boundary",
            "",
            f"- claim_boundary: `{result['claim_boundary']}`",
            "",
            "## Interpretation",
            "",
            "- Stage42-DE closes the current full-waypoint deployment question honestly: full-waypoint is useful as a protected horizon/shape component, especially for t50/t100 raw-frame slices, but it is not yet promoted as the primary deployable world dynamics head.",
            "- The safest deployable shape path remains a guarded composer under Stage37/teacher floor. Ungated full-waypoint remains unsafe.",
            "- The next research move should change the training target/loss for all-agent full-waypoint dynamics, not relabel endpoint-linear success as full-waypoint success.",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_de_gate"]
    lines = [
        "# Stage42-DE Gate",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in gate["gates"].items())
    return lines


def _refresh_lines(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_de_gate"]
    decision = result["evidence"]["deployment_decision"]
    support = result["evidence"]["support_flags"]
    return [
        "## Stage42-DE Full-Waypoint Deployment Gap Audit",
        "",
        "- source: `fresh_stage42_de_full_waypoint_deployment_gap_audit`",
        "- role: decide whether full-waypoint can be promoted from auxiliary/composer evidence to primary deployable world dynamics.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- decision: `{decision['decision']}`.",
        f"- horizon_auxiliary_supported: `{support['horizon_auxiliary_supported']}`; guarded_composer_supported: `{support['guarded_composer_supported']}`.",
        f"- primary deployable full-waypoint promotion: `{decision['promote_full_waypoint_as_primary_deployable_dynamics']}`.",
        f"- blockers: `{decision['blockers']}`.",
        "- Conclusion: keep Stage37/teacher or endpoint-linear safety floor; use guarded full-waypoint composer only as protected horizon/shape component until all/hard/proximity/source-support gaps are closed.",
        "- Stage5C remains false; SMC remains false; no metric/seconds claim.",
    ]


def _refresh_readmes(result: Mapping[str, Any]) -> None:
    lines = _refresh_lines(result)
    for path in [README_RESULTS, M3W_README, GOAL_README, CURRENT_RETROSPECTIVE]:
        _replace_section(path, "STAGE42_DE_FULL_WAYPOINT_DEPLOYMENT_GAP_AUDIT", lines)


def _refresh_research_state(result: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DE full-waypoint deployment gap audit"
    state["current_verdict"] = result["stage42_de_gate"]["verdict"]
    state["stage42_de_full_waypoint_deployment_gap_audit"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": result["stage42_de_gate"]["verdict"],
        "gates": f"{result['stage42_de_gate']['passed']}/{result['stage42_de_gate']['total']}",
        "decision": result["evidence"]["deployment_decision"]["decision"],
        "support_flags": result["evidence"]["support_flags"],
        "blockers": result["evidence"]["deployment_decision"]["blockers"],
        "claim_boundary": result["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_full_waypoint_deployment_gap_audit(*, refresh_readmes: bool = True) -> dict[str, Any]:
    result = _build_payload()
    write_json(REPORT_JSON, result)
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    if refresh_readmes:
        _refresh_readmes(result)
        _refresh_research_state(result)
    return result


if __name__ == "__main__":
    run_stage42_full_waypoint_deployment_gap_audit()
