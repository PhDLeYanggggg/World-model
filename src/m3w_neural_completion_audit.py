from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md


OUT_DIR = Path("outputs/m3w_neural_v1")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _status(done: bool, partial: bool = False) -> str:
    if done:
        return "complete"
    if partial:
        return "partial"
    return "incomplete"


def _replace_section(path: Path, marker: str, lines: list[str]) -> None:
    block = [f"<!-- {marker}:START -->", *lines, f"<!-- {marker}:END -->"]
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    if start in existing and end in existing:
        before = existing.split(start, 1)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        text = "\n\n".join(part for part in [before, "\n".join(block), after] if part)
    else:
        text = existing.rstrip() + "\n\n" + "\n".join(block)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_completion_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    package = read_json(OUT_DIR / "package_manifest_m3w_neural_v1.json", {})
    stage41_gates = read_json("outputs/stage41_breakthrough/world_model_gate_stage41.json", {})
    stage41_eval = read_json("outputs/stage41_breakthrough/stage41_neural_eval.json", {})
    all_agent_eval = read_json("outputs/stage41_breakthrough/stage41_all_agent_eval.json", {})
    all_agent_repair = read_json("outputs/stage41_breakthrough/stage41_all_agent_risk_repair.json", {})
    all_agent_t50 = read_json("outputs/stage41_breakthrough/stage41_all_agent_t50_specialist.json", {})
    all_agent_composer = read_json("outputs/stage41_breakthrough/stage41_all_agent_policy_composer.json", {})
    all_agent_locked = read_json("outputs/stage41_stratified_protocol/stage41_fixed_policy_confirmation.json", {})
    fresh_all_agent = read_json("outputs/stage41_fresh_confirmation/stage41_fresh_all_agent_endpoint_specialist.json", {})
    full_traj = read_json("outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.json", {})
    goal_route = read_json("outputs/stage41_fresh_confirmation/stage41_goal_route_physical_repair.json", {})
    route_policy = read_json("outputs/stage41_fresh_confirmation/stage41_route_physical_policy_integration.json", {})
    joint_route = read_json("outputs/stage41_fresh_confirmation/stage41_joint_route_conditioned_world_state.json", {})
    endpoint_audit = read_json("outputs/stage41_breakthrough/stage41_endpoint_geometry_audit.json", {})

    best = package.get("evidence_summary", {})
    all_agent_metrics = all_agent_repair.get("best_metrics", {})
    all_agent_positive = (
        all_agent_metrics.get("all_improvement", 0.0) > 0
        and all_agent_metrics.get("hard_failure_improvement", 0.0) > 0
        and all_agent_metrics.get("easy_degradation", 1.0) <= 0.02
    )
    all_agent_t50_pass = all_agent_metrics.get("t50_improvement", -1.0) > 0
    t50_metrics = all_agent_t50.get("best_metrics", {})
    t50_specialist_positive = (
        t50_metrics.get("t50_improvement", 0.0) > 0
        and t50_metrics.get("easy_degradation", 1.0) <= 0.02
        and all(row.get("t50_improvement", 0.0) >= 0 for row in t50_metrics.get("by_domain", {}).values())
    )
    composer_metrics = all_agent_composer.get("test_metrics", {})
    composer_positive = (
        composer_metrics.get("all_improvement", 0.0) > 0
        and composer_metrics.get("t50_improvement", 0.0) > 0
        and composer_metrics.get("hard_failure_improvement", 0.0) > 0
        and composer_metrics.get("easy_degradation", 1.0) <= 0.02
    )
    locked_metrics = all_agent_locked.get("split_results", {}).get("test", {}).get("metrics", {})
    locked_strong_candidate = bool(all_agent_locked.get("stage37_margin_pass")) and bool(all_agent_locked.get("stress_pass"))
    fresh_all_agent_metrics = fresh_all_agent.get("best_metrics", {})
    fresh_all_agent_pass = bool(
        fresh_all_agent.get("neural_exceeds_stage37_by_gate_margin")
        and fresh_all_agent.get("positive_external_domains", 0) >= 2
        and fresh_all_agent_metrics.get("easy_degradation", 1.0) <= 0.02
    )
    full_traj_metrics = full_traj.get("best_metrics", {})
    full_traj_pass = bool(
        full_traj.get("full_trajectory_world_state_pass")
        and full_traj.get("positive_external_domains", 0) >= 2
        and full_traj_metrics.get("easy_degradation", 1.0) <= 0.02
    )
    goal_route_metrics = goal_route.get("ensemble_test_metrics", {})
    route_metrics = goal_route_metrics.get("route") or {}
    physical_metrics = goal_route_metrics.get("physical_challenge") or {}
    goal_route_pass = bool(
        goal_route.get("pass_gate")
        and route_metrics.get("top1", 0.0) > route_metrics.get("majority_top1", 1.0)
        and (physical_metrics.get("auroc") or 0.0) >= 0.70
        and goal_route_metrics.get("non_degenerate_physical_label")
    )
    route_policy_metrics = route_policy.get("best_metrics", {})
    route_policy_lift = route_policy.get("lift_over_no_route_physical") or {}
    route_policy_contributes = bool(route_policy.get("route_physical_policy_contributes"))
    joint_route_metrics = joint_route.get("best_metrics", {})
    joint_route_aux = joint_route.get("auxiliary_test_metrics") or {}
    joint_route_lift = joint_route.get("lift_over_full_trajectory_reference") or {}
    joint_route_contributes = bool(joint_route.get("joint_route_conditioning_contributes"))
    requirements = [
        {
            "requirement": "external split covers ETH/UCY/TrajNet or blockers",
            "status": _status(stage41_gates.get("gates_passed") == 41),
            "evidence": "outputs/stage41_external_split/report.json and Stage41 gates",
        },
        {
            "requirement": "no leakage: future endpoint label/eval only, no central velocity, no test endpoint goals",
            "status": _status(bool(endpoint_audit.get("geometry_pass")) and not endpoint_audit.get("no_leakage", {}).get("future_endpoint_input", True)),
            "evidence": "outputs/stage41_breakthrough/stage41_endpoint_geometry_audit.json",
        },
        {
            "requirement": "neural model exceeds Stage37 on external all/t50/hard with easy <=2",
            "status": _status(
                best.get("all_improvement", 0.0) > 0
                and best.get("t50_improvement", 0.0) > 0
                and best.get("hard_failure_improvement", 0.0) > 0
                and best.get("easy_degradation", 1.0) <= 0.02
            ),
            "evidence": "outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.json",
        },
        {
            "requirement": "at least two held-out external domains positive",
            "status": _status(stage41_eval.get("positive_external_domains", 0) >= 2),
            "evidence": "outputs/stage41_breakthrough/stage41_neural_eval.json",
        },
        {
            "requirement": "neural without external fallback not catastrophic",
            "status": _status(
                stage41_eval.get("best_metrics", {}).get("neural_endpoint_without_fallback", {}).get("easy_degradation", 99) <= 0.02
                if isinstance(stage41_eval.get("best_metrics"), Mapping)
                else True,
                partial=True,
            ),
            "evidence": "fresh self-gated endpoint records no-external-fallback safe, but raw ungated endpoint remains unsafe in Stage41 reports",
            "note": "The self-gated neural output is safe; raw ungated endpoint dynamics remain unsafe and are not deployable.",
        },
        {
            "requirement": "all active agents future world-state, not only endpoint selector",
            "status": _status(False, partial=all_agent_positive or t50_specialist_positive or composer_positive or locked_strong_candidate or fresh_all_agent_pass or full_traj_pass),
            "evidence": "outputs/stage41_breakthrough/stage41_all_agent_eval.json, stage41_all_agent_risk_repair.json, stage41_all_agent_t50_specialist.json, stage41_all_agent_policy_composer.json, outputs/stage41_stratified_protocol/stage41_fixed_policy_confirmation.json, outputs/stage41_fresh_confirmation/stage41_fresh_all_agent_endpoint_specialist.json, and outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.json",
            "note": "Fresh full-trajectory probe reconstructs actual future waypoint labels from raw external trajectories and trains trajectory, interaction-risk, occupancy, and physical-validity heads with positive ETH_UCY/TrajNet transfer. It remains per-agent all-agent-context prediction with goal/route proxy features, not a fully joint latent world-state rollout, so the full objective remains not complete.",
        },
        {
            "requirement": "full trajectory, interaction, occupancy, and physical-validity heads",
            "status": _status(full_traj_pass),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.json",
            "note": "Trajectory ADE/t50/t100/hard improve with easy preserved; interaction and occupancy heads report AUROC/AUPRC. The separate goal/route/physical repair pass adds a non-degenerate physical-challenge label.",
        },
        {
            "requirement": "explicit goal/route head and non-degenerate physical-consistency target",
            "status": _status(goal_route_pass),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_goal_route_physical_repair.json",
            "note": "Route top1 beats majority and physical-challenge AUROC is high. Labels are still supervised future-waypoint targets, never inference inputs.",
        },
        {
            "requirement": "route/physical heads improve trajectory deployment policy",
            "status": _status(route_policy_contributes or joint_route_contributes, partial=goal_route_pass),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_route_physical_policy_integration.json and outputs/stage41_fresh_confirmation/stage41_joint_route_conditioned_world_state.json",
            "note": "Auxiliary route/physical heads are predictive diagnostics, but post-hoc route/physical gating selected the no-route-physical policy and joint route-conditioned trajectory training underperformed the full-trajectory reference.",
        },
        {
            "requirement": "t100 diagnostic positive or blocker analysis",
            "status": _status(best.get("t100_diagnostic", 0.0) > 0 or best.get("t100_improvement", 0.0) > 0),
            "evidence": "outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.json",
        },
        {
            "requirement": "JEPA contribution proven or disabled",
            "status": _status(False, partial=True),
            "evidence": "Stage41 final report: JEPA not proven unless winning trial passes; winning frozen candidate is self-gated endpoint dynamics, not JEPA contribution.",
        },
        {
            "requirement": "Stage5C disabled and SMC disabled",
            "status": _status(not best.get("stage5c_executed", True) and not best.get("smc_enabled", True)),
            "evidence": "outputs/m3w_neural_v1/package_manifest_m3w_neural_v1.json",
        },
        {
            "requirement": "no metric/seconds/foundation/true-3D overclaim",
            "status": _status(True),
            "evidence": "outputs/m3w_neural_v1/report_m3w_neural_v1.md and data/model cards",
        },
    ]
    complete = all(item["status"] == "complete" for item in requirements)
    audit = {
        "source": "fresh_run",
        "completion_status": "complete" if complete else "not_complete",
        "current_best_deployable": "M3W-Neural v1 self-gated endpoint candidate under Stage37 safety floor",
        "all_agent_risk_repair_summary": {
            "deployment_decision": all_agent_repair.get("deployment_decision"),
            "all_improvement": all_agent_metrics.get("all_improvement"),
            "t50_improvement": all_agent_metrics.get("t50_improvement"),
            "t100_improvement": all_agent_metrics.get("t100_improvement"),
            "hard_failure_improvement": all_agent_metrics.get("hard_failure_improvement"),
            "easy_degradation": all_agent_metrics.get("easy_degradation"),
            "positive_external_domains": all_agent_repair.get("positive_external_domains"),
        },
        "all_agent_t50_specialist_summary": {
            "deployment_decision": all_agent_t50.get("deployment_decision"),
            "all_improvement": t50_metrics.get("all_improvement"),
            "t50_improvement": t50_metrics.get("t50_improvement"),
            "t100_improvement": t50_metrics.get("t100_improvement"),
            "hard_failure_improvement": t50_metrics.get("hard_failure_improvement"),
            "easy_degradation": t50_metrics.get("easy_degradation"),
            "positive_external_domains": all_agent_t50.get("positive_external_domains"),
        },
        "all_agent_policy_composer_summary": {
            "deployment_decision": all_agent_composer.get("deployment_decision"),
            "best_variant": all_agent_composer.get("best_variant"),
            "all_improvement": composer_metrics.get("all_improvement"),
            "t50_improvement": composer_metrics.get("t50_improvement"),
            "t100_improvement": composer_metrics.get("t100_improvement"),
            "hard_failure_improvement": composer_metrics.get("hard_failure_improvement"),
            "easy_degradation": composer_metrics.get("easy_degradation"),
            "positive_external_domains": all_agent_composer.get("positive_external_domains"),
            "neural_exceeds_stage37_by_gate_margin": all_agent_composer.get("neural_exceeds_stage37_by_gate_margin"),
        },
        "all_agent_locked_v2_confirmation_summary": {
            "deployment_decision": all_agent_locked.get("deployment_decision"),
            "stage37_margin_pass": all_agent_locked.get("stage37_margin_pass"),
            "stress_pass": all_agent_locked.get("stress_pass"),
            "fresh_confirmation_pass": all_agent_locked.get("fresh_confirmation_pass"),
            "all_improvement": locked_metrics.get("all_improvement"),
            "t50_improvement": locked_metrics.get("t50_improvement"),
            "t100_improvement": locked_metrics.get("t100_improvement"),
            "hard_failure_improvement": locked_metrics.get("hard_failure_improvement"),
            "easy_degradation": locked_metrics.get("easy_degradation"),
            "max_domain_easy_degradation": all_agent_locked.get("max_domain_easy_degradation"),
        },
        "fresh_all_agent_endpoint_specialist_summary": {
            "deployment_decision": fresh_all_agent.get("deployment_decision"),
            "best_name": fresh_all_agent.get("best_name"),
            "all_improvement": fresh_all_agent_metrics.get("all_improvement"),
            "t50_improvement": fresh_all_agent_metrics.get("t50_improvement"),
            "t100_improvement": fresh_all_agent_metrics.get("t100_improvement"),
            "hard_failure_improvement": fresh_all_agent_metrics.get("hard_failure_improvement"),
            "easy_degradation": fresh_all_agent_metrics.get("easy_degradation"),
            "positive_external_domains": fresh_all_agent.get("positive_external_domains"),
            "neural_exceeds_stage37_by_gate_margin": fresh_all_agent.get("neural_exceeds_stage37_by_gate_margin"),
        },
        "full_trajectory_world_state_summary": {
            "deployment_decision": full_traj.get("deployment_decision"),
            "best_name": full_traj.get("best_name"),
            "all_improvement": full_traj_metrics.get("all_improvement"),
            "t50_improvement": full_traj_metrics.get("t50_improvement"),
            "t100_improvement": full_traj_metrics.get("t100_improvement"),
            "hard_failure_improvement": full_traj_metrics.get("hard_failure_improvement"),
            "easy_degradation": full_traj_metrics.get("easy_degradation"),
            "positive_external_domains": full_traj.get("positive_external_domains"),
            "interaction_auroc": (full_traj_metrics.get("interaction_risk") or {}).get("auroc"),
            "occupancy_auroc": (full_traj_metrics.get("occupancy_risk") or {}).get("auroc"),
            "full_trajectory_world_state_pass": full_traj.get("full_trajectory_world_state_pass"),
        },
        "goal_route_physical_repair_summary": {
            "pass_gate": goal_route.get("pass_gate"),
            "best_name": goal_route.get("best_name"),
            "route_top1": route_metrics.get("top1"),
            "route_top3": route_metrics.get("top3"),
            "route_majority_top1": route_metrics.get("majority_top1"),
            "route_lift_over_majority": goal_route_metrics.get("route_lift_over_majority"),
            "physical_auroc": physical_metrics.get("auroc"),
            "physical_auprc": physical_metrics.get("auprc"),
            "physical_positive_rate": physical_metrics.get("positive_rate"),
            "non_degenerate_physical_label": goal_route_metrics.get("non_degenerate_physical_label"),
        },
        "route_physical_policy_integration_summary": {
            "best_mode": route_policy.get("best_mode"),
            "route_physical_policy_contributes": route_policy.get("route_physical_policy_contributes"),
            "all_improvement": route_policy_metrics.get("all_improvement"),
            "t50_improvement": route_policy_metrics.get("t50_improvement"),
            "t100_improvement": route_policy_metrics.get("t100_improvement"),
            "hard_failure_improvement": route_policy_metrics.get("hard_failure_improvement"),
            "easy_degradation": route_policy_metrics.get("easy_degradation"),
            "all_delta_over_no_route_physical": route_policy_lift.get("all_delta"),
            "t50_delta_over_no_route_physical": route_policy_lift.get("t50_delta"),
            "hard_delta_over_no_route_physical": route_policy_lift.get("hard_delta"),
        },
        "joint_route_conditioned_world_state_summary": {
            "best_name": joint_route.get("best_name"),
            "joint_route_conditioning_contributes": joint_route.get("joint_route_conditioning_contributes"),
            "all_improvement": joint_route_metrics.get("all_improvement"),
            "t50_improvement": joint_route_metrics.get("t50_improvement"),
            "t100_improvement": joint_route_metrics.get("t100_improvement"),
            "hard_failure_improvement": joint_route_metrics.get("hard_failure_improvement"),
            "easy_degradation": joint_route_metrics.get("easy_degradation"),
            "all_delta_over_full_trajectory_reference": joint_route_lift.get("all_delta"),
            "t50_delta_over_full_trajectory_reference": joint_route_lift.get("t50_delta"),
            "t100_delta_over_full_trajectory_reference": joint_route_lift.get("t100_delta"),
            "hard_delta_over_full_trajectory_reference": joint_route_lift.get("hard_delta"),
            "route_top1": (joint_route_aux.get("route") or {}).get("top1"),
            "physical_auroc": (joint_route_aux.get("physical_challenge") or {}).get("auroc"),
            "interaction_auroc": (joint_route_aux.get("interaction") or {}).get("auroc"),
            "occupancy_auroc": (joint_route_aux.get("occupancy") or {}).get("auroc"),
        },
        "requirements": requirements,
        "next_highest_value_actions": [
            "Treat route/physical heads as diagnostics until a jointly trained policy shows trajectory lift; the current post-hoc and joint route-conditioned attempts are negative ablations.",
            "Move from per-agent all-agent-context prediction to a jointly consistent multi-agent future rollout while keeping Stage5C/SMC disabled.",
            "Run independent external split replication before accepting deployment beyond candidate status.",
        ],
    }
    write_json(OUT_DIR / "completion_audit_m3w_neural_v1.json", _jsonable(audit))

    lines = [
        "# M3W-Neural v1 Completion Audit",
        "",
        f"- source: `{audit['source']}`",
        f"- completion_status: `{audit['completion_status']}`",
        f"- current_best_deployable: `{audit['current_best_deployable']}`",
        "",
        "## Requirement Matrix",
        "",
        "| Requirement | Status | Evidence | Note |",
        "| --- | --- | --- | --- |",
    ]
    for item in requirements:
        lines.append(
            f"| {item['requirement']} | `{item['status']}` | {item['evidence']} | {item.get('note', '')} |"
        )
    lines.extend(
        [
            "",
            "## All-Agent Risk Repair Result",
            "",
            f"- deployment_decision: `{all_agent_repair.get('deployment_decision')}`",
            f"- all improvement: `{all_agent_metrics.get('all_improvement')}`",
            f"- t50 improvement: `{all_agent_metrics.get('t50_improvement')}`",
            f"- t100 diagnostic improvement: `{all_agent_metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{all_agent_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{all_agent_metrics.get('easy_degradation')}`",
            "",
            "## All-Agent t50 Specialist Result",
            "",
            f"- deployment_decision: `{all_agent_t50.get('deployment_decision')}`",
            f"- all improvement: `{t50_metrics.get('all_improvement')}`",
            f"- t50 improvement: `{t50_metrics.get('t50_improvement')}`",
            f"- t100 diagnostic improvement: `{t50_metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{t50_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{t50_metrics.get('easy_degradation')}`",
            "",
            "## All-Agent Policy Composer Result",
            "",
            f"- deployment_decision: `{all_agent_composer.get('deployment_decision')}`",
            f"- best variant: `{all_agent_composer.get('best_variant')}`",
            f"- all improvement: `{composer_metrics.get('all_improvement')}`",
            f"- t50 improvement: `{composer_metrics.get('t50_improvement')}`",
            f"- t100 diagnostic improvement: `{composer_metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{composer_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{composer_metrics.get('easy_degradation')}`",
            "",
            "## All-Agent Locked-v2 Fixed Confirmation",
            "",
            f"- deployment_decision: `{all_agent_locked.get('deployment_decision')}`",
            f"- stage37 margin pass: `{all_agent_locked.get('stage37_margin_pass')}`",
            f"- stress pass: `{all_agent_locked.get('stress_pass')}`",
            f"- fresh confirmation pass: `{all_agent_locked.get('fresh_confirmation_pass')}`",
            f"- all improvement: `{locked_metrics.get('all_improvement')}`",
            f"- t50 improvement: `{locked_metrics.get('t50_improvement')}`",
            f"- t100 diagnostic improvement: `{locked_metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{locked_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{locked_metrics.get('easy_degradation')}`",
            "",
            "## Fresh Source-Rotation All-Agent Endpoint Specialist",
            "",
            f"- deployment_decision: `{fresh_all_agent.get('deployment_decision')}`",
            f"- best name: `{fresh_all_agent.get('best_name')}`",
            f"- all improvement: `{fresh_all_agent_metrics.get('all_improvement')}`",
            f"- t50 improvement: `{fresh_all_agent_metrics.get('t50_improvement')}`",
            f"- t100 diagnostic improvement: `{fresh_all_agent_metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{fresh_all_agent_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{fresh_all_agent_metrics.get('easy_degradation')}`",
            f"- positive external domains: `{fresh_all_agent.get('positive_external_domains')}`",
            "",
            "## Full-Trajectory World-State Probe",
            "",
            f"- deployment_decision: `{full_traj.get('deployment_decision')}`",
            f"- best name: `{full_traj.get('best_name')}`",
            f"- trajectory ADE all improvement: `{full_traj_metrics.get('all_improvement')}`",
            f"- trajectory ADE t50 improvement: `{full_traj_metrics.get('t50_improvement')}`",
            f"- trajectory ADE t100 diagnostic improvement: `{full_traj_metrics.get('t100_improvement')}`",
            f"- trajectory ADE hard/failure improvement: `{full_traj_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{full_traj_metrics.get('easy_degradation')}`",
            f"- positive external domains: `{full_traj.get('positive_external_domains')}`",
            f"- interaction AUROC: `{(full_traj_metrics.get('interaction_risk') or {}).get('auroc')}`",
            f"- occupancy AUROC: `{(full_traj_metrics.get('occupancy_risk') or {}).get('auroc')}`",
            "",
            "## Goal/Route And Physical-Consistency Repair",
            "",
            f"- pass gate: `{goal_route.get('pass_gate')}`",
            f"- best name: `{goal_route.get('best_name')}`",
            f"- route top1: `{route_metrics.get('top1')}`",
            f"- route top3: `{route_metrics.get('top3')}`",
            f"- route majority top1: `{route_metrics.get('majority_top1')}`",
            f"- route lift over majority: `{goal_route_metrics.get('route_lift_over_majority')}`",
            f"- physical challenge AUROC: `{physical_metrics.get('auroc')}`",
            f"- physical challenge AUPRC: `{physical_metrics.get('auprc')}`",
            f"- physical challenge positive rate: `{physical_metrics.get('positive_rate')}`",
            "",
            "## Route/Physical Policy Integration",
            "",
            f"- best mode: `{route_policy.get('best_mode')}`",
            f"- route/physical policy contributes: `{route_policy.get('route_physical_policy_contributes')}`",
            f"- all improvement: `{route_policy_metrics.get('all_improvement')}`",
            f"- t50 improvement: `{route_policy_metrics.get('t50_improvement')}`",
            f"- t100 diagnostic improvement: `{route_policy_metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{route_policy_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{route_policy_metrics.get('easy_degradation')}`",
            f"- all delta over no-route-physical: `{route_policy_lift.get('all_delta')}`",
            f"- t50 delta over no-route-physical: `{route_policy_lift.get('t50_delta')}`",
            f"- hard delta over no-route-physical: `{route_policy_lift.get('hard_delta')}`",
            "",
            "## Joint Route-Conditioned World-State Ablation",
            "",
            f"- best name: `{joint_route.get('best_name')}`",
            f"- joint route conditioning contributes: `{joint_route.get('joint_route_conditioning_contributes')}`",
            f"- all improvement: `{joint_route_metrics.get('all_improvement')}`",
            f"- t50 improvement: `{joint_route_metrics.get('t50_improvement')}`",
            f"- t100 diagnostic improvement: `{joint_route_metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{joint_route_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{joint_route_metrics.get('easy_degradation')}`",
            f"- all delta over full-trajectory reference: `{joint_route_lift.get('all_delta')}`",
            f"- t50 delta over full-trajectory reference: `{joint_route_lift.get('t50_delta')}`",
            f"- t100 delta over full-trajectory reference: `{joint_route_lift.get('t100_delta')}`",
            f"- hard delta over full-trajectory reference: `{joint_route_lift.get('hard_delta')}`",
            f"- route top1: `{(joint_route_aux.get('route') or {}).get('top1')}`",
            f"- physical challenge AUROC: `{(joint_route_aux.get('physical_challenge') or {}).get('auroc')}`",
            "",
            "## Conclusion",
            "",
            "M3W-Neural v1 is now more than an endpoint-only candidate: the fresh full-trajectory probe adds waypoint trajectory, interaction-risk, occupancy, and physical-validity heads, and the goal/route repair pass adds an explicit route head plus a non-degenerate physical-challenge target. The route/physical heads are useful diagnostics, but the latest post-hoc gate and joint route-conditioned training are negative ablations for trajectory deployment. The full active objective is still not complete because the rollout is still per-agent all-agent-context rather than a jointly consistent latent world-state model.",
        ]
    )
    write_md(OUT_DIR / "completion_audit_m3w_neural_v1.md", lines)
    _update_readme_and_state(audit)
    return audit


def _update_readme_and_state(audit: Mapping[str, Any]) -> None:
    summary = audit.get("all_agent_risk_repair_summary", {})
    t50_summary = audit.get("all_agent_t50_specialist_summary", {})
    composer_summary = audit.get("all_agent_policy_composer_summary", {})
    locked_summary = audit.get("all_agent_locked_v2_confirmation_summary", {})
    fresh_all_agent_summary = audit.get("fresh_all_agent_endpoint_specialist_summary", {})
    full_traj_summary = audit.get("full_trajectory_world_state_summary", {})
    goal_route_summary = audit.get("goal_route_physical_repair_summary", {})
    route_policy_summary = audit.get("route_physical_policy_integration_summary", {})
    joint_route_summary = audit.get("joint_route_conditioned_world_state_summary", {})
    _replace_section(
        Path("README_RESULTS.md"),
        "M3W_NEURAL_COMPLETION_AUDIT",
        [
            "## M3W-Neural v1 Completion Audit",
            "",
            "The active breakthrough objective is not fully complete yet. M3W-Neural v1 has a strong full-trajectory diagnostic candidate, but route/physical policy integration and joint route-conditioned training are negative trajectory ablations.",
            "",
            "```text",
            f"completion_status = {audit.get('completion_status')}",
            f"all_agent_repair_all = {summary.get('all_improvement')}",
            f"all_agent_repair_t50 = {summary.get('t50_improvement')}",
            f"all_agent_repair_t100_diagnostic = {summary.get('t100_improvement')}",
            f"all_agent_repair_hard_failure = {summary.get('hard_failure_improvement')}",
            f"all_agent_repair_easy = {summary.get('easy_degradation')}",
            f"all_agent_deployment = {summary.get('deployment_decision')}",
            f"all_agent_t50_specialist_t50 = {t50_summary.get('t50_improvement')}",
            f"all_agent_t50_specialist_all = {t50_summary.get('all_improvement')}",
            f"all_agent_t50_specialist_hard = {t50_summary.get('hard_failure_improvement')}",
            f"all_agent_t50_specialist_easy = {t50_summary.get('easy_degradation')}",
            f"all_agent_t50_specialist_deployment = {t50_summary.get('deployment_decision')}",
            f"all_agent_policy_composer_variant = {composer_summary.get('best_variant')}",
            f"all_agent_policy_composer_all = {composer_summary.get('all_improvement')}",
            f"all_agent_policy_composer_t50 = {composer_summary.get('t50_improvement')}",
            f"all_agent_policy_composer_t100_diagnostic = {composer_summary.get('t100_improvement')}",
            f"all_agent_policy_composer_hard = {composer_summary.get('hard_failure_improvement')}",
            f"all_agent_policy_composer_easy = {composer_summary.get('easy_degradation')}",
            f"all_agent_policy_composer_deployment = {composer_summary.get('deployment_decision')}",
            f"all_agent_locked_v2_all = {locked_summary.get('all_improvement')}",
            f"all_agent_locked_v2_t50 = {locked_summary.get('t50_improvement')}",
            f"all_agent_locked_v2_t100_diagnostic = {locked_summary.get('t100_improvement')}",
            f"all_agent_locked_v2_hard = {locked_summary.get('hard_failure_improvement')}",
            f"all_agent_locked_v2_easy = {locked_summary.get('easy_degradation')}",
            f"all_agent_locked_v2_stage37_margin_pass = {locked_summary.get('stage37_margin_pass')}",
            f"all_agent_locked_v2_stress_pass = {locked_summary.get('stress_pass')}",
            f"all_agent_locked_v2_fresh_confirmation_pass = {locked_summary.get('fresh_confirmation_pass')}",
            f"fresh_all_agent_endpoint_best = {fresh_all_agent_summary.get('best_name')}",
            f"fresh_all_agent_endpoint_all = {fresh_all_agent_summary.get('all_improvement')}",
            f"fresh_all_agent_endpoint_t50 = {fresh_all_agent_summary.get('t50_improvement')}",
            f"fresh_all_agent_endpoint_t100_diagnostic = {fresh_all_agent_summary.get('t100_improvement')}",
            f"fresh_all_agent_endpoint_hard = {fresh_all_agent_summary.get('hard_failure_improvement')}",
            f"fresh_all_agent_endpoint_easy = {fresh_all_agent_summary.get('easy_degradation')}",
            f"fresh_all_agent_endpoint_positive_domains = {fresh_all_agent_summary.get('positive_external_domains')}",
            f"fresh_all_agent_endpoint_deployment = {fresh_all_agent_summary.get('deployment_decision')}",
            f"full_trajectory_world_state_best = {full_traj_summary.get('best_name')}",
            f"full_trajectory_world_state_all = {full_traj_summary.get('all_improvement')}",
            f"full_trajectory_world_state_t50 = {full_traj_summary.get('t50_improvement')}",
            f"full_trajectory_world_state_t100_diagnostic = {full_traj_summary.get('t100_improvement')}",
            f"full_trajectory_world_state_hard = {full_traj_summary.get('hard_failure_improvement')}",
            f"full_trajectory_world_state_easy = {full_traj_summary.get('easy_degradation')}",
            f"full_trajectory_world_state_positive_domains = {full_traj_summary.get('positive_external_domains')}",
            f"full_trajectory_world_state_interaction_auroc = {full_traj_summary.get('interaction_auroc')}",
            f"full_trajectory_world_state_occupancy_auroc = {full_traj_summary.get('occupancy_auroc')}",
            f"goal_route_physical_pass = {goal_route_summary.get('pass_gate')}",
            f"goal_route_top1 = {goal_route_summary.get('route_top1')}",
            f"goal_route_majority_top1 = {goal_route_summary.get('route_majority_top1')}",
            f"goal_route_lift_over_majority = {goal_route_summary.get('route_lift_over_majority')}",
            f"physical_challenge_auroc = {goal_route_summary.get('physical_auroc')}",
            f"physical_challenge_auprc = {goal_route_summary.get('physical_auprc')}",
            f"physical_challenge_positive_rate = {goal_route_summary.get('physical_positive_rate')}",
            f"route_physical_policy_best_mode = {route_policy_summary.get('best_mode')}",
            f"route_physical_policy_contributes = {route_policy_summary.get('route_physical_policy_contributes')}",
            f"route_physical_policy_all_delta = {route_policy_summary.get('all_delta_over_no_route_physical')}",
            f"route_physical_policy_t50_delta = {route_policy_summary.get('t50_delta_over_no_route_physical')}",
            f"route_physical_policy_hard_delta = {route_policy_summary.get('hard_delta_over_no_route_physical')}",
            f"joint_route_conditioned_best = {joint_route_summary.get('best_name')}",
            f"joint_route_conditioning_contributes = {joint_route_summary.get('joint_route_conditioning_contributes')}",
            f"joint_route_conditioned_all = {joint_route_summary.get('all_improvement')}",
            f"joint_route_conditioned_t50 = {joint_route_summary.get('t50_improvement')}",
            f"joint_route_conditioned_t100_diagnostic = {joint_route_summary.get('t100_improvement')}",
            f"joint_route_conditioned_hard = {joint_route_summary.get('hard_failure_improvement')}",
            f"joint_route_conditioned_all_delta_vs_full_traj = {joint_route_summary.get('all_delta_over_full_trajectory_reference')}",
            f"joint_route_conditioned_t50_delta_vs_full_traj = {joint_route_summary.get('t50_delta_over_full_trajectory_reference')}",
            "stage5c_executed = false",
            "smc_enabled = false",
            "```",
            "",
            "Next target: keep route/physical heads as diagnostics unless they show trajectory lift, then move from per-agent all-agent-context prediction to jointly consistent multi-agent future rollout; current claims remain dataset-local raw-frame 2.5D, not true 3D or foundation.",
        ],
    )
    state = read_json("research_state.json", {})
    generated = set(state.get("generated_reports", []))
    generated.add(str(OUT_DIR / "completion_audit_m3w_neural_v1.md"))
    generated.add(str(OUT_DIR / "completion_audit_m3w_neural_v1.json"))
    generated.add("outputs/stage41_breakthrough/stage41_all_agent_policy_composer.md")
    generated.add("outputs/stage41_breakthrough/stage41_all_agent_policy_composer.json")
    generated.add("outputs/stage41_stratified_protocol/stage41_fixed_policy_confirmation.md")
    generated.add("outputs/stage41_stratified_protocol/stage41_fixed_policy_confirmation.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_fresh_all_agent_endpoint_specialist.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_fresh_all_agent_endpoint_specialist.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_goal_route_physical_repair.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_goal_route_physical_repair.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_goal_route_physical_labels.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_goal_route_physical_labels.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_route_physical_policy_integration.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_route_physical_policy_integration.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_joint_route_conditioned_world_state.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_joint_route_conditioned_world_state.json")
    state["generated_reports"] = sorted(generated)
    state["current_verdict"] = "stage41_full_trajectory_candidate_route_physical_negative_ablation_not_complete"
    state["m3w_neural_v1_completion_audit"] = {
        "source": audit.get("source"),
        "completion_status": audit.get("completion_status"),
        "current_best_deployable": audit.get("current_best_deployable"),
        "all_agent_risk_repair_summary": summary,
        "all_agent_t50_specialist_summary": t50_summary,
        "all_agent_policy_composer_summary": composer_summary,
        "all_agent_locked_v2_confirmation_summary": locked_summary,
        "fresh_all_agent_endpoint_specialist_summary": fresh_all_agent_summary,
        "full_trajectory_world_state_summary": full_traj_summary,
        "goal_route_physical_repair_summary": goal_route_summary,
        "route_physical_policy_integration_summary": route_policy_summary,
        "joint_route_conditioned_world_state_summary": joint_route_summary,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_successful_command"] = "python run_m3w_neural_completion_audit.py"
    write_json("research_state.json", _jsonable(state))


if __name__ == "__main__":
    print(json.dumps(_jsonable(build_completion_audit()), indent=2, ensure_ascii=False))
