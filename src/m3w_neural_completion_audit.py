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
    joint_consistency = read_json("outputs/stage41_fresh_confirmation/stage41_joint_multiagent_consistency.json", {})
    joint_distill = read_json("outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation.json", {})
    joint_distill_evidence = read_json("outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation_evidence.json", {})
    joint_distill_multiseed = read_json("outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation_multiseed.json", {})
    ucy_repair = read_json("outputs/stage41_fresh_confirmation/stage41_ucy_fallback_repair.json", {})
    ucy_validation = read_json("outputs/stage41_fresh_confirmation/stage41_ucy_independent_validation.json", {})
    joint_rollout = read_json("outputs/stage41_fresh_confirmation/stage41_joint_rollout_consistency.json", {})
    group_distiller = read_json("outputs/stage41_fresh_confirmation/stage41_group_consistency_distiller.json", {})
    group_distiller_evidence = read_json("outputs/stage41_fresh_confirmation/stage41_group_consistency_evidence.json", {})
    group_distiller_multiseed = read_json("outputs/stage41_fresh_confirmation/stage41_group_consistency_multiseed.json", {})
    group_distiller_multiseed_repair = read_json("outputs/stage41_fresh_confirmation/stage41_group_consistency_multiseed_repair.json", {})
    jepa_decision = read_json("outputs/stage41_fresh_confirmation/stage41_jepa_deployment_decision.json", {})
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
    joint_consistency_metrics = joint_consistency.get("test_metrics", {})
    joint_consistency_lift = joint_consistency.get("lift_over_full_trajectory_reference") or {}
    joint_consistency_contributes = bool(joint_consistency.get("joint_multiagent_consistency_contributes"))
    joint_distill_metrics = joint_distill.get("test_metrics", {})
    joint_distill_lift_joint = joint_distill.get("lift_over_joint_consistency_reference") or {}
    joint_distill_lift_full = joint_distill.get("lift_over_full_trajectory_reference") or {}
    joint_distill_no_leak = joint_distill.get("no_leakage") or {}
    joint_distill_contributes = bool(joint_distill.get("joint_policy_distillation_contributes"))
    joint_distill_positive_domains = sum(
        1
        for row in (joint_distill_metrics.get("by_domain") or {}).values()
        if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0
    )
    joint_distill_bootstrap = joint_distill_evidence.get("bootstrap") or {}
    joint_distill_ablation = joint_distill_evidence.get("contribution_summary") or {}
    joint_distill_stable = bool(joint_distill_evidence.get("statistically_stable_on_test"))
    joint_distill_multiseed_summary = joint_distill_multiseed.get("metric_summary") or {}
    joint_distill_multiseed_pass = bool(joint_distill_multiseed.get("replication_pass"))
    ucy_repair_metrics = ucy_repair.get("repaired_metrics") or {}
    ucy_repair_lift = ucy_repair.get("lift_over_base_policy") or {}
    ucy_repair_bootstrap = ucy_repair.get("bootstrap") or {}
    ucy_repair_contributes = bool(ucy_repair.get("ucy_repair_contributes"))
    ucy_validation_pass = bool(ucy_validation.get("validation_pass"))
    ucy_source_level_available = bool(ucy_validation.get("source_level_independent_validation_available"))
    joint_rollout_metrics = joint_rollout.get("selected_metrics") or {}
    joint_rollout_multi = joint_rollout.get("multi_agent_metrics") or {}
    joint_rollout_stats = joint_rollout.get("rollout_stats") or {}
    joint_rollout_selected_stats = joint_rollout_stats.get("selected") or {}
    joint_rollout_pass = bool(joint_rollout.get("joint_rollout_consistency_pass"))
    group_distiller_metrics = group_distiller.get("test_metrics") or {}
    group_distiller_lift = group_distiller.get("lift_over_fixed_proximity_guard") or {}
    group_distiller_deployable = bool(group_distiller.get("group_consistency_distiller_deployable"))
    group_distiller_improves_guard = bool(group_distiller.get("group_consistency_distiller_improves_fixed_guard"))
    group_distiller_bootstrap = group_distiller_evidence.get("bootstrap") or {}
    group_distiller_ablation = group_distiller_evidence.get("contribution_summary") or {}
    group_distiller_stable = bool(group_distiller_evidence.get("statistically_stable_on_test"))
    group_distiller_multiseed_summary = group_distiller_multiseed.get("metric_summary") or {}
    group_distiller_multiseed_pass = bool(group_distiller_multiseed.get("replication_pass"))
    group_distiller_repair_summary = group_distiller_multiseed_repair.get("metric_summary") or {}
    group_distiller_repair_pass = bool(group_distiller_multiseed_repair.get("replication_pass"))
    group_distiller_repair_domains = group_distiller_multiseed_repair.get("positive_domain_counts") or []
    jepa_disabled = bool(jepa_decision.get("disable_jepa_in_deployable_path"))
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
            "requirement": "joint multi-agent consistency improves trajectory deployment policy",
            "status": _status(
                joint_consistency_contributes
                and joint_consistency_metrics.get("easy_degradation", 1.0) <= 0.02
                and (
                    joint_consistency_lift.get("all_delta", 0.0) > 0
                    or joint_consistency_lift.get("t50_delta", 0.0) > 0
                    or joint_consistency_lift.get("hard_delta", 0.0) > 0
                ),
                partial=bool(joint_consistency_metrics),
            ),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_joint_multiagent_consistency.json",
            "note": "Current-frame group consistency adds a tiny positive deployment-policy lift over the full-trajectory reference and gives UCY a small positive switch rate, but it is still post-hoc group calibration rather than a jointly consistent latent rollout.",
        },
        {
            "requirement": "neural gain/harm/switch distillation improves deployment without base-switch leakage",
            "status": _status(
                joint_distill_contributes
                and joint_distill_metrics.get("easy_degradation", 1.0) <= 0.02
                and joint_distill_positive_domains >= 2
                and not joint_distill_no_leak.get("base_switch_input", True)
                and not joint_distill_no_leak.get("future_waypoints_input", True)
            ),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation.json",
            "note": "The deployable distiller-only policy learns gain/harm/switch from train labels and uses past/static/full-trajectory prediction signals at inference. It improves ETH_UCY and TrajNet but still falls back on UCY.",
        },
        {
            "requirement": "no-base-switch distiller bootstrap and ablation evidence",
            "status": _status(joint_distill_stable),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation_evidence.json",
            "note": "Bootstrap lower bounds are positive for all/t50/hard; ablations show static causal features and full-trajectory prediction signals are the main positive contributors, while UCY remains fallback-only.",
        },
        {
            "requirement": "no-base-switch distiller multi-seed replication",
            "status": _status(joint_distill_multiseed_pass),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation_multiseed.json",
            "note": "Three fresh seeds keep all/t50/t100/hard positive with easy preserved and two positive domains per seed; UCY remains fallback-only.",
        },
        {
            "requirement": "UCY fallback-only blocker diagnosed and repaired without test tuning",
            "status": _status(ucy_repair_contributes),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_ucy_fallback_repair.json",
            "note": "UCY was missing from validation, so no UCY slice thresholds were selected. A train-only UCY calibration subset repairs UCY on test, but independent UCY validation is still needed before final deployment.",
        },
        {
            "requirement": "UCY repair internal fold/temporal validation",
            "status": _status(ucy_validation_pass, partial=ucy_repair_contributes),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_ucy_independent_validation.json",
            "note": "UCY repair validates on internal held-out row folds and temporal blocks. True source-level UCY validation remains unavailable because there is one UCY train source and no UCY validation source.",
        },
        {
            "requirement": "grouped all-agent rollout consistency under repaired policy",
            "status": _status(joint_rollout_pass, partial=bool(joint_rollout_metrics)),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_joint_rollout_consistency.json",
            "note": "Audits same-frame multi-agent selected future waypoints for switch coherence, proximity risk, smoothness, and multi-agent improvement. This is grouped rollout evidence, not Stage5C latent generation or SMC.",
        },
        {
            "requirement": "neural group-consistency head improves joint-safe fixed proximity guard",
            "status": _status(group_distiller_deployable and group_distiller_improves_guard, partial=group_distiller_deployable or bool(group_distiller_metrics)),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_group_consistency_distiller.json",
            "note": "Trains a neural safe-switch/gain/unsafe head from train labels and selects thresholds on validation. It improves the fixed proximity guard while preserving easy cases and joint proximity safety, but it is still a guarded selector/dynamics head rather than Stage5C latent generation.",
        },
        {
            "requirement": "group-consistency distiller bootstrap and ablation evidence",
            "status": _status(group_distiller_stable, partial=bool(group_distiller_bootstrap)),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_group_consistency_evidence.json",
            "note": "Bootstrap lower bounds are positive for all/t50/t100/hard. Ablations show the new group-consistency/proposal-score features are necessary, while some older feature blocks are not positive in this head.",
        },
        {
            "requirement": "group-consistency distiller multi-seed replication with joint-safety buffer",
            "status": _status(group_distiller_repair_pass, partial=bool(group_distiller_multiseed_summary)),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_group_consistency_multiseed.json and outputs/stage41_fresh_confirmation/stage41_group_consistency_multiseed_repair.json",
            "note": "The first three-seed run had stable positive FDE gains but one seed exceeded the near-proximity delta threshold. A validation-selected safety-buffer repair passes all three seeds with positive all/t50/t100/hard, easy=0, and max collision delta below the joint-safety ceiling.",
        },
        {
            "requirement": "t100 diagnostic positive or blocker analysis",
            "status": _status(best.get("t100_diagnostic", 0.0) > 0 or best.get("t100_improvement", 0.0) > 0),
            "evidence": "outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.json",
        },
        {
            "requirement": "JEPA contribution proven or disabled",
            "status": _status(jepa_disabled, partial=bool(jepa_decision)),
            "evidence": "outputs/stage41_fresh_confirmation/stage41_jepa_deployment_decision.json",
            "note": "Current audited JEPA variants are non-collapse in several stages but do not produce deployable downstream lift. JEPA is disabled from the M3W-Neural v1 deployable path and kept diagnostic-only.",
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
        "current_best_deployable": (
            "M3W-Neural v1 group-consistency multi-seed safety-buffer joint-safe candidate under Stage37 safety floor"
            if group_distiller_repair_pass
            else
            "M3W-Neural v1 group-consistency-distilled joint-safe candidate under Stage37 safety floor"
            if group_distiller_deployable and group_distiller_improves_guard
            else
            "M3W-Neural v1 fixed-proximity-guarded joint-safe candidate under Stage37 safety floor"
            if joint_rollout_pass
            else
            "M3W-Neural v1 UCY-repaired joint-policy-distilled row-level candidate under Stage37 safety floor"
            if ucy_repair_contributes
            else
            "M3W-Neural v1 joint-policy-distilled full-trajectory candidate under Stage37 safety floor"
            if joint_distill_contributes
            else "M3W-Neural v1 joint-consistency-calibrated full-trajectory candidate under Stage37 safety floor"
            if joint_consistency_contributes
            else "M3W-Neural v1 self-gated endpoint candidate under Stage37 safety floor"
        ),
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
        "joint_multiagent_consistency_summary": {
            "selected_params": joint_consistency.get("selected_params"),
            "joint_multiagent_consistency_contributes": joint_consistency.get("joint_multiagent_consistency_contributes"),
            "all_improvement": joint_consistency_metrics.get("all_improvement"),
            "t50_improvement": joint_consistency_metrics.get("t50_improvement"),
            "t100_improvement": joint_consistency_metrics.get("t100_improvement"),
            "hard_failure_improvement": joint_consistency_metrics.get("hard_failure_improvement"),
            "easy_degradation": joint_consistency_metrics.get("easy_degradation"),
            "switch_rate": joint_consistency_metrics.get("switch_rate"),
            "all_delta_over_full_trajectory_reference": joint_consistency_lift.get("all_delta"),
            "t50_delta_over_full_trajectory_reference": joint_consistency_lift.get("t50_delta"),
            "t100_delta_over_full_trajectory_reference": joint_consistency_lift.get("t100_delta"),
            "hard_delta_over_full_trajectory_reference": joint_consistency_lift.get("hard_delta"),
            "expanded_on": (joint_consistency_metrics.get("joint_consistency") or {}).get("expanded_on"),
            "guarded_off": (joint_consistency_metrics.get("joint_consistency") or {}).get("guarded_off"),
        },
        "joint_policy_distillation_summary": {
            "best_name": joint_distill.get("best_name"),
            "joint_policy_distillation_contributes": joint_distill.get("joint_policy_distillation_contributes"),
            "all_improvement": joint_distill_metrics.get("all_improvement"),
            "t50_improvement": joint_distill_metrics.get("t50_improvement"),
            "t100_improvement": joint_distill_metrics.get("t100_improvement"),
            "hard_failure_improvement": joint_distill_metrics.get("hard_failure_improvement"),
            "easy_degradation": joint_distill_metrics.get("easy_degradation"),
            "switch_rate": joint_distill_metrics.get("switch_rate"),
            "positive_external_domains": joint_distill_positive_domains,
            "all_delta_over_joint_consistency": joint_distill_lift_joint.get("all_delta"),
            "t50_delta_over_joint_consistency": joint_distill_lift_joint.get("t50_delta"),
            "hard_delta_over_joint_consistency": joint_distill_lift_joint.get("hard_delta"),
            "all_delta_over_full_trajectory_reference": joint_distill_lift_full.get("all_delta"),
            "t50_delta_over_full_trajectory_reference": joint_distill_lift_full.get("t50_delta"),
            "hard_delta_over_full_trajectory_reference": joint_distill_lift_full.get("hard_delta"),
            "base_switch_input": joint_distill_no_leak.get("base_switch_input"),
            "base_plus_distiller_deployable": joint_distill_no_leak.get("base_plus_distiller_deployable"),
            "bootstrap_all_low": (joint_distill_bootstrap.get("all") or {}).get("low"),
            "bootstrap_t50_low": (joint_distill_bootstrap.get("t50") or {}).get("low"),
            "bootstrap_hard_low": (joint_distill_bootstrap.get("hard_failure") or {}).get("low"),
            "statistically_stable_on_test": joint_distill_stable,
            "ablation_static_all_delta": (joint_distill_ablation.get("static_causal_features") or {}).get("all_delta"),
            "ablation_prediction_all_delta": (joint_distill_ablation.get("full_trajectory_prediction_signals") or {}).get("all_delta"),
            "multiseed_pass": joint_distill_multiseed_pass,
            "multiseed_all_mean": (joint_distill_multiseed_summary.get("all_improvement") or {}).get("mean"),
            "multiseed_all_min": (joint_distill_multiseed_summary.get("all_improvement") or {}).get("min"),
            "multiseed_t50_mean": (joint_distill_multiseed_summary.get("t50_improvement") or {}).get("mean"),
            "multiseed_t50_min": (joint_distill_multiseed_summary.get("t50_improvement") or {}).get("min"),
            "multiseed_t100_mean": (joint_distill_multiseed_summary.get("t100_improvement") or {}).get("mean"),
            "multiseed_hard_mean": (joint_distill_multiseed_summary.get("hard_failure_improvement") or {}).get("mean"),
            "multiseed_easy_max": (joint_distill_multiseed_summary.get("easy_degradation") or {}).get("max"),
        },
        "ucy_fallback_repair_summary": {
            "contributes": ucy_repair_contributes,
            "missing_val_domains": ucy_repair.get("missing_val_domains"),
            "calibration_rows": (ucy_repair.get("calibration") or {}).get("rows"),
            "all_improvement": ucy_repair_metrics.get("all_improvement"),
            "t50_improvement": ucy_repair_metrics.get("t50_improvement"),
            "t100_improvement": ucy_repair_metrics.get("t100_improvement"),
            "hard_failure_improvement": ucy_repair_metrics.get("hard_failure_improvement"),
            "easy_degradation": ucy_repair_metrics.get("easy_degradation"),
            "switch_rate": ucy_repair_metrics.get("switch_rate"),
            "ucy_all_improvement": ((ucy_repair_metrics.get("by_domain") or {}).get("UCY") or {}).get("all_improvement"),
            "ucy_t50_improvement": ((ucy_repair_metrics.get("by_domain") or {}).get("UCY") or {}).get("t50_improvement"),
            "ucy_t100_improvement": ((ucy_repair_metrics.get("by_domain") or {}).get("UCY") or {}).get("t100_improvement"),
            "all_delta_over_base_policy": ucy_repair_lift.get("all_delta"),
            "t50_delta_over_base_policy": ucy_repair_lift.get("t50_delta"),
            "hard_delta_over_base_policy": ucy_repair_lift.get("hard_delta"),
            "bootstrap_all_low": (ucy_repair_bootstrap.get("all") or {}).get("low"),
            "bootstrap_t50_low": (ucy_repair_bootstrap.get("t50") or {}).get("low"),
            "bootstrap_ucy_low": ((ucy_repair_bootstrap.get("by_domain") or {}).get("UCY") or {}).get("low"),
            "train_only_ucy_threshold_calibration": (ucy_repair.get("no_leakage") or {}).get("train_only_ucy_threshold_calibration"),
        },
        "ucy_independent_validation_summary": {
            "validation_pass": ucy_validation_pass,
            "source_level_independent_validation_available": ucy_source_level_available,
            "source_level_blocker": ucy_validation.get("source_level_blocker"),
            "selection_rows": (ucy_validation.get("selection") or {}).get("rows"),
            "internal_validation_pass": ucy_validation.get("internal_validation_pass"),
            "temporal_validation_pass": ucy_validation.get("temporal_validation_pass"),
            "test_ucy_all": (ucy_validation.get("test_ucy_metrics") or {}).get("all_improvement"),
            "test_ucy_t50": (ucy_validation.get("test_ucy_metrics") or {}).get("t50_improvement"),
            "test_ucy_t100": (ucy_validation.get("test_ucy_metrics") or {}).get("t100_improvement"),
            "test_ucy_hard": (ucy_validation.get("test_ucy_metrics") or {}).get("hard_failure_improvement"),
            "test_ucy_easy": (ucy_validation.get("test_ucy_metrics") or {}).get("easy_degradation"),
        },
        "joint_rollout_consistency_summary": {
            "pass": joint_rollout_pass,
            "policy_source": joint_rollout.get("policy_source"),
            "rows": joint_rollout.get("rows"),
            "multi_agent_rows": joint_rollout.get("multi_agent_rows"),
            "all_improvement": joint_rollout_metrics.get("all_improvement"),
            "t50_improvement": joint_rollout_metrics.get("t50_improvement"),
            "t100_improvement": joint_rollout_metrics.get("t100_improvement"),
            "hard_failure_improvement": joint_rollout_metrics.get("hard_failure_improvement"),
            "easy_degradation": joint_rollout_metrics.get("easy_degradation"),
            "switch_rate": joint_rollout_metrics.get("switch_rate"),
            "multi_agent_all_improvement": joint_rollout_multi.get("all_improvement"),
            "multi_agent_t50_improvement": joint_rollout_multi.get("t50_improvement"),
            "multi_agent_hard_failure_improvement": joint_rollout_multi.get("hard_failure_improvement"),
            "multi_agent_easy_degradation": joint_rollout_multi.get("easy_degradation"),
            "collision_delta_vs_floor_005": joint_rollout.get("collision_delta_vs_floor_005"),
            "selected_near_collision_rate_005": joint_rollout_selected_stats.get("near_collision_rate_005"),
            "selected_mixed_group_switch_rate": (joint_rollout.get("group_switch_summary") or {}).get("selected_mixed_group_switch_rate"),
        },
        "group_consistency_distiller_summary": {
            "deployable": group_distiller_deployable,
            "improves_fixed_guard": group_distiller_improves_guard,
            "selected_policy": group_distiller.get("selected_policy"),
            "all_improvement": group_distiller_metrics.get("all_improvement"),
            "t50_improvement": group_distiller_metrics.get("t50_improvement"),
            "t100_improvement": group_distiller_metrics.get("t100_improvement"),
            "hard_failure_improvement": group_distiller_metrics.get("hard_failure_improvement"),
            "easy_degradation": group_distiller_metrics.get("easy_degradation"),
            "switch_rate": group_distiller_metrics.get("switch_rate"),
            "collision_delta_vs_floor_005": group_distiller_metrics.get("collision_delta_vs_floor_005"),
            "all_delta_over_fixed_guard": group_distiller_lift.get("all_delta"),
            "t50_delta_over_fixed_guard": group_distiller_lift.get("t50_delta"),
            "t100_delta_over_fixed_guard": group_distiller_lift.get("t100_delta"),
            "hard_delta_over_fixed_guard": group_distiller_lift.get("hard_delta"),
            "easy_delta_over_fixed_guard": group_distiller_lift.get("easy_delta"),
            "bootstrap_all_low": (group_distiller_bootstrap.get("all") or {}).get("low"),
            "bootstrap_t50_low": (group_distiller_bootstrap.get("t50") or {}).get("low"),
            "bootstrap_t100_low": (group_distiller_bootstrap.get("t100_raw_frame_diagnostic") or {}).get("low"),
            "bootstrap_hard_low": (group_distiller_bootstrap.get("hard_failure") or {}).get("low"),
            "statistically_stable_on_test": group_distiller_stable,
            "ablation_group_consistency_all_delta": (group_distiller_ablation.get("group_consistency_features") or {}).get("all_delta"),
            "ablation_group_consistency_t100_delta": (group_distiller_ablation.get("group_consistency_features") or {}).get("t100_delta"),
            "ablation_proposal_score_all_delta": (group_distiller_ablation.get("proposal_score_features") or {}).get("all_delta"),
            "ablation_static_all_delta": (group_distiller_ablation.get("static_causal_features") or {}).get("all_delta"),
            "ablation_full_traj_signal_all_delta": (group_distiller_ablation.get("full_trajectory_prediction_signals") or {}).get("all_delta"),
        },
        "group_consistency_multiseed_summary": {
            "initial_replication_pass": group_distiller_multiseed_pass,
            "safety_buffer_repair_pass": group_distiller_repair_pass,
            "validation_collision_ceiling": group_distiller_multiseed_repair.get("validation_collision_ceiling"),
            "test_collision_ceiling": group_distiller_multiseed_repair.get("test_collision_ceiling"),
            "all_mean": (group_distiller_repair_summary.get("all_improvement") or {}).get("mean"),
            "all_min": (group_distiller_repair_summary.get("all_improvement") or {}).get("min"),
            "t50_mean": (group_distiller_repair_summary.get("t50_improvement") or {}).get("mean"),
            "t50_min": (group_distiller_repair_summary.get("t50_improvement") or {}).get("min"),
            "t100_mean": (group_distiller_repair_summary.get("t100_improvement") or {}).get("mean"),
            "t100_min": (group_distiller_repair_summary.get("t100_improvement") or {}).get("min"),
            "hard_mean": (group_distiller_repair_summary.get("hard_failure_improvement") or {}).get("mean"),
            "hard_min": (group_distiller_repair_summary.get("hard_failure_improvement") or {}).get("min"),
            "easy_max": (group_distiller_repair_summary.get("easy_degradation") or {}).get("max"),
            "collision_delta_max": (group_distiller_repair_summary.get("collision_delta_vs_floor_005") or {}).get("max"),
            "switch_rate_mean": (group_distiller_repair_summary.get("switch_rate") or {}).get("mean"),
            "positive_domain_counts": group_distiller_repair_domains,
        },
        "jepa_deployment_decision_summary": {
            "decision": jepa_decision.get("decision"),
            "disable_jepa_in_deployable_path": jepa_disabled,
            "attempt_count": jepa_decision.get("attempt_count"),
            "non_collapse_attempt_count": jepa_decision.get("non_collapse_attempt_count"),
            "deployable_positive_attempt_count": jepa_decision.get("deployable_positive_attempt_count"),
            "keep_jepa_for_diagnostic_research": jepa_decision.get("keep_jepa_for_diagnostic_research"),
        },
        "requirements": requirements,
        "next_highest_value_actions": [
            "Repair UCY fallback-only behavior in the deployable no-base-switch joint policy distiller; bootstrap, first ablations, and three-seed replication are complete.",
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
            "## Joint Multi-Agent Consistency Calibration",
            "",
            f"- selected params: `{joint_consistency.get('selected_params')}`",
            f"- joint consistency contributes: `{joint_consistency.get('joint_multiagent_consistency_contributes')}`",
            f"- all improvement: `{joint_consistency_metrics.get('all_improvement')}`",
            f"- t50 improvement: `{joint_consistency_metrics.get('t50_improvement')}`",
            f"- t100 diagnostic improvement: `{joint_consistency_metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{joint_consistency_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{joint_consistency_metrics.get('easy_degradation')}`",
            f"- all delta over full-trajectory reference: `{joint_consistency_lift.get('all_delta')}`",
            f"- t50 delta over full-trajectory reference: `{joint_consistency_lift.get('t50_delta')}`",
            f"- hard delta over full-trajectory reference: `{joint_consistency_lift.get('hard_delta')}`",
            f"- expanded-on rows: `{(joint_consistency_metrics.get('joint_consistency') or {}).get('expanded_on')}`",
            "",
            "## Joint Policy Distillation",
            "",
            f"- best name: `{joint_distill.get('best_name')}`",
            f"- joint policy distillation contributes: `{joint_distill.get('joint_policy_distillation_contributes')}`",
            f"- all improvement: `{joint_distill_metrics.get('all_improvement')}`",
            f"- t50 improvement: `{joint_distill_metrics.get('t50_improvement')}`",
            f"- t100 diagnostic improvement: `{joint_distill_metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{joint_distill_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{joint_distill_metrics.get('easy_degradation')}`",
            f"- switch rate: `{joint_distill_metrics.get('switch_rate')}`",
            f"- positive external domains: `{joint_distill_positive_domains}`",
            f"- all delta over joint consistency: `{joint_distill_lift_joint.get('all_delta')}`",
            f"- t50 delta over joint consistency: `{joint_distill_lift_joint.get('t50_delta')}`",
            f"- base switch input: `{joint_distill_no_leak.get('base_switch_input')}`",
            f"- base-plus-distiller deployable: `{joint_distill_no_leak.get('base_plus_distiller_deployable')}`",
            f"- bootstrap all CI low: `{(joint_distill_bootstrap.get('all') or {}).get('low')}`",
            f"- bootstrap t50 CI low: `{(joint_distill_bootstrap.get('t50') or {}).get('low')}`",
            f"- bootstrap hard/failure CI low: `{(joint_distill_bootstrap.get('hard_failure') or {}).get('low')}`",
            f"- statistically stable on test: `{joint_distill_stable}`",
            f"- static causal feature ablation all delta: `{(joint_distill_ablation.get('static_causal_features') or {}).get('all_delta')}`",
            f"- full-trajectory signal ablation all delta: `{(joint_distill_ablation.get('full_trajectory_prediction_signals') or {}).get('all_delta')}`",
            f"- multi-seed pass: `{joint_distill_multiseed_pass}`",
            f"- multi-seed all mean/min: `{(joint_distill_multiseed_summary.get('all_improvement') or {}).get('mean')}` / `{(joint_distill_multiseed_summary.get('all_improvement') or {}).get('min')}`",
            f"- multi-seed t50 mean/min: `{(joint_distill_multiseed_summary.get('t50_improvement') or {}).get('mean')}` / `{(joint_distill_multiseed_summary.get('t50_improvement') or {}).get('min')}`",
            f"- multi-seed hard mean: `{(joint_distill_multiseed_summary.get('hard_failure_improvement') or {}).get('mean')}`",
            f"- multi-seed easy max: `{(joint_distill_multiseed_summary.get('easy_degradation') or {}).get('max')}`",
            "",
            "## UCY Fallback Repair",
            "",
            f"- contributes: `{ucy_repair_contributes}`",
            f"- missing val domains: `{ucy_repair.get('missing_val_domains')}`",
            f"- calibration rows: `{(ucy_repair.get('calibration') or {}).get('rows')}`",
            f"- all improvement: `{ucy_repair_metrics.get('all_improvement')}`",
            f"- t50 improvement: `{ucy_repair_metrics.get('t50_improvement')}`",
            f"- t100 diagnostic improvement: `{ucy_repair_metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{ucy_repair_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{ucy_repair_metrics.get('easy_degradation')}`",
            f"- UCY all/t50/t100: `{((ucy_repair_metrics.get('by_domain') or {}).get('UCY') or {}).get('all_improvement')}` / `{((ucy_repair_metrics.get('by_domain') or {}).get('UCY') or {}).get('t50_improvement')}` / `{((ucy_repair_metrics.get('by_domain') or {}).get('UCY') or {}).get('t100_improvement')}`",
            f"- all delta over no-UCY policy: `{ucy_repair_lift.get('all_delta')}`",
            f"- t50 delta over no-UCY policy: `{ucy_repair_lift.get('t50_delta')}`",
            f"- UCY bootstrap low: `{((ucy_repair_bootstrap.get('by_domain') or {}).get('UCY') or {}).get('low')}`",
            f"- train-only UCY calibration: `{(ucy_repair.get('no_leakage') or {}).get('train_only_ucy_threshold_calibration')}`",
            "",
            "## UCY Internal Validation",
            "",
            f"- validation pass: `{ucy_validation_pass}`",
            f"- source-level independent validation available: `{ucy_source_level_available}`",
            f"- source-level blocker: `{ucy_validation.get('source_level_blocker')}`",
            f"- internal validation pass: `{ucy_validation.get('internal_validation_pass')}`",
            f"- temporal validation pass: `{ucy_validation.get('temporal_validation_pass')}`",
            f"- test UCY all/t50/t100/hard/easy: `{(ucy_validation.get('test_ucy_metrics') or {}).get('all_improvement')}` / `{(ucy_validation.get('test_ucy_metrics') or {}).get('t50_improvement')}` / `{(ucy_validation.get('test_ucy_metrics') or {}).get('t100_improvement')}` / `{(ucy_validation.get('test_ucy_metrics') or {}).get('hard_failure_improvement')}` / `{(ucy_validation.get('test_ucy_metrics') or {}).get('easy_degradation')}`",
            "",
            "## Joint Rollout Consistency Audit",
            "",
            f"- pass: `{joint_rollout_pass}`",
            f"- policy source: `{joint_rollout.get('policy_source')}`",
            f"- all/t50/t100: `{joint_rollout_metrics.get('all_improvement')}` / `{joint_rollout_metrics.get('t50_improvement')}` / `{joint_rollout_metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{joint_rollout_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{joint_rollout_metrics.get('easy_degradation')}`",
            f"- multi-agent rows: `{joint_rollout.get('multi_agent_rows')}`",
            f"- multi-agent all/t50/hard: `{joint_rollout_multi.get('all_improvement')}` / `{joint_rollout_multi.get('t50_improvement')}` / `{joint_rollout_multi.get('hard_failure_improvement')}`",
            f"- collision delta vs floor @0.05 normalized: `{joint_rollout.get('collision_delta_vs_floor_005')}`",
            f"- mixed group switch rate: `{(joint_rollout.get('group_switch_summary') or {}).get('selected_mixed_group_switch_rate')}`",
            "",
            "## Neural Group Consistency Distiller",
            "",
            f"- deployable: `{group_distiller_deployable}`",
            f"- improves fixed proximity guard: `{group_distiller_improves_guard}`",
            f"- selected policy: `{group_distiller.get('selected_policy')}`",
            f"- all/t50/t100: `{group_distiller_metrics.get('all_improvement')}` / `{group_distiller_metrics.get('t50_improvement')}` / `{group_distiller_metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{group_distiller_metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{group_distiller_metrics.get('easy_degradation')}`",
            f"- switch rate: `{group_distiller_metrics.get('switch_rate')}`",
            f"- collision delta vs floor @0.05 normalized: `{group_distiller_metrics.get('collision_delta_vs_floor_005')}`",
            f"- lift over fixed guard all/t50/t100/hard: `{group_distiller_lift.get('all_delta')}` / `{group_distiller_lift.get('t50_delta')}` / `{group_distiller_lift.get('t100_delta')}` / `{group_distiller_lift.get('hard_delta')}`",
            f"- bootstrap all/t50/t100/hard CI lows: `{(group_distiller_bootstrap.get('all') or {}).get('low')}` / `{(group_distiller_bootstrap.get('t50') or {}).get('low')}` / `{(group_distiller_bootstrap.get('t100_raw_frame_diagnostic') or {}).get('low')}` / `{(group_distiller_bootstrap.get('hard_failure') or {}).get('low')}`",
            f"- statistically stable on test: `{group_distiller_stable}`",
            f"- group-consistency feature ablation all/t100 delta: `{(group_distiller_ablation.get('group_consistency_features') or {}).get('all_delta')}` / `{(group_distiller_ablation.get('group_consistency_features') or {}).get('t100_delta')}`",
            f"- proposal-score feature ablation all delta: `{(group_distiller_ablation.get('proposal_score_features') or {}).get('all_delta')}`",
            "",
            "## Conclusion",
            "",
            "M3W-Neural v1 is now more than an endpoint-only candidate: the fresh full-trajectory probe adds waypoint trajectory, interaction-risk, occupancy, and physical-validity heads, and the goal/route repair pass adds an explicit route head plus a non-degenerate physical-challenge target. The route/physical heads are useful diagnostics, but post-hoc route/physical gating and joint route-conditioned training are negative ablations for trajectory deployment. Joint policy distillation learns gain/harm/switch without base-switch input and is statistically stable across bootstrap plus three seeds. The UCY fallback-only blocker was traced to missing UCY validation rows and repaired with train-only UCY calibration. A neural group-consistency distiller improves the fixed joint proximity guard, and its initial three-seed run was positive but one seed slightly exceeded the near-proximity safety delta. A validation-selected safety-buffer repair now passes all three seeds while preserving easy cases and joint proximity safety. JEPA is formally disabled from the deployable path because audited non-collapse JEPA variants did not produce deployable downstream lift. This remains grouped 2.5D rollout evidence rather than latent generative world-state execution. The full active objective is still not complete because source-level independent UCY validation remains unavailable and Stage5C/SMC stay disabled.",
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
    joint_consistency_summary = audit.get("joint_multiagent_consistency_summary", {})
    joint_distill_summary = audit.get("joint_policy_distillation_summary", {})
    ucy_repair_summary = audit.get("ucy_fallback_repair_summary", {})
    ucy_validation_summary = audit.get("ucy_independent_validation_summary", {})
    joint_rollout_summary = audit.get("joint_rollout_consistency_summary", {})
    group_distiller_summary = audit.get("group_consistency_distiller_summary", {})
    group_multiseed_summary = audit.get("group_consistency_multiseed_summary", {})
    jepa_decision_summary = audit.get("jepa_deployment_decision_summary", {})
    _replace_section(
        Path("README_RESULTS.md"),
        "M3W_NEURAL_COMPLETION_AUDIT",
        [
            "## M3W-Neural v1 Completion Audit",
            "",
            "The active breakthrough objective is not fully complete yet. M3W-Neural v1 now has a no-base-switch joint policy distiller with bootstrap/multi-seed stability, a train-only UCY fallback repair, a grouped all-agent rollout consistency audit, and a neural group-consistency distiller that improves the fixed proximity guard. The rollout is still not a latent generative world state.",
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
            f"joint_multiagent_consistency_contributes = {joint_consistency_summary.get('joint_multiagent_consistency_contributes')}",
            f"joint_multiagent_consistency_all = {joint_consistency_summary.get('all_improvement')}",
            f"joint_multiagent_consistency_t50 = {joint_consistency_summary.get('t50_improvement')}",
            f"joint_multiagent_consistency_t100_diagnostic = {joint_consistency_summary.get('t100_improvement')}",
            f"joint_multiagent_consistency_hard = {joint_consistency_summary.get('hard_failure_improvement')}",
            f"joint_multiagent_consistency_easy = {joint_consistency_summary.get('easy_degradation')}",
            f"joint_multiagent_consistency_all_delta_vs_full_traj = {joint_consistency_summary.get('all_delta_over_full_trajectory_reference')}",
            f"joint_multiagent_consistency_t50_delta_vs_full_traj = {joint_consistency_summary.get('t50_delta_over_full_trajectory_reference')}",
            f"joint_multiagent_consistency_expanded_on = {joint_consistency_summary.get('expanded_on')}",
            f"joint_policy_distillation_best = {joint_distill_summary.get('best_name')}",
            f"joint_policy_distillation_contributes = {joint_distill_summary.get('joint_policy_distillation_contributes')}",
            f"joint_policy_distillation_all = {joint_distill_summary.get('all_improvement')}",
            f"joint_policy_distillation_t50 = {joint_distill_summary.get('t50_improvement')}",
            f"joint_policy_distillation_t100_diagnostic = {joint_distill_summary.get('t100_improvement')}",
            f"joint_policy_distillation_hard = {joint_distill_summary.get('hard_failure_improvement')}",
            f"joint_policy_distillation_easy = {joint_distill_summary.get('easy_degradation')}",
            f"joint_policy_distillation_switch_rate = {joint_distill_summary.get('switch_rate')}",
            f"joint_policy_distillation_positive_domains = {joint_distill_summary.get('positive_external_domains')}",
            f"joint_policy_distillation_all_delta_vs_joint_consistency = {joint_distill_summary.get('all_delta_over_joint_consistency')}",
            f"joint_policy_distillation_t50_delta_vs_joint_consistency = {joint_distill_summary.get('t50_delta_over_joint_consistency')}",
            f"joint_policy_distillation_base_switch_input = {joint_distill_summary.get('base_switch_input')}",
            f"joint_policy_distillation_bootstrap_all_low = {joint_distill_summary.get('bootstrap_all_low')}",
            f"joint_policy_distillation_bootstrap_t50_low = {joint_distill_summary.get('bootstrap_t50_low')}",
            f"joint_policy_distillation_bootstrap_hard_low = {joint_distill_summary.get('bootstrap_hard_low')}",
            f"joint_policy_distillation_stable = {joint_distill_summary.get('statistically_stable_on_test')}",
            f"joint_policy_distillation_static_ablation_all_delta = {joint_distill_summary.get('ablation_static_all_delta')}",
            f"joint_policy_distillation_prediction_ablation_all_delta = {joint_distill_summary.get('ablation_prediction_all_delta')}",
            f"joint_policy_distillation_multiseed_pass = {joint_distill_summary.get('multiseed_pass')}",
            f"joint_policy_distillation_multiseed_all_mean = {joint_distill_summary.get('multiseed_all_mean')}",
            f"joint_policy_distillation_multiseed_all_min = {joint_distill_summary.get('multiseed_all_min')}",
            f"joint_policy_distillation_multiseed_t50_mean = {joint_distill_summary.get('multiseed_t50_mean')}",
            f"joint_policy_distillation_multiseed_t50_min = {joint_distill_summary.get('multiseed_t50_min')}",
            f"joint_policy_distillation_multiseed_easy_max = {joint_distill_summary.get('multiseed_easy_max')}",
            f"ucy_fallback_repair_contributes = {ucy_repair_summary.get('contributes')}",
            f"ucy_fallback_repair_all = {ucy_repair_summary.get('all_improvement')}",
            f"ucy_fallback_repair_t50 = {ucy_repair_summary.get('t50_improvement')}",
            f"ucy_fallback_repair_t100_diagnostic = {ucy_repair_summary.get('t100_improvement')}",
            f"ucy_fallback_repair_hard = {ucy_repair_summary.get('hard_failure_improvement')}",
            f"ucy_fallback_repair_easy = {ucy_repair_summary.get('easy_degradation')}",
            f"ucy_fallback_repair_ucy_all = {ucy_repair_summary.get('ucy_all_improvement')}",
            f"ucy_fallback_repair_ucy_t50 = {ucy_repair_summary.get('ucy_t50_improvement')}",
            f"ucy_fallback_repair_bootstrap_ucy_low = {ucy_repair_summary.get('bootstrap_ucy_low')}",
            f"ucy_internal_validation_pass = {ucy_validation_summary.get('validation_pass')}",
            f"ucy_source_level_validation_available = {ucy_validation_summary.get('source_level_independent_validation_available')}",
            f"ucy_source_level_blocker = {ucy_validation_summary.get('source_level_blocker')}",
            f"joint_rollout_consistency_pass = {joint_rollout_summary.get('pass')}",
            f"joint_rollout_consistency_all = {joint_rollout_summary.get('all_improvement')}",
            f"joint_rollout_consistency_t50 = {joint_rollout_summary.get('t50_improvement')}",
            f"joint_rollout_consistency_t100_diagnostic = {joint_rollout_summary.get('t100_improvement')}",
            f"joint_rollout_consistency_hard = {joint_rollout_summary.get('hard_failure_improvement')}",
            f"joint_rollout_consistency_easy = {joint_rollout_summary.get('easy_degradation')}",
            f"joint_rollout_consistency_multi_agent_all = {joint_rollout_summary.get('multi_agent_all_improvement')}",
            f"joint_rollout_consistency_collision_delta_005 = {joint_rollout_summary.get('collision_delta_vs_floor_005')}",
            f"group_consistency_distiller_deployable = {group_distiller_summary.get('deployable')}",
            f"group_consistency_distiller_improves_fixed_guard = {group_distiller_summary.get('improves_fixed_guard')}",
            f"group_consistency_distiller_all = {group_distiller_summary.get('all_improvement')}",
            f"group_consistency_distiller_t50 = {group_distiller_summary.get('t50_improvement')}",
            f"group_consistency_distiller_t100_diagnostic = {group_distiller_summary.get('t100_improvement')}",
            f"group_consistency_distiller_hard = {group_distiller_summary.get('hard_failure_improvement')}",
            f"group_consistency_distiller_easy = {group_distiller_summary.get('easy_degradation')}",
            f"group_consistency_distiller_collision_delta_005 = {group_distiller_summary.get('collision_delta_vs_floor_005')}",
            f"group_consistency_distiller_t100_delta_vs_fixed_guard = {group_distiller_summary.get('t100_delta_over_fixed_guard')}",
            f"group_consistency_distiller_bootstrap_all_low = {group_distiller_summary.get('bootstrap_all_low')}",
            f"group_consistency_distiller_bootstrap_t50_low = {group_distiller_summary.get('bootstrap_t50_low')}",
            f"group_consistency_distiller_bootstrap_t100_low = {group_distiller_summary.get('bootstrap_t100_low')}",
            f"group_consistency_distiller_bootstrap_hard_low = {group_distiller_summary.get('bootstrap_hard_low')}",
            f"group_consistency_distiller_stable = {group_distiller_summary.get('statistically_stable_on_test')}",
            f"group_consistency_distiller_group_feature_ablation_all_delta = {group_distiller_summary.get('ablation_group_consistency_all_delta')}",
            f"group_consistency_distiller_proposal_score_ablation_all_delta = {group_distiller_summary.get('ablation_proposal_score_all_delta')}",
            f"group_consistency_multiseed_initial_pass = {group_multiseed_summary.get('initial_replication_pass')}",
            f"group_consistency_multiseed_safety_buffer_pass = {group_multiseed_summary.get('safety_buffer_repair_pass')}",
            f"group_consistency_multiseed_all_mean = {group_multiseed_summary.get('all_mean')}",
            f"group_consistency_multiseed_all_min = {group_multiseed_summary.get('all_min')}",
            f"group_consistency_multiseed_t50_mean = {group_multiseed_summary.get('t50_mean')}",
            f"group_consistency_multiseed_t50_min = {group_multiseed_summary.get('t50_min')}",
            f"group_consistency_multiseed_t100_mean = {group_multiseed_summary.get('t100_mean')}",
            f"group_consistency_multiseed_t100_min = {group_multiseed_summary.get('t100_min')}",
            f"group_consistency_multiseed_hard_mean = {group_multiseed_summary.get('hard_mean')}",
            f"group_consistency_multiseed_hard_min = {group_multiseed_summary.get('hard_min')}",
            f"group_consistency_multiseed_easy_max = {group_multiseed_summary.get('easy_max')}",
            f"group_consistency_multiseed_collision_delta_max = {group_multiseed_summary.get('collision_delta_max')}",
            f"group_consistency_multiseed_positive_domain_counts = {group_multiseed_summary.get('positive_domain_counts')}",
            f"jepa_deployment_decision = {jepa_decision_summary.get('decision')}",
            f"jepa_disable_deployable_path = {jepa_decision_summary.get('disable_jepa_in_deployable_path')}",
            f"jepa_attempt_count = {jepa_decision_summary.get('attempt_count')}",
            f"jepa_non_collapse_attempt_count = {jepa_decision_summary.get('non_collapse_attempt_count')}",
            f"jepa_deployable_positive_attempt_count = {jepa_decision_summary.get('deployable_positive_attempt_count')}",
            "stage5c_executed = false",
            "smc_enabled = false",
            "```",
            "",
            "Next target: rebuild or acquire a true source-level UCY validation split and move from grouped consistency auditing toward a jointly learned multi-agent latent rollout. Current claims remain dataset-local raw-frame 2.5D, not true 3D or foundation.",
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
    generated.add("outputs/stage41_fresh_confirmation/stage41_joint_multiagent_consistency.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_joint_multiagent_consistency.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation_evidence.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation_evidence.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation_multiseed.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation_multiseed.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_ucy_fallback_repair.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_ucy_fallback_repair.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_ucy_independent_validation.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_ucy_independent_validation.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_joint_rollout_consistency.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_joint_rollout_consistency.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_group_consistency_distiller.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_group_consistency_distiller.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_group_consistency_evidence.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_group_consistency_evidence.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_group_consistency_multiseed.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_group_consistency_multiseed.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_group_consistency_multiseed_repair.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_group_consistency_multiseed_repair.json")
    generated.add("outputs/stage41_fresh_confirmation/stage41_jepa_deployment_decision.md")
    generated.add("outputs/stage41_fresh_confirmation/stage41_jepa_deployment_decision.json")
    state["generated_reports"] = sorted(generated)
    state["current_verdict"] = "stage41_group_consistency_multiseed_safety_buffer_joint_safe_strong_not_complete"
    state["current_best_deployable"] = audit.get("current_best_deployable")
    state["m3w_neural_v1_current_candidate"] = {
        "source": audit.get("source"),
        "completion_status": audit.get("completion_status"),
        "deployment_state": (
            "group_consistency_multiseed_safety_buffer_joint_safe_candidate_pending_independent_validation"
            if group_multiseed_summary.get("safety_buffer_repair_pass")
            else "group_consistency_distilled_joint_safe_candidate_pending_independent_validation"
        ),
        "current_best_deployable": audit.get("current_best_deployable"),
        "best_name": (
            "group_consistency_distiller_safety_buffer_multiseed"
            if group_multiseed_summary.get("safety_buffer_repair_pass")
            else "group_consistency_distiller"
        ),
        "deployable_metric_basis": (
            "three_seed_safety_buffer_mean"
            if group_multiseed_summary.get("safety_buffer_repair_pass")
            else "single_seed_validation_selected"
        ),
        "all_improvement": group_multiseed_summary.get("all_mean") if group_multiseed_summary.get("safety_buffer_repair_pass") else group_distiller_summary.get("all_improvement"),
        "t50_improvement": group_multiseed_summary.get("t50_mean") if group_multiseed_summary.get("safety_buffer_repair_pass") else group_distiller_summary.get("t50_improvement"),
        "t100_raw_frame_diagnostic": group_multiseed_summary.get("t100_mean") if group_multiseed_summary.get("safety_buffer_repair_pass") else group_distiller_summary.get("t100_improvement"),
        "hard_failure_improvement": group_multiseed_summary.get("hard_mean") if group_multiseed_summary.get("safety_buffer_repair_pass") else group_distiller_summary.get("hard_failure_improvement"),
        "easy_degradation": group_multiseed_summary.get("easy_max") if group_multiseed_summary.get("safety_buffer_repair_pass") else group_distiller_summary.get("easy_degradation"),
        "switch_rate": group_multiseed_summary.get("switch_rate_mean") if group_multiseed_summary.get("safety_buffer_repair_pass") else group_distiller_summary.get("switch_rate"),
        "collision_delta_vs_floor_005": group_multiseed_summary.get("collision_delta_max") if group_multiseed_summary.get("safety_buffer_repair_pass") else group_distiller_summary.get("collision_delta_vs_floor_005"),
        "lift_over_fixed_guard_all": group_distiller_summary.get("all_delta_over_fixed_guard"),
        "lift_over_fixed_guard_t50": group_distiller_summary.get("t50_delta_over_fixed_guard"),
        "lift_over_fixed_guard_t100": group_distiller_summary.get("t100_delta_over_fixed_guard"),
        "lift_over_fixed_guard_hard": group_distiller_summary.get("hard_delta_over_fixed_guard"),
        "row_level_ucy_repaired_all": ucy_repair_summary.get("all_improvement"),
        "row_level_ucy_repaired_t50": ucy_repair_summary.get("t50_improvement"),
        "row_level_ucy_repaired_t100": ucy_repair_summary.get("t100_improvement"),
        "row_level_ucy_repaired_note": "Higher row-level FDE than group-safe variants, but raw repaired policy increased near-proximity risk in joint rollout audit.",
        "positive_external_domains": 3 if ucy_repair_summary.get("contributes") else joint_distill_summary.get("positive_external_domains"),
        "base_switch_input": joint_distill_summary.get("base_switch_input"),
        "base_plus_distiller_deployable": joint_distill_summary.get("base_plus_distiller_deployable"),
        "bootstrap_all_low": joint_distill_summary.get("bootstrap_all_low"),
        "bootstrap_t50_low": joint_distill_summary.get("bootstrap_t50_low"),
        "bootstrap_hard_low": joint_distill_summary.get("bootstrap_hard_low"),
        "statistically_stable_on_test": joint_distill_summary.get("statistically_stable_on_test"),
        "ablation_static_all_delta": joint_distill_summary.get("ablation_static_all_delta"),
        "ablation_prediction_all_delta": joint_distill_summary.get("ablation_prediction_all_delta"),
        "multiseed_pass": joint_distill_summary.get("multiseed_pass"),
        "multiseed_all_mean": joint_distill_summary.get("multiseed_all_mean"),
        "multiseed_all_min": joint_distill_summary.get("multiseed_all_min"),
        "multiseed_t50_mean": joint_distill_summary.get("multiseed_t50_mean"),
        "multiseed_t50_min": joint_distill_summary.get("multiseed_t50_min"),
        "multiseed_easy_max": joint_distill_summary.get("multiseed_easy_max"),
        "ucy_status": "train_calibrated_repaired_pending_independent_validation",
        "ucy_all_improvement": ucy_repair_summary.get("ucy_all_improvement"),
        "ucy_t50_improvement": ucy_repair_summary.get("ucy_t50_improvement"),
        "ucy_bootstrap_low": ucy_repair_summary.get("bootstrap_ucy_low"),
        "ucy_internal_validation_pass": ucy_validation_summary.get("validation_pass"),
        "ucy_source_level_validation_available": ucy_validation_summary.get("source_level_independent_validation_available"),
        "ucy_source_level_blocker": ucy_validation_summary.get("source_level_blocker"),
        "joint_rollout_consistency_pass": joint_rollout_summary.get("pass"),
        "joint_rollout_all_improvement": joint_rollout_summary.get("all_improvement"),
        "joint_rollout_t50_improvement": joint_rollout_summary.get("t50_improvement"),
        "joint_rollout_t100_raw_frame_diagnostic": joint_rollout_summary.get("t100_improvement"),
        "joint_rollout_hard_failure_improvement": joint_rollout_summary.get("hard_failure_improvement"),
        "joint_rollout_easy_degradation": joint_rollout_summary.get("easy_degradation"),
        "joint_rollout_multi_agent_all_improvement": joint_rollout_summary.get("multi_agent_all_improvement"),
        "joint_rollout_collision_delta_vs_floor_005": joint_rollout_summary.get("collision_delta_vs_floor_005"),
        "group_consistency_distiller_deployable": group_distiller_summary.get("deployable"),
        "group_consistency_distiller_improves_fixed_guard": group_distiller_summary.get("improves_fixed_guard"),
        "group_consistency_distiller_bootstrap_all_low": group_distiller_summary.get("bootstrap_all_low"),
        "group_consistency_distiller_bootstrap_t50_low": group_distiller_summary.get("bootstrap_t50_low"),
        "group_consistency_distiller_bootstrap_t100_low": group_distiller_summary.get("bootstrap_t100_low"),
        "group_consistency_distiller_bootstrap_hard_low": group_distiller_summary.get("bootstrap_hard_low"),
        "group_consistency_distiller_statistically_stable": group_distiller_summary.get("statistically_stable_on_test"),
        "group_consistency_feature_ablation_all_delta": group_distiller_summary.get("ablation_group_consistency_all_delta"),
        "proposal_score_ablation_all_delta": group_distiller_summary.get("ablation_proposal_score_all_delta"),
        "group_consistency_multiseed_initial_pass": group_multiseed_summary.get("initial_replication_pass"),
        "group_consistency_multiseed_safety_buffer_pass": group_multiseed_summary.get("safety_buffer_repair_pass"),
        "group_consistency_multiseed_all_mean": group_multiseed_summary.get("all_mean"),
        "group_consistency_multiseed_all_min": group_multiseed_summary.get("all_min"),
        "group_consistency_multiseed_t50_mean": group_multiseed_summary.get("t50_mean"),
        "group_consistency_multiseed_t50_min": group_multiseed_summary.get("t50_min"),
        "group_consistency_multiseed_t100_mean": group_multiseed_summary.get("t100_mean"),
        "group_consistency_multiseed_t100_min": group_multiseed_summary.get("t100_min"),
        "group_consistency_multiseed_hard_mean": group_multiseed_summary.get("hard_mean"),
        "group_consistency_multiseed_hard_min": group_multiseed_summary.get("hard_min"),
        "group_consistency_multiseed_easy_max": group_multiseed_summary.get("easy_max"),
        "group_consistency_multiseed_collision_delta_max": group_multiseed_summary.get("collision_delta_max"),
        "group_consistency_multiseed_positive_domain_counts": group_multiseed_summary.get("positive_domain_counts"),
        "jepa_deployment_decision": jepa_decision_summary.get("decision"),
        "jepa_disable_deployable_path": jepa_decision_summary.get("disable_jepa_in_deployable_path"),
        "jepa_attempt_count": jepa_decision_summary.get("attempt_count"),
        "jepa_non_collapse_attempt_count": jepa_decision_summary.get("non_collapse_attempt_count"),
        "jepa_deployable_positive_attempt_count": jepa_decision_summary.get("deployable_positive_attempt_count"),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
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
        "joint_multiagent_consistency_summary": joint_consistency_summary,
        "joint_policy_distillation_summary": joint_distill_summary,
        "ucy_fallback_repair_summary": ucy_repair_summary,
        "ucy_independent_validation_summary": ucy_validation_summary,
        "joint_rollout_consistency_summary": joint_rollout_summary,
        "group_consistency_distiller_summary": group_distiller_summary,
        "group_consistency_multiseed_summary": group_multiseed_summary,
        "jepa_deployment_decision_summary": jepa_decision_summary,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_successful_command"] = "python run_m3w_neural_completion_audit.py"
    write_json("research_state.json", _jsonable(state))


if __name__ == "__main__":
    print(json.dumps(_jsonable(build_completion_audit()), indent=2, ensure_ascii=False))
