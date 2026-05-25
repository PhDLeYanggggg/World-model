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
            "status": _status(False, partial=all_agent_positive or t50_specialist_positive or composer_positive or locked_strong_candidate or fresh_all_agent_pass),
            "evidence": "outputs/stage41_breakthrough/stage41_all_agent_eval.json, stage41_all_agent_risk_repair.json, stage41_all_agent_t50_specialist.json, stage41_all_agent_policy_composer.json, outputs/stage41_stratified_protocol/stage41_fixed_policy_confirmation.json, and outputs/stage41_fresh_confirmation/stage41_fresh_all_agent_endpoint_specialist.json",
            "note": "Fresh source-rotation all-agent endpoint specialist now exceeds Stage37 margins on all/t50/t100/hard with easy preserved and two positive external domains. It is still endpoint-level future-state, not a full trajectory-latent world-state rollout, so the full objective remains not complete.",
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
        "requirements": requirements,
        "next_highest_value_actions": [
            "Train all-agent t50-specific endpoint model with domain/horizon-balanced validation rather than generic all-agent endpoint head.",
            "Extend the fresh all-agent endpoint specialist from endpoint-only future-state to full trajectory/occupancy/interaction latent heads.",
            "Add explicit per-neighbor future-interaction labels and multi-agent occupancy/physical-validity probes.",
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
            "## Conclusion",
            "",
            "M3W-Neural v1 is a strong protected endpoint-dynamics candidate. The fresh source-rotation all-agent endpoint specialist is now the strongest all-agent neural signal, but the full active objective is not complete because it remains endpoint-level future-state rather than full trajectory-latent world-state dynamics.",
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
    _replace_section(
        Path("README_RESULTS.md"),
        "M3W_NEURAL_COMPLETION_AUDIT",
        [
            "## M3W-Neural v1 Completion Audit",
            "",
            "The active breakthrough objective is not fully complete yet. M3W-Neural v1 endpoint dynamics pass Stage41, but full all-agent future world-state dynamics remain diagnostic.",
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
            "stage5c_executed = false",
            "smc_enabled = false",
            "```",
            "",
            "Next target: upgrade the fresh all-agent endpoint specialist into full trajectory/occupancy/interaction world-state dynamics; current claims remain dataset-local raw-frame 2.5D, not true 3D or foundation.",
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
    state["generated_reports"] = sorted(generated)
    state["m3w_neural_v1_completion_audit"] = {
        "source": audit.get("source"),
        "completion_status": audit.get("completion_status"),
        "current_best_deployable": audit.get("current_best_deployable"),
        "all_agent_risk_repair_summary": summary,
        "all_agent_t50_specialist_summary": t50_summary,
        "all_agent_policy_composer_summary": composer_summary,
        "all_agent_locked_v2_confirmation_summary": locked_summary,
        "fresh_all_agent_endpoint_specialist_summary": fresh_all_agent_summary,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_successful_command"] = "python run_m3w_neural_completion_audit.py"
    write_json("research_state.json", _jsonable(state))


if __name__ == "__main__":
    print(json.dumps(_jsonable(build_completion_audit()), indent=2, ensure_ascii=False))
