from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class AuditGate:
    name: str
    passed: bool
    score: float
    evidence: str
    fix: str
    severity: str = "major"


def run_world_model_self_audit(
    metrics_path: str | Path = "outputs/reports/metrics_stage2.json",
    report_path: str | Path = "outputs/reports/report_stage2.md",
    output_markdown: str | Path = "outputs/reports/world_model_expert_self_audit.md",
    output_json: str | Path = "outputs/reports/world_model_expert_self_audit.json",
) -> Dict:
    metrics_path = Path(metrics_path)
    report_path = Path(report_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    gates = build_audit_gates(metrics, report_text)
    score = round(sum(g.score for g in gates), 2)
    verdict = verdict_from_score(score, gates)
    payload = {
        "score": score,
        "max_score": 100,
        "verdict": verdict,
        "model_type": infer_model_type(metrics, report_text),
        "passed_gates": sum(g.passed for g in gates),
        "total_gates": len(gates),
        "gates": [gate.__dict__ for gate in gates],
        "priority_actions": priority_actions(gates),
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(output_markdown).write_text(render_markdown(payload), encoding="utf-8")
    return payload


def build_audit_gates(metrics: Dict, report_text: str) -> List[AuditGate]:
    constant = metrics["constant_velocity_baseline"]
    hand = metrics["hand_physics_baseline"]
    deterministic = metrics["deterministic_neural_residual"]
    stochastic = metrics["stochastic_neural_residual"]
    hybrid_smc = metrics["physics_plus_neural_residual_SMC"]
    learned_smc = metrics["learned_neural_SMC"]

    gates = [
        gate_data_protocol(metrics, report_text),
        gate_no_aerial_t100_overclaim(report_text),
        gate_long_horizon_accuracy(hybrid_smc),
        gate_learned_residual_improvement(hand, deterministic, hybrid_smc),
        gate_stochastic_coverage(stochastic, hybrid_smc),
        gate_physical_consistency(hybrid_smc),
        gate_constraints_help(constant, hybrid_smc),
        gate_semantic_diversity(hybrid_smc),
        gate_smc_value(hand, deterministic, hybrid_smc, learned_smc),
        gate_real_world_readiness(report_text),
    ]
    return gates


def gate_data_protocol(metrics: Dict, report_text: str) -> AuditGate:
    all_h100 = all("100" in model.get("horizons", {}) and model["horizons"]["100"].get("evaluated_horizon") == 100 for model in metrics.values())
    has_truth_claim = "Synthetic data: verified t+100" in report_text or "Synthetic t+100 是否可验证" in report_text
    passed = all_h100 and has_truth_claim
    return AuditGate(
        "synthetic_t100_protocol",
        passed,
        10.0 if passed else 2.0,
        f"horizon_100_present={all_h100}; report_states_verified_synthetic_t100={has_truth_claim}",
        "Keep synthetic t+100 as verified evaluation and never mix it with real-data free-run metrics.",
        "critical",
    )


def gate_no_aerial_t100_overclaim(report_text: str) -> AuditGate:
    has_free_run_warning = "qualitative free-run" in report_text and "不能报告 ADE@100/FDE@100" in report_text
    suspicious = "AerialMPT ADE@100" in report_text or "AerialMPT FDE@100" in report_text
    passed = has_free_run_warning and not suspicious
    return AuditGate(
        "no_real_t100_overclaim",
        passed,
        10.0 if passed else 0.0,
        f"free_run_warning={has_free_run_warning}; suspicious_real_t100_metric={suspicious}",
        "Keep AerialMPT t+100 qualitative-only until a real long sequence is loaded.",
        "critical",
    )


def gate_long_horizon_accuracy(model: Dict) -> AuditGate:
    ade = model["horizons"]["100"]["ADE_m"]
    fde = model["horizons"]["100"]["FDE_m"]
    passed = ade < 2.0 and fde < 5.0
    partial = ade < 5.0
    score = 15.0 if passed else (6.0 if partial else 2.0)
    return AuditGate(
        "long_horizon_accuracy",
        passed,
        score,
        f"hybrid_SMC ADE@100={ade:.3f}m; FDE@100={fde:.3f}m; target ADE<2m and FDE<5m",
        "Train on larger/diverse synthetic data, add explicit goal inference, and tune model against rollout loss not only one-step residual.",
        "critical",
    )


def gate_learned_residual_improvement(hand: Dict, deterministic: Dict, hybrid_smc: Dict) -> AuditGate:
    hand_ade = hand["horizons"]["100"]["ADE_m"]
    det_ade = deterministic["horizons"]["100"]["ADE_m"]
    hybrid_ade = hybrid_smc["horizons"]["100"]["ADE_m"]
    det_gain = (hand_ade - det_ade) / max(1e-9, hand_ade)
    hybrid_gain = (hand_ade - hybrid_ade) / max(1e-9, hand_ade)
    passed = det_gain >= 0.10 or hybrid_gain >= 0.10
    partial = det_gain > 0 or hybrid_gain > 0
    return AuditGate(
        "learned_dynamics_beats_hand_physics",
        passed,
        12.0 if passed else (5.0 if partial else 0.0),
        f"deterministic_gain={det_gain:.2%}; hybrid_SMC_gain={hybrid_gain:.2%}; required >=10%",
        "Use multi-step rollout loss, residual dropout, event-balanced minibatches, and learned goal posterior instead of only one-step residual acceleration.",
        "critical",
    )


def gate_stochastic_coverage(stochastic: Dict, hybrid_smc: Dict) -> AuditGate:
    branches = max(stochastic.get("branch_count", 0), hybrid_smc.get("branch_count", 0))
    coverage = max(stochastic.get("coverage@64", 0), hybrid_smc.get("coverage@64", 0))
    best_fde = min(
        stochastic["horizons"]["100"]["best_of_64_FDE_m"],
        hybrid_smc["horizons"]["100"]["best_of_64_FDE_m"],
    )
    passed = branches >= 64 and coverage > 0.15 and best_fde < 5.0
    partial = branches >= 8 and best_fde < 12.0
    return AuditGate(
        "stochastic_multibranch_coverage",
        passed,
        12.0 if passed else (4.0 if partial else 0.0),
        f"branch_count={branches}; coverage@64={coverage}; best_FDE@100={best_fde:.3f}m",
        "Make SMC particles represent latent goals/intents, report actual best-of-N, and run true 64-particle evaluation after runtime stabilizes.",
        "critical",
    )


def gate_physical_consistency(model: Dict) -> AuditGate:
    collision = model["collision_violation_rate"]
    boundary = model["boundary_violation_rate"]
    obstacle = model["obstacle_violation_rate"]
    min_gap = model["min_gap_m"]
    passed = collision < 0.01 and boundary < 0.02 and obstacle < 0.005 and min_gap >= -0.01
    partial = collision < 0.02 and boundary < 0.06 and obstacle < 0.01
    return AuditGate(
        "physical_consistency",
        passed,
        12.0 if passed else (6.0 if partial else 1.0),
        f"collision={collision}; boundary={boundary}; obstacle={obstacle}; min_gap={min_gap}m",
        "Make projection cost visible in weights, add stronger wall-aware path planning, and penalize boundary drift in rollout not just one-step training.",
        "critical",
    )


def gate_constraints_help(constant: Dict, model: Dict) -> AuditGate:
    collision_gain = constant["collision_violation_rate"] - model["collision_violation_rate"]
    boundary_gain = constant["boundary_violation_rate"] - model["boundary_violation_rate"]
    passed = collision_gain > 0 and boundary_gain > 0
    return AuditGate(
        "constraints_reduce_violations",
        passed,
        8.0 if passed else 1.0,
        f"collision_reduction={collision_gain:.5f}; boundary_reduction={boundary_gain:.5f}",
        "Keep physical projection, but separate correction from likelihood so invalid proposals are not silently sanitized.",
        "major",
    )


def gate_semantic_diversity(model: Dict) -> AuditGate:
    diversity = model["cluster_diversity_score"]
    event_acc = model["semantic_event_accuracy"]
    passed = diversity >= 0.55 and event_acc >= 0.4
    partial = diversity > 0.25
    return AuditGate(
        "terminal_semantic_diversity",
        passed,
        8.0 if passed else (3.0 if partial else 0.0),
        f"cluster_diversity={diversity}; semantic_event_accuracy={event_acc}",
        "Cluster over intent, density, pass time, split/merge, detour, and jam features; evaluate event-balanced episodes and avoid endpoint-only semantics.",
        "major",
    )


def gate_smc_value(hand: Dict, deterministic: Dict, hybrid_smc: Dict, learned_smc: Dict) -> AuditGate:
    det_best = deterministic["horizons"]["100"]["FDE_m"]
    hybrid_best = hybrid_smc["horizons"]["100"]["best_of_64_FDE_m"]
    learned_fde = learned_smc["horizons"]["100"]["FDE_m"]
    hand_fde = hand["horizons"]["100"]["FDE_m"]
    passed = hybrid_best < det_best * 0.85 and learned_fde <= hand_fde * 1.05
    partial = hybrid_best < det_best
    return AuditGate(
        "smc_adds_predictive_value",
        passed,
        8.0 if passed else (3.0 if partial else 0.0),
        f"det_FDE@100={det_best:.3f}; hybrid_best_FDE@100={hybrid_best:.3f}; learned_SMC_FDE@100={learned_fde:.3f}; hand_FDE@100={hand_fde:.3f}",
        "Use learned proposal as residual around physics, not standalone acceleration; add observation/state likelihood on synthetic and latent-goal rejuvenation.",
        "major",
    )


def gate_real_world_readiness(report_text: str) -> AuditGate:
    has_limits = "AerialMPT bauma3" in report_text and "t+12" in report_text and ("weak" in report_text.lower() or "弱" in report_text)
    has_next_data = "Stanford Drone" in report_text or "TrajNet++" in report_text or "ETH" in report_text
    passed = False
    return AuditGate(
        "real_world_readiness",
        passed,
        0.0 if has_limits else 1.0,
        f"real_data_limits_acknowledged={has_limits}; next_data_named={has_next_data}; no calibrated long real-data loader yet",
        "Build a long real-data loader, calibrated scene geometry, and verified t+100 real benchmark before claiming real-world readiness.",
        "critical",
    )


def infer_model_type(metrics: Dict, report_text: str) -> str:
    if "pseudo-3D physics-informed learned residual state-space world model" in report_text:
        return "pseudo-3D physics-informed learned residual state-space world model"
    if "learned" in report_text:
        return "partially learned 2.5D state-space world model"
    return "physics-informed scaffold"


def verdict_from_score(score: float, gates: List[AuditGate]) -> str:
    critical_failures = [g for g in gates if not g.passed and g.severity == "critical"]
    if score >= 85 and not critical_failures:
        return "world-class_candidate"
    if score >= 65 and len(critical_failures) <= 1:
        return "promising_research_model"
    if score >= 45:
        return "prototype_with_major_failures"
    return "not_yet_a_serious_world_model"


def priority_actions(gates: List[AuditGate]) -> List[str]:
    failed = sorted([g for g in gates if not g.passed], key=lambda gate: (gate.severity != "critical", -gate.score))
    return [gate.fix for gate in failed[:6]]


def render_markdown(payload: Dict) -> str:
    lines = [
        "# World Model Expert Self-Audit",
        "",
        "## Verdict",
        "",
        f"- Score: `{payload['score']}/{payload['max_score']}`",
        f"- Verdict: `{payload['verdict']}`",
        f"- Model type: `{payload['model_type']}`",
        f"- Gates passed: `{payload['passed_gates']}/{payload['total_gates']}`",
        "",
        "## Blunt Expert Assessment",
        "",
        blunt_assessment(payload),
        "",
        "## Gate Results",
        "",
        "| Gate | Pass | Score | Evidence | Required Fix |",
        "| --- | --- | --- | --- | --- |",
    ]
    for gate in payload["gates"]:
        lines.append(
            f"| {gate['name']} | {gate['passed']} | {gate['score']} | {gate['evidence']} | {gate['fix']} |"
        )
    lines.extend(
        [
            "",
            "## Priority Actions",
            "",
        ]
    )
    for idx, action in enumerate(payload["priority_actions"], start=1):
        lines.append(f"{idx}. {action}")
    lines.extend(
        [
            "",
            "## Bar For A Truly Exceptional World Model",
            "",
            "- Verified t+100 on synthetic and at least one calibrated real long-trajectory dataset.",
            "- Learned dynamics must beat hand physics by a meaningful margin, not a rounding error.",
            "- SMC/multi-branch futures must increase coverage and produce semantically distinct terminal modes.",
            "- Physical constraints must reduce collision, obstacle, boundary, speed, and acceleration violations without hiding bad proposals.",
            "- Camera / homography / scene geometry uncertainty must be explicit.",
            "- Reports must separate verified forecasts from qualitative free-run every time.",
        ]
    )
    return "\n".join(lines) + "\n"


def blunt_assessment(payload: Dict) -> str:
    verdict = payload["verdict"]
    if verdict == "not_yet_a_serious_world_model":
        return (
            "当前系统还不是惊世骇俗的世界模型。它已经是一个诚实的 2.5D learned-residual scaffold，"
            "但长期预测、coverage、语义多样性、真实数据可验证性都没有达标。真正值得肯定的是："
            "它现在能明确地证明自己还不够好。"
        )
    if verdict == "prototype_with_major_failures":
        return (
            "当前系统是有价值的研究原型，但离世界级模型还差一截。它有可验证 synthetic t+100，"
            "也有 learned residual，但多分支概率和真实数据闭环仍然薄弱。"
        )
    if verdict == "promising_research_model":
        return "当前系统已经像一个严肃研究模型，但还需要真实长轨迹和更强多模态 coverage 来支撑大声明。"
    return "当前系统达到候选世界级标准，但仍需外部真实数据复现。"


if __name__ == "__main__":
    result = run_world_model_self_audit()
    print(json.dumps({"score": result["score"], "verdict": result["verdict"]}, ensure_ascii=False))
