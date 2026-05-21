from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def run_stage4_gates(metrics: Dict, episode_summary: Dict, prior_audit_score: float = 58.0, output_path: str | Path = "outputs/reports/world_model_gate_stage4.md") -> Dict:
    gates = [
        gate_real_data(episode_summary),
        gate_verified_horizon(episode_summary),
        gate_learned_dynamics(metrics),
        gate_coverage(metrics),
        gate_physical_validity(metrics),
        gate_semantic_diversity(metrics),
        gate_audit_score(prior_audit_score),
    ]
    payload = {"gates": gates, "passed": sum(g["pass"] for g in gates), "total": len(gates), "overall_pass": all(g["pass"] for g in gates)}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(render_gate_markdown(payload), encoding="utf-8")
    return payload


def gate_real_data(summary: Dict) -> Dict:
    connected = bool(summary.get("total_tracks", 0) > 0 and summary.get("total_frames", 0) > 0)
    return gate("Real Data Gate", connected, {"total_tracks": summary.get("total_tracks"), "total_frames": summary.get("total_frames"), "dataset_name": summary.get("dataset_name")}, "至少一个真实长轨迹数据源被读取并产生 trajectory table。", "提供 TGSIM/TrajNet/ETH/SDD 路径并确认 loader 解析出 tracks。")


def gate_verified_horizon(summary: Dict) -> Dict:
    ok = bool(summary.get("whether_t100_verified", False))
    return gate("Verified Horizon Gate", ok, {"samples_t100": summary.get("samples_t100"), "whether_t100_verified": summary.get("whether_t100_verified")}, "真实数据可构建 t+100 supervised windows。", "使用更长轨迹或降低采样间隔；不能用 free-run 代替真值。")


def gate_learned_dynamics(metrics: Dict) -> Dict:
    hand = metrics.get("hand_physics_baseline", {})
    det = metrics.get("deterministic_neural_residual", {})
    sto = metrics.get("stochastic_neural_residual", {})
    evidence = {}
    passed = False
    if hand.get("available") and "100" in hand.get("horizons", {}):
        hand_ade = hand["horizons"]["100"]["ADE"]
        hand_fde = hand["horizons"]["100"]["FDE"]
        best_ade = min(det.get("horizons", {}).get("100", {}).get("ADE", 1e9), sto.get("horizons", {}).get("100", {}).get("ADE", 1e9))
        best_fde = min(det.get("horizons", {}).get("100", {}).get("FDE", 1e9), sto.get("horizons", {}).get("100", {}).get("FDE", 1e9))
        ade_gain = (hand_ade - best_ade) / max(1e-9, hand_ade)
        fde_gain = (hand_fde - best_fde) / max(1e-9, hand_fde)
        passed = ade_gain >= 0.05 or fde_gain >= 0.05
        evidence = {"hand_ADE100": hand_ade, "best_learned_ADE100": best_ade, "ADE_gain": ade_gain, "hand_FDE100": hand_fde, "best_learned_FDE100": best_fde, "FDE_gain": fde_gain}
    else:
        evidence = {"reason": "No verified ADE@100/FDE@100 metrics available."}
    return gate("Learned Dynamics Gate", passed, evidence, "learned residual 在真实 t+100 上至少比 hand physics 好 5%。", "训练真实数据 residual；加入多步 rollout loss；不要只优化 one-step。")


def gate_coverage(metrics: Dict) -> Dict:
    hand = metrics.get("hand_physics_baseline", {})
    smc = metrics.get("physics_plus_neural_residual_SMC", {})
    passed = False
    evidence = {}
    if hand.get("available") and smc.get("available"):
        hand_cov = hand.get("coverage_FDE_lt_5m", 0)
        smc_cov = smc.get("coverage_FDE_lt_5m", 0)
        passed = smc_cov > hand_cov
        evidence = {"hand_coverage_FDE_lt_5m": hand_cov, "smc_coverage_FDE_lt_5m": smc_cov}
    else:
        evidence = {"reason": "Coverage metrics unavailable."}
    return gate("Coverage Gate", passed, evidence, "多分支模型提高 coverage_FDE_lt_5m 或 minFDE@N。", "让 SMC particle 表达 goal/intent，而不是只加局部噪声。")


def gate_physical_validity(metrics: Dict) -> Dict:
    hand = metrics.get("hand_physics_baseline", {})
    learned = metrics.get("deterministic_neural_residual", {})
    passed = False
    evidence = {}
    if hand.get("available") and learned.get("available"):
        hand_valid = hand.get("physical_validity_rate", 0)
        learned_valid = learned.get("physical_validity_rate", 0)
        passed = learned_valid >= hand_valid - 0.03
        evidence = {"hand_physical_validity": hand_valid, "learned_physical_validity": learned_valid}
    else:
        evidence = {"reason": "Physical validity metrics unavailable."}
    return gate("Physical Validity Gate", passed, evidence, "learned model 不应明显恶化 collision/boundary/obstacle violation。", "把 projection cost 和真实几何约束纳入 loss/weight。")


def gate_semantic_diversity(metrics: Dict) -> Dict:
    smc = metrics.get("physics_plus_neural_residual_SMC", {})
    diversity = smc.get("cluster_diversity_score") if smc.get("available") else None
    passed = bool(diversity is not None and diversity >= 0.55)
    return gate("Semantic Diversity Gate", passed, {"cluster_diversity_score": diversity, "semantic_event_accuracy": smc.get("semantic_event_accuracy") if smc.get("available") else None}, "terminal clusters 至少产生 3 个可信语义模式。", "真实数据需要语义标签或可解释事件特征；否则只能报告 diversity，不能报告 event accuracy。")


def gate_audit_score(score: float) -> Dict:
    passed = score >= 70
    return gate("Audit Score Gate", passed, {"expert_audit_score": score, "target": 70}, "expert audit score >= 70。", "先过真实 t+100、learned dynamics、coverage 三个硬门槛。")


def gate(name: str, passed: bool, evidence: Dict, explanation: str, fix: str) -> Dict:
    return {"gate": name, "pass": bool(passed), "evidence": evidence, "short_explanation": explanation, "next_fix": fix}


def render_gate_markdown(payload: Dict) -> str:
    lines = ["# Stage 4 World Model Gates", "", f"Passed: `{payload['passed']}/{payload['total']}`", f"Overall pass: `{payload['overall_pass']}`", "", "| Gate | Pass | Evidence | Explanation | Next Fix |", "| --- | --- | --- | --- | --- |"]
    for gate_row in payload["gates"]:
        lines.append(f"| {gate_row['gate']} | {gate_row['pass']} | `{json.dumps(gate_row['evidence'], ensure_ascii=False)}` | {gate_row['short_explanation']} | {gate_row['next_fix']} |")
    return "\n".join(lines) + "\n"
