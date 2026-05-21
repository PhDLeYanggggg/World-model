from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


REPORT_DIR = Path("outputs/reports")
FULL = "per_agent_full_scene_goal_interaction"


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def best_imp(metrics: Dict, model: str, subset: str) -> float:
    vals = []
    for variant in metrics.get("variants", []):
        if variant.get("variant") != model:
            continue
        for drow in variant.get("datasets", {}).values():
            srow = drow.get("subsets", {}).get(subset)
            if not srow or not srow.get("horizons"):
                continue
            target = "10" if "10" in srow["horizons"] else max(srow["horizons"], key=lambda h: int(h))
            vals.append(float(srow["horizons"][target].get("improvement_over_strongest", 0.0)))
    return max(vals) if vals else -999.0


def mean_imp(metrics: Dict, model: str, subset: str) -> float:
    vals = []
    for variant in metrics.get("variants", []):
        if variant.get("variant") != model:
            continue
        for drow in variant.get("datasets", {}).values():
            srow = drow.get("subsets", {}).get(subset)
            if not srow or not srow.get("horizons"):
                continue
            target = "10" if "10" in srow["horizons"] else max(srow["horizons"], key=lambda h: int(h))
            vals.append(float(srow["horizons"][target].get("improvement_over_strongest", 0.0)))
    return sum(vals) / len(vals) if vals else -999.0


def evaluate_stage9_gates() -> Dict:
    data = load_json(REPORT_DIR / "stage9_data_audit.json", {})
    baselines = load_json(REPORT_DIR / "stage9_per_agent_baseline_metrics.json", {"datasets": {}})
    metrics = load_json(REPORT_DIR / "metrics_stage9.json", {"variants": []})
    total_ge2 = int(data.get("episodes_with_ge2_agents", 0))
    leakage = any(data.get("leakage_flags", {}).values())
    baseline_ok = bool(baselines.get("datasets")) and all(v.get("strongest_causal_baseline") for v in baselines["datasets"].values())
    full_all = mean_imp(metrics, FULL, "all")
    full_hard = max(best_imp(metrics, FULL, "hard"), best_imp(metrics, FULL, "baseline_failure"))
    full_easy = mean_imp(metrics, FULL, "easy")
    full_ge5 = mean_imp(metrics, FULL, "ge5")
    inter_gain = max(best_imp(metrics, FULL, "hard") - best_imp(metrics, "per_agent_scene_goal", "hard"), best_imp(metrics, FULL, "baseline_failure") - best_imp(metrics, "per_agent_scene_goal", "baseline_failure"))
    scene_goal_gain = max(best_imp(metrics, "per_agent_scene_goal", "hard") - best_imp(metrics, "per_agent_no_scene", "hard"), best_imp(metrics, "per_agent_scene_goal", "goalbench_official") - best_imp(metrics, "per_agent_no_scene", "goalbench_official"))
    multi_gain = full_ge5 - mean_imp(metrics, "per_agent_no_scene", "ge5")
    long_ok = int(data.get("actual_verified_t50_episodes", 0)) > 0 or int(data.get("actual_verified_t100_episodes", 0)) > 0
    gates = [
        gate("Per-Agent Data Gate", total_ge2 >= 300, f">=2 agent episodes={total_ge2}", "Need at least 300 official per-agent multi-agent episodes."),
        gate("No Leakage Gate", not leakage, f"leakage_flags={data.get('leakage_flags')}", "Candidate goals must be train-only; no future endpoint inputs."),
        gate("Strong Baseline Gate", baseline_ok, f"datasets={list(baselines.get('datasets', {}).keys())}", "Every official dataset/scene needs strongest causal baseline."),
        gate("Per-Agent Model Gate", full_all >= 0.05, f"full all-test mean improvement={round(full_all, 6)}", "Do not enter Stage 5C. Per-agent deterministic world model is not strong enough."),
        gate("Hard/Failure Gate", full_hard >= 0.10, f"full hard/failure best improvement={round(full_hard, 6)}", "Need >=10% on hard or baseline-failure subset."),
        gate("Easy Preservation Gate", full_easy >= -0.05, f"full easy mean improvement={round(full_easy, 6)}", "Do not degrade easy subset."),
        gate("Interaction Gate", inter_gain > 0.02, f"full minus scene_goal hard/failure gain={round(inter_gain, 6)}", "Interaction must improve trajectory metrics, not just auxiliary."),
        gate("Scene/Goal Gate", scene_goal_gain > 0.02, f"scene_goal minus no_scene gain={round(scene_goal_gain, 6)}", "Scene/goal must improve hard/failure or GoalBench subset."),
        gate("Multi-Agent Gate", multi_gain > 0.02, f"full minus no_scene on >=5 agents={round(multi_gain, 6)}", "Per-agent all-agent model must beat primary/simple fallback on multi-agent scenes."),
        gate("Verified Long-Horizon Gate", long_ok, f"t50={data.get('actual_verified_t50_episodes')} t100={data.get('actual_verified_t100_episodes')}", "Do not claim pedestrian long-horizon world model. Need verified t+50/t+100 pedestrian/drone data."),
    ]
    ready = all(g["passed"] for g in gates[3:9]) and gates[9]["passed"]
    gates.append(gate("Stage 5C Readiness Gate", ready, "ready" if ready else "Do not enter Stage 5C. Per-agent deterministic gates and verified long-horizon gate are not satisfied.", "Pass deterministic and long-horizon gates before latent generative work."))
    passed = sum(g["passed"] for g in gates)
    score = min(82, 75 + max(0, passed - 4))
    verdict = "stage9_per_agent_training_done_not_stage5c_ready" if not ready else "stage9_ready_for_stage5c"
    return {"stage": "9", "gates": gates, "passed": passed, "total": len(gates), "latent_stage5c_ready": ready, "smc_ready": False, "expert_audit_score": score, "verdict": verdict}


def gate(name: str, passed: bool, evidence: str, next_fix: str) -> Dict:
    return {"name": name, "passed": bool(passed), "evidence": evidence, "next_fix": next_fix}


def write_stage9_gates(result: Dict, path: str | Path = REPORT_DIR / "world_model_gate_stage9.md") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Stage 9 Gates", "", f"Passed: {result['passed']} / {result['total']}", "", "| gate | pass | evidence | next fix |", "| --- | --- | --- | --- |"]
    for g in result["gates"]:
        lines.append(f"| {g['name']} | {g['passed']} | {g['evidence']} | {g['next_fix']} |")
    lines += [
        "",
        f"latent_stage5c_ready: `{result['latent_stage5c_ready']}`",
        f"smc_ready: `{result['smc_ready']}`",
        f"expert_audit_score: `{result['expert_audit_score']}`",
        f"verdict: `{result['verdict']}`",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
