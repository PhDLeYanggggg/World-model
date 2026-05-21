from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


REPORT_DIR = Path("outputs/reports")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def best_improvements(metrics: Dict, subset: str, exclude_diagnostic: bool = True) -> Dict[str, float]:
    out = {}
    for variant in metrics.get("variants", []):
        name = variant.get("variant", "")
        if exclude_diagnostic and "diagnostic" in name:
            continue
        for dataset, drow in variant.get("datasets", {}).items():
            srow = drow.get("subsets", {}).get(subset)
            if not srow or not srow.get("horizons"):
                continue
            target = "100" if "100" in srow["horizons"] else max(srow["horizons"], key=lambda h: int(h))
            imp = float(srow["horizons"][target].get("improvement_over_strongest", 0.0))
            out[dataset] = max(out.get(dataset, -999.0), imp)
    return out


def variant_improvement(metrics: Dict, variant_name: str, subset: str) -> Dict[str, float]:
    out = {}
    for variant in metrics.get("variants", []):
        if variant.get("variant") != variant_name:
            continue
        for dataset, drow in variant.get("datasets", {}).items():
            srow = drow.get("subsets", {}).get(subset)
            if not srow or not srow.get("horizons"):
                continue
            target = "100" if "100" in srow["horizons"] else max(srow["horizons"], key=lambda h: int(h))
            out[dataset] = float(srow["horizons"][target].get("improvement_over_strongest", 0.0))
    return out


def evaluate_gates() -> Dict:
    data_audit = load_json(REPORT_DIR / "stage8_data_audit.json", [])
    scene = load_json(REPORT_DIR / "stage8_scene_gold_report.json", {})
    multi = load_json(REPORT_DIR / "stage8_multiagent_episode_report.json", {"datasets": []})
    goal = load_json(REPORT_DIR / "stage8_goal_predictor_metrics.json", {})
    failure = load_json(REPORT_DIR / "stage8_failure_predictor_comparison.json", {"variants": {}, "stage7_reference": {}})
    metrics = load_json(REPORT_DIR / "metrics_stage8.json", {"variants": []})
    interaction = load_json(REPORT_DIR / "stage8_interaction_ablation.json", {"metrics": {}})

    ped_long = any(r.get("whether_eligible_for_stage8_gate") and (r.get("actual_verified_t50") or r.get("actual_verified_t100")) for r in data_audit)
    gold_or_silver = int(scene.get("gold_scenes", 0)) + int(scene.get("silver_scenes", 0))
    usable_scene = int(scene.get("total_scene_packs", 0))
    scene_gate = gold_or_silver >= 1 or usable_scene >= 2
    ge2 = sum(int(row.get("episodes_with_ge2_agents", 0)) for row in multi.get("datasets", []))
    multi_gate = ge2 >= 50
    goal_test = goal.get("test", {})
    top3_saturated = bool(goal_test.get("top3_saturated", False) or goal_test.get("majority_top3", 0.0) >= 0.95)
    goal_gate = bool(goal_test) and (
        goal_test.get("top1_accuracy", 0.0) > goal_test.get("majority_top1", 0.0) + 0.02
        or goal_test.get("top3_accuracy", 0.0) > goal_test.get("majority_top3", 0.0) + 0.02
        or (top3_saturated and goal_test.get("goal_NLL", 99) < 2.5 and goal_test.get("goal_ECE", 99) < 0.35)
    )
    stage7_auc = float(failure.get("stage7_reference", {}).get("best_stage7_test_AUROC", 0.0))
    best_failure_auc = max((v.get("test", {}).get("AUROC", 0.0) for v in failure.get("variants", {}).values()), default=0.0)
    failure_gate = best_failure_auc > stage7_auc + 0.005
    failure_imp = best_improvements(metrics, "baseline_failure")
    hard_imp = best_improvements(metrics, "hardbench")
    easy_imp = best_improvements(metrics, "easy")
    long_imp = {**best_improvements(metrics, "verified_t50"), **best_improvements(metrics, "verified_t100")}
    failure_corr_gate = any(v >= 0.10 for v in failure_imp.values())
    hard_gate = any(v >= 0.10 for v in hard_imp.values())
    easy_gate = bool(easy_imp) and all(v >= -0.05 for v in easy_imp.values())
    full_hard = variant_improvement(metrics, "scene_goal_multiagent_v2", "hardbench")
    no_inter_hard = variant_improvement(metrics, "scene_goal_v2", "hardbench")
    interaction_gate = bool(full_hard) and any(full_hard.get(ds, -999) > no_inter_hard.get(ds, -999) + 0.02 for ds in full_hard)
    if interaction.get("metrics", {}).get("improves_hard_failure_trajectory_performance"):
        interaction_gate = True
    long_gate = any(v >= 0.05 for v in long_imp.values())
    gates = [
        gate("Pedestrian/Drone Long-Horizon Gate", ped_long, f"eligible pedestrian/drone long-horizon sources={sum(1 for r in data_audit if r.get('whether_eligible_for_stage8_gate'))}", "Provide local SDD/OpenTraj/full pedestrian data with verified t+50/t+100."),
        gate("Scene-Gold Gate", scene_gate, f"gold/silver={gold_or_silver}; usable_scene_packs={usable_scene}", "Manually confirm at least one pedestrian/drone scene or two usable scene packs."),
        gate("Multi-Agent Episode Gate", multi_gate, f"episodes_with_ge2_agents={ge2}", "Build more multi-agent windows from real pedestrian/drone scenes."),
        gate("GoalBench-Gold Gate", goal_gate, f"test={goal_test}", "Improve gold/silver goals; avoid majority top-k saturation."),
        gate("Failure Predictor Gate", failure_gate, f"stage7_best_AUROC={stage7_auc}; stage8_best_AUROC={best_failure_auc}", "Scene/goal/multi-agent features should improve failure prediction."),
        gate("Failure Correction Gate", failure_corr_gate, f"BaselineFailureBench improvements={failure_imp}", "Need >=10% over strongest baseline on BaselineFailureBench."),
        gate("HardBench Gate", hard_gate, f"HardBench improvements={hard_imp}", "Need >=10% over strongest baseline on HardBench."),
        gate("Easy Preservation Gate", easy_gate, f"easy improvements={easy_imp}", "Keep easy cases near baseline."),
        gate("Interaction Gate", interaction_gate, f"full_hard={full_hard}; no_interaction_hard={no_inter_hard}; aux={interaction.get('metrics', {})}", "Multi-agent interaction must improve hard/failure trajectory metrics."),
        gate("Verified Long-Horizon Gate", long_gate, f"verified long-horizon improvements={long_imp}", "Need >=5% on verified t+50/t+100."),
    ]
    ready = gates[2]["passed"] and gates[3]["passed"] and gates[4]["passed"] and gates[5]["passed"] and gates[6]["passed"] and gates[7]["passed"] and gates[8]["passed"] and (gates[0]["passed"] or gates[9]["passed"])
    gates.append(gate("Stage 5C Readiness Gate", ready, "ready" if ready else "Do not enter Stage 5C. Scene/goal-conditioned deterministic correction is not strong enough.", "Pass Stage 8 deterministic scene/goal gates first."))
    passed = sum(g["passed"] for g in gates)
    score = min(80, 71 + max(0, passed - 4))
    verdict = "stage8_scene_goal_multiagent_scaffold_not_stage5c_ready" if not ready else "stage8_ready_for_stage5c"
    return {"stage": "8", "gates": gates, "passed": passed, "total": len(gates), "latent_stage5c_ready": ready, "smc_ready": False, "expert_audit_score": score, "verdict": verdict}


def gate(name: str, passed: bool, evidence: str, next_fix: str) -> Dict:
    return {"name": name, "passed": bool(passed), "evidence": evidence, "next_fix": next_fix}


def write_report(result: Dict, path: str | Path = REPORT_DIR / "world_model_gate_stage8.md") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Stage 8 Gates", "", f"Passed: {result['passed']} / {result['total']}", "", "| gate | pass | evidence | next fix |", "| --- | --- | --- | --- |"]
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
