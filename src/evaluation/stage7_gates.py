from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


REPORT_DIR = Path("outputs/reports")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def best_improvements(metrics: Dict, subset: str, official_only: bool = True) -> Dict[str, float]:
    allowed = {"goal_conditioned_world_model", "goal_scene_interaction_residual", "goal_interaction_residual", "topk_goal_mixture_diagnostic"}
    out = {}
    for variant in metrics.get("variants", []):
        name = variant.get("variant", "")
        if official_only and name == "topk_goal_mixture_diagnostic":
            continue
        for dataset, drow in variant.get("datasets", {}).items():
            srow = drow.get("subsets", {}).get(subset)
            if not srow:
                continue
            horizons = srow.get("horizons", {})
            if not horizons:
                continue
            target = "100" if "100" in horizons else max(horizons, key=lambda h: int(h))
            imp = float(horizons[target].get("improvement_over_strongest", 0.0))
            out[dataset] = max(out.get(dataset, -999.0), imp)
    return out


def alpha_stats(metrics: Dict) -> Dict:
    rows = []
    for variant in metrics.get("variants", []):
        if variant.get("variant") == "goal_scene_interaction_residual":
            rows = variant.get("alpha_rows", [])
            break
    if not rows and metrics.get("variants"):
        rows = metrics["variants"][0].get("alpha_rows", [])
    easy = [r["alpha"] for r in rows if r.get("subset") == "easy"]
    hard = [r["alpha"] for r in rows if r.get("subset") == "hard"]
    failure = [r["alpha"] for r in rows if r.get("baseline_failure")]
    return {
        "easy": float(np.mean(easy)) if easy else None,
        "hard": float(np.mean(hard)) if hard else None,
        "failure": float(np.mean(failure)) if failure else None,
    }


def evaluate_gates() -> Dict:
    scene_audit = load_json(REPORT_DIR / "stage7_scene_data_audit.json", [])
    scene_pack = load_json(REPORT_DIR / "stage7_scene_pack_report.json", {"scene_packs": []})
    goalbench = load_json(REPORT_DIR / "goalbench_summary_stage7.json", {"datasets": {}})
    goal_metrics = load_json(REPORT_DIR / "goal_predictor_metrics_stage7.json", {})
    failure_cmp = load_json(REPORT_DIR / "stage7_failure_predictor_comparison.json", {"variants": {}, "stage6_reference": {}})
    metrics = load_json(REPORT_DIR / "metrics_stage7.json", {"variants": []})
    interaction_aux = load_json(REPORT_DIR / "stage7_interaction_auxiliary_report.json", {})
    ped_long_ok = any(r.get("eligible_for_pedestrian_drone_long_horizon_gate") for r in scene_audit)
    scene_packs = scene_pack.get("scene_packs", [])
    ped_scene = [p for p in scene_packs if p.get("dataset_name") in {"trajnet", "eth_ucy"}]
    scene_pack_ok = len(scene_packs) >= 2 or len(ped_scene) >= 1
    goal_test = goal_metrics.get("test", {})
    goal_ok = bool(goal_test) and goal_test.get("top3_goal_accuracy", 0.0) > goal_test.get("majority_top3", 0.0) + 0.02 and goal_test.get("goal_NLL", 99) < 2.5
    stage6_ref = failure_cmp.get("stage6_reference", {})
    best_fp = max((v.get("test", {}).get("AUROC", 0.0) for v in failure_cmp.get("variants", {}).values()), default=0.0)
    hard_best_fp = max((v.get("test", {}).get("AUPRC", 0.0) for v in failure_cmp.get("variants", {}).values()), default=0.0)
    fp_ok = best_fp > float(stage6_ref.get("AUROC", 0.0)) + 0.01 or hard_best_fp > float(stage6_ref.get("AUPRC", 0.0)) + 0.02
    failure_imp = best_improvements(metrics, "baseline_failure")
    hard_imp = best_improvements(metrics, "hard")
    easy_imp = best_improvements(metrics, "easy")
    long_imp = {**best_improvements(metrics, "verified_t50"), **best_improvements(metrics, "verified_t100")}
    failure_ok = any(v >= 0.10 for v in failure_imp.values())
    hard_ok = any(v >= 0.10 for v in hard_imp.values())
    easy_ok = all(v >= -0.05 for v in easy_imp.values()) if easy_imp else False
    interaction_ok = bool(interaction_aux.get("metrics", {}).get("improves_hard_failure_trajectory_performance", False))
    long_ok = any(v >= 0.05 for v in long_imp.values())
    gates = [
        gate("Pedestrian/Drone Long-Horizon Gate", ped_long_ok, f"{sum(1 for r in scene_audit if r.get('eligible_for_pedestrian_drone_long_horizon_gate'))} pedestrian/drone sources support verified t+50/t+100", "Add SDD/OpenTraj/full pedestrian data with verified long horizon."),
        gate("Scene Pack Gate", scene_pack_ok, f"{len(scene_packs)} scene packs; pedestrian scene packs={len(ped_scene)}", "Add real scene images/homographies/walkable annotations."),
        gate("GoalBench Gate", goal_ok, f"test top3={goal_test.get('top3_goal_accuracy')}, majority_top3={goal_test.get('majority_top3')}, NLL={goal_test.get('goal_NLL')}", "Improve candidate goals; avoid too-few-goal top3 saturation."),
        gate("Failure Predictor Improvement Gate", fp_ok, f"stage6_AUROC={stage6_ref.get('AUROC')}, best_stage7_AUROC={best_fp}, best_stage7_AUPRC={hard_best_fp}", "Goal/scene features must improve failure prediction."),
        gate("Failure Correction Gate", failure_ok, f"BaselineFailureBench improvements={failure_imp}", "Need >=10% improvement on BaselineFailureBench."),
        gate("HardBench Gate", hard_ok, f"HardBench improvements={hard_imp}", "Need >=10% improvement on official hard subset."),
        gate("Easy Preservation Gate", easy_ok, f"easy improvements={easy_imp}", "Do not degrade easy cases."),
        gate("Interaction Auxiliary Gate", interaction_ok, f"metrics={interaction_aux.get('metrics', {})}", "Auxiliary interaction tasks must improve hard/failure trajectory metrics."),
        gate("Verified Long-Horizon Gate", long_ok, f"verified long-horizon improvements={long_imp}", "Need >=5% on at least one verified t+50/t+100 source."),
    ]
    ready = gates[2]["passed"] and gates[3]["passed"] and gates[4]["passed"] and gates[5]["passed"] and gates[6]["passed"] and (gates[0]["passed"] or gates[8]["passed"])
    gates.append(gate("Stage 5C Readiness Gate", ready, "ready" if ready else "Do not enter Stage 5C. Goal-conditioned deterministic world model is not strong enough.", "Pass Stage 7 deterministic gates first."))
    passed = sum(g["passed"] for g in gates)
    score = 70 + max(0, passed - 4)
    verdict = "stage7_scene_goal_grounding_built_but_not_stage5c_ready" if not ready else "stage7_ready_for_stage5c"
    return {"stage": "7", "gates": gates, "passed": passed, "total": len(gates), "latent_stage5c_ready": ready, "smc_ready": False, "expert_audit_score": score, "verdict": verdict}


def gate(name: str, passed: bool, evidence: str, next_fix: str) -> Dict:
    return {"name": name, "passed": bool(passed), "evidence": evidence, "next_fix": next_fix}


def write_report(result: Dict, path: str | Path = REPORT_DIR / "world_model_gate_stage7.md") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Stage 7 Gates", "", f"Passed: {result['passed']} / {result['total']}", "", "| gate | pass | evidence | next fix |", "| --- | --- | --- | --- |"]
    for g in result["gates"]:
        lines.append(f"| {g['name']} | {g['passed']} | {g['evidence']} | {g['next_fix']} |")
    lines += ["", f"latent_stage5c_ready: `{result['latent_stage5c_ready']}`", f"smc_ready: `{result['smc_ready']}`", f"expert_audit_score: `{result['expert_audit_score']}`", f"verdict: `{result['verdict']}`"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

