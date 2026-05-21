from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src.stage10_common import REPORT_DIR, read_json, write_json


def evaluate_stage10_gates() -> Dict:
    data = read_json(REPORT_DIR / "stage10_data_audit.json", [])
    ann = read_json(REPORT_DIR / "stage10_annotation_report.json", {"annotations": []})
    packs = read_json(REPORT_DIR / "stage10_scene_pack_report.json", {})
    eps = read_json(REPORT_DIR / "stage10_multiagent_episode_report.json", {"datasets": []})
    hard = read_json(REPORT_DIR / "stage10_hard_failure_report.json", {"summary": {}}).get("summary", {})
    goal = read_json(REPORT_DIR / "stage10_goalbench_v3_report.json", {})
    pedestrian_loaded = [r for r in data if r.get("eligible_for_stage10")]
    long_loaded = [r for r in data if r.get("eligible_for_stage10") and (r.get("actual_verified_t50") or r.get("actual_verified_t100"))]
    human_scenes = sum(r.get("annotation_quality") in {"gold_human", "silver_human_confirmed"} for r in ann.get("annotations", []))
    rule_scenes = sum(r.get("annotation_quality") == "silver_rule_confirmed" for r in ann.get("annotations", []))
    scene_pack_goal = int(packs.get("scenes_with_goals", 0))
    scene_pack_walkable = int(packs.get("scenes_with_walkable", 0))
    ge2 = sum(int(r.get("episodes_ge2_agents", 0)) for r in eps.get("datasets", []))
    hard_count = int(hard.get("hard_episodes", 0)) + int(hard.get("baseline_failure_episodes", 0))
    official_goal = int(goal.get("official_records_count", 0))
    inferred_only_goal = goal.get("records_by_annotation_quality", {}).get("inferred_only", 0) if isinstance(goal.get("records_by_annotation_quality"), dict) else 0
    leakage_ok = no_leakage()
    gates = [
        gate("Pedestrian/Drone Data Gate", bool(pedestrian_loaded), "pass" if pedestrian_loaded else "fail", f"loaded={[r['dataset_name'] for r in pedestrian_loaded]}", "Load at least one real pedestrian/drone source."),
        gate("Long-Horizon Gate", bool(long_loaded), "pass" if long_loaded else "fail", f"verified_t50_or_t100={[r['dataset_name'] for r in long_loaded]}", "Cannot claim pedestrian long-horizon world model until this passes."),
        gate("Human/Silver Annotation Gate", human_scenes >= 3, "pass" if human_scenes >= 3 else ("partial" if rule_scenes >= 3 else "fail"), f"human_confirmed={human_scenes}, silver_rule_confirmed={rule_scenes}", "Need at least 3 gold_human or silver_human_confirmed scenes."),
        gate("Scene Pack Gate", scene_pack_goal >= 3 and scene_pack_walkable >= 3, "pass" if scene_pack_goal >= 3 and scene_pack_walkable >= 3 else "fail", f"walkable={scene_pack_walkable}, goals={scene_pack_goal}", "Need usable walkable + goal scene packs."),
        gate("Multi-Agent Episode Gate", ge2 >= 500, "pass" if ge2 >= 500 else ("partial" if ge2 > 0 else "fail"), f">=2_agent_episodes={ge2}", "Need 500 multi-agent episodes or mark partial."),
        gate("Hard/Failure Episode Gate", hard_count >= 100, "pass" if hard_count >= 100 else ("partial" if hard_count > 0 else "fail"), f"hard_plus_failure={hard_count}", "Need at least 100 hard/failure episodes."),
        gate("GoalBench v3 Gate", official_goal >= 500 and official_goal > inferred_only_goal, "pass" if official_goal >= 500 and official_goal > inferred_only_goal else ("partial" if official_goal > 0 else "fail"), f"official_records={official_goal}, inferred_only_records={inferred_only_goal}", "Need 500 official non-inferred records."),
        gate("No Leakage Gate", leakage_ok, "pass" if leakage_ok else "fail", "train-only candidate goals; no future endpoint input; causal velocity", "Repair leakage flags."),
    ]
    stage11_ready = gates[0]["passed"] and gates[2]["passed"] and gates[3]["passed"] and gates[6]["passed"] and gates[7]["passed"] and gates[4]["status"] in {"pass", "partial"}
    gates.append(gate("Stage 11 Readiness Gate", stage11_ready, "pass" if stage11_ready else "fail", "ready" if stage11_ready else "not_ready", "Pass data, human annotation, scene pack, GoalBench and no-leakage gates."))
    gates.append(gate("Stage 5C Readiness Gate", False, "fail", "Stage 10 is data/annotation only; latent generative remains forbidden.", "Keep disabled."))
    passed = sum(g["passed"] for g in gates)
    score = 75 + (2 if gates[3]["passed"] else 0) + (2 if gates[6]["passed"] else 0) - (0 if stage11_ready else 1)
    verdict = "stage10_ready_for_stage11_training" if stage11_ready else "stage10_data_annotation_package_partial_not_stage11_ready"
    return {"stage": "10", "gates": gates, "passed": passed, "total": len(gates), "stage11_ready": stage11_ready, "latent_stage5c_ready": False, "smc_ready": False, "expert_audit_score": score, "verdict": verdict}


def no_leakage() -> bool:
    records = read_json("data/stage10_goalbench_v3/goalbench_v3_records.json", [])
    if any(r.get("test_endpoints_used_for_candidates") or r.get("future_endpoint_used_as_input") for r in records):
        return False
    anns = read_json(REPORT_DIR / "stage10_annotation_report.json", {"annotations": []}).get("annotations", [])
    if any(r.get("test_endpoints_used") for r in anns):
        return False
    return bool(anns)


def gate(name: str, passed: bool, status: str, evidence: str, next_fix: str) -> Dict:
    return {"name": name, "passed": bool(passed), "status": status, "evidence": evidence, "next_fix": next_fix}


def write_stage10_gate_report(result: Dict) -> None:
    write_json(REPORT_DIR / "world_model_gate_stage10.json", result)
    lines = ["# Stage 10 Gates", "", f"Passed: {result['passed']} / {result['total']}", "", "| gate | status | pass | evidence | next fix |", "| --- | --- | --- | --- | --- |"]
    for g in result["gates"]:
        lines.append(f"| {g['name']} | {g['status']} | {g['passed']} | {g['evidence']} | {g['next_fix']} |")
    lines += [
        "",
        f"stage11_ready: `{result['stage11_ready']}`",
        f"latent_stage5c_ready: `{result['latent_stage5c_ready']}`",
        f"smc_ready: `{result['smc_ready']}`",
        f"expert_audit_score: `{result['expert_audit_score']}`",
        f"verdict: `{result['verdict']}`",
    ]
    (REPORT_DIR / "world_model_gate_stage10.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = evaluate_stage10_gates()
    write_stage10_gate_report(result)
    print(json.dumps({"passed": result["passed"], "total": result["total"], "stage11_ready": result["stage11_ready"]}, indent=2))


if __name__ == "__main__":
    main()
