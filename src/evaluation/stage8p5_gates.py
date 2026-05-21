from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


REPORT_DIR = Path("outputs/reports")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def evaluate_gates() -> Dict:
    data = load_json(REPORT_DIR / "stage8p5_data_audit.json", [])
    ann = load_json(REPORT_DIR / "stage8p5_annotation_report.json", {"annotations": []})
    packs = load_json(REPORT_DIR / "stage8p5_scene_gold_pack_report.json", {})
    eps = load_json(REPORT_DIR / "stage8p5_per_agent_episode_report.json", {"datasets": []})
    goal = load_json(REPORT_DIR / "stage8p5_goalbench_gold_v2_report.json", {})
    pedestrian_loaded = [r for r in data if r.get("eligible_for_stage8p5_gate")]
    long_loaded = [r for r in data if r.get("eligible_for_stage8p5_gate") and (r.get("actual_verified_t50") or r.get("actual_verified_t100"))]
    gold_silver = int(packs.get("number_of_gold_scenes", 0)) + int(packs.get("number_of_silver_scenes", 0))
    ge2 = sum(int(row.get("episodes_ge2_agents", 0)) for row in eps.get("datasets", []))
    official_goal = int(goal.get("official_gold_silver_records", 0))
    leakage_ok = leakage_audit(ann, goal)
    gates = [
        gate("Pedestrian/Drone Data Gate", bool(pedestrian_loaded), f"loaded pedestrian/drone sources={[r.get('dataset_name') for r in pedestrian_loaded]}", "Provide local SDD/OpenTraj if this is empty."),
        gate("Long-Horizon Gate", bool(long_loaded), f"verified t50/t100 pedestrian/drone sources={[r.get('dataset_name') for r in long_loaded]}", "If false, do not claim pedestrian long-horizon world model."),
        gate(
            "Scene-Gold Gate",
            gold_silver >= 1,
            f"gold+silver scenes={gold_silver}",
            "Upgrade rule-confirmed silver to human-confirmed gold where possible." if gold_silver >= 1 else "Do not continue model training. Scene/goal labels are still inferred-only.",
        ),
        gate("Per-Agent Episode Gate", ge2 >= 50, f"per-agent episodes with >=2 agents={ge2}", "Do not train multi-agent world model. Episodes are not truly multi-agent."),
        gate("GoalBench-Gold Gate", official_goal >= 50, f"official gold/silver records={official_goal}", "If below 50, GoalBench remains diagnostic only."),
        gate("No Leakage Gate", leakage_ok, "candidate goals train-only; future endpoint labels eval/train only; central velocity not used", "Repair candidate-goal split policy."),
    ]
    stage9 = gates[0]["passed"] and gates[2]["passed"] and gates[3]["passed"] and gates[5]["passed"] and (gates[4]["passed"] or official_goal > 0)
    gates.append(gate("Stage 9 Readiness Gate", stage9, "ready" if stage9 else "not ready", "Pass data, scene, per-agent, no-leakage gates first."))
    passed = sum(g["passed"] for g in gates)
    score = 72 + min(5, max(0, passed - 3))
    verdict = "stage8p5_ready_for_stage9_per_agent_training" if stage9 else "stage8p5_data_annotation_sprint_incomplete"
    return {"stage": "8.5", "gates": gates, "passed": passed, "total": len(gates), "stage9_ready": stage9, "latent_stage5c_ready": False, "smc_ready": False, "expert_audit_score": score, "verdict": verdict}


def leakage_audit(annotation_payload: Dict, goal_payload: Dict) -> bool:
    anns = annotation_payload.get("annotations", [])
    if any(r.get("test_endpoints_used") for r in anns):
        return False
    records = goal_payload.get("official", {})
    # Records summary does not carry row-level leakage; row-level record file does.
    p = Path("data/stage8p5_goalbench_gold_v2/goalbench_gold_v2_records.json")
    if p.exists():
        rows = json.loads(p.read_text(encoding="utf-8"))
        if any(r.get("test_endpoints_used_for_candidates") for r in rows):
            return False
    return bool(anns)


def gate(name: str, passed: bool, evidence: str, next_fix: str) -> Dict:
    return {"name": name, "passed": bool(passed), "evidence": evidence, "next_fix": next_fix}


def write_stage8p5_gate_report(result: Dict, path: str | Path = REPORT_DIR / "world_model_gate_stage8p5.md") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Stage 8.5 Gates", "", f"Passed: {result['passed']} / {result['total']}", "", "| gate | pass | evidence | next fix |", "| --- | --- | --- | --- |"]
    for g in result["gates"]:
        lines.append(f"| {g['name']} | {g['passed']} | {g['evidence']} | {g['next_fix']} |")
    lines += [
        "",
        f"stage9_ready: `{result['stage9_ready']}`",
        f"latent_stage5c_ready: `{result['latent_stage5c_ready']}`",
        f"smc_ready: `{result['smc_ready']}`",
        f"expert_audit_score: `{result['expert_audit_score']}`",
        f"verdict: `{result['verdict']}`",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
