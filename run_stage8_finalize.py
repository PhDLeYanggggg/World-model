from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List


REPORT_DIR = Path("outputs/reports")
PACKAGE = Path("outputs/world_model_stage8_results")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def main() -> None:
    gate = load_json(REPORT_DIR / "world_model_gate_stage8.json", {})
    data_audit = load_json(REPORT_DIR / "stage8_data_audit.json", [])
    scene = load_json(REPORT_DIR / "stage8_scene_gold_report.json", {})
    multi = load_json(REPORT_DIR / "stage8_multiagent_episode_report.json", {"datasets": []})
    goal = load_json(REPORT_DIR / "stage8_goal_predictor_metrics.json", {})
    failure = load_json(REPORT_DIR / "stage8_failure_predictor_comparison.json", {})
    metrics = load_json(REPORT_DIR / "metrics_stage8.json", {})
    interaction = load_json(REPORT_DIR / "stage8_interaction_ablation.json", {})

    write_final_report(gate, data_audit, scene, multi, goal, failure, metrics, interaction)
    write_failure_analysis(gate, data_audit, scene, goal, metrics)
    write_model_card(gate)
    write_data_card(data_audit, scene, multi)
    write_next_steps(gate)
    package_outputs()
    print(f"Stage 8 package: {PACKAGE.resolve()}")


def write_final_report(gate: Dict, data_audit: List[Dict], scene: Dict, multi: Dict, goal: Dict, failure: Dict, metrics: Dict, interaction: Dict) -> None:
    ped_long = [r for r in data_audit if r.get("whether_eligible_for_stage8_gate")]
    ge2 = sum(int(row.get("episodes_with_ge2_agents", 0)) for row in multi.get("datasets", []))
    goal_test = goal.get("test", {})
    best_failure = max((v.get("test", {}).get("AUROC", 0.0) for v in failure.get("variants", {}).values()), default=0.0)
    best_failure_imp = best_improvement(metrics, "baseline_failure")
    best_hard_imp = best_improvement(metrics, "hardbench")
    best_long_imp = max(best_improvement(metrics, "verified_t50"), best_improvement(metrics, "verified_t100"))
    lines = [
        "# Stage 8 Final Report",
        "",
        "Stage 8 upgrades the scaffold toward a scene/goal-grounded pedestrian world model, but it does not make the project a true 3D or large-scale foundation world model.",
        "",
        "## Current Status",
        "",
        "- The system is still a multi-source 2.5D/pseudo-3D trajectory world-state benchmark scaffold.",
        "- Latent generative modeling remains disabled.",
        "- SMC remains disabled.",
        "- Traffic t+100 results are not pedestrian world-model evidence.",
        "- Inferred goals are not true goals.",
        "",
        "## Direct Answers",
        "",
        f"1. pedestrian/drone t+50/t+100 replenished: {'yes' if ped_long else 'no'}",
        f"2. gold/silver scene annotations: {int(scene.get('gold_scenes', 0)) + int(scene.get('silver_scenes', 0))}",
        f"3. true multi-agent episodes built: {'yes' if ge2 >= 50 else 'partial'} ({ge2} episodes with >=2 agents)",
        f"4. GoalBench-Gold vs majority: top1={goal_test.get('top1_accuracy')}, majority_top1={goal_test.get('majority_top1')}; top3={goal_test.get('top3_accuracy')}, majority_top3={goal_test.get('majority_top3')}",
        f"5. Stage 8 failure predictor best AUROC: {best_failure}",
        f"6. BaselineFailureBench best improvement: {best_failure_imp}",
        f"7. HardBench best improvement: {best_hard_imp}",
        f"8. interaction encoder trajectory evidence: {interaction.get('metrics', {})}",
        f"9. verified long-horizon best improvement: {best_long_imp}",
        f"10. Stage 5C ready: {gate.get('latent_stage5c_ready', False)}",
        f"11. SMC ready: {gate.get('smc_ready', False)}",
        "",
        "## Final Conclusion",
        "",
        f"项目是否跑通：{'是' if gate else '否'}",
        f"pedestrian/drone long-horizon 是否补上：{'是' if ped_long else '否'}",
        f"scene-gold annotation 是否建立：{'是' if int(scene.get('gold_scenes', 0)) + int(scene.get('silver_scenes', 0)) > 0 else '部分'}",
        f"multi-agent episodes 是否建立：{'是' if ge2 >= 50 else '部分'}",
        f"GoalBench-Gold 是否有效：{'是' if goal_test.get('beats_majority') else '部分'}",
        f"failure predictor 是否改善：{'是' if best_failure > failure.get('stage7_reference', {}).get('best_stage7_test_AUROC', 0.0) else '否'}",
        f"goal-conditioned world model 是否改善 failure/hard cases：{'是' if best_failure_imp >= 0.10 and best_hard_imp >= 0.10 else '部分' if best_failure_imp > 0 or best_hard_imp > 0 else '否'}",
        f"interaction encoder 是否有效：{'是' if interaction.get('metrics', {}).get('improves_hard_failure_trajectory_performance') else '否'}",
        f"verified long-horizon 是否改善：{'是' if best_long_imp >= 0.05 else '否'}",
        f"latent generative Stage 5C 是否 ready：{'是' if gate.get('latent_stage5c_ready') else '否'}",
        f"SMC 是否 ready：{'是' if gate.get('smc_ready') else '否'}",
        f"当前 verdict：{gate.get('verdict', 'unknown')}",
        f"expert audit score：{gate.get('expert_audit_score', 'unknown')}",
        "",
        "如果不能进入 Stage 5C，下一步先修什么：",
        "",
        "1. 提供或接入本地 Stanford Drone Dataset / OpenTraj 原始数据，并确认 license 与路径。",
        "2. 对至少一个 pedestrian/drone scene 做 gold/silver exit/goal/walkable annotation，而不是只用 endpoint inferred goals。",
        "3. 把 world model 从 primary-agent residual 升级为 per-agent multi-agent residual，并在 HardBench/BaselineFailureBench 上稳定超过 strongest causal baseline。",
    ]
    (REPORT_DIR / "report_stage8_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_improvement(metrics: Dict, subset: str) -> float:
    best = -999.0
    for variant in metrics.get("variants", []):
        if "diagnostic" in variant.get("variant", ""):
            continue
        for drow in variant.get("datasets", {}).values():
            srow = drow.get("subsets", {}).get(subset)
            if not srow or not srow.get("horizons"):
                continue
            target = "100" if "100" in srow["horizons"] else max(srow["horizons"], key=lambda h: int(h))
            best = max(best, float(srow["horizons"][target].get("improvement_over_strongest", 0.0)))
    return 0.0 if best == -999.0 else round(best, 6)


def write_failure_analysis(gate: Dict, data_audit: List[Dict], scene: Dict, goal: Dict, metrics: Dict) -> None:
    lines = [
        "# Stage 8 Failure Analysis",
        "",
        "The largest failures are data-grounding failures, not a lack of larger neural architecture.",
        "",
        "- No local SDD/OpenTraj long-horizon pedestrian/drone source was verified during this run.",
        f"- Gold/silver scene annotations: {int(scene.get('gold_scenes', 0)) + int(scene.get('silver_scenes', 0))}; most scene goals remain inferred-only.",
        f"- Goal predictor test metrics: {goal.get('test', {})}",
        f"- Best BaselineFailureBench improvement: {best_improvement(metrics, 'baseline_failure')}",
        f"- Best HardBench improvement: {best_improvement(metrics, 'hardbench')}",
        "- The residual head corrects only the primary agent in the multi-agent episode, so it is not yet a full multi-agent dynamics model.",
        "- Do not enter Stage 5C until deterministic scene/goal correction passes the failure/hard gates.",
    ]
    (REPORT_DIR / "failure_analysis_stage8.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_card(gate: Dict) -> None:
    lines = [
        "# Stage 8 Model Card",
        "",
        "Model type: deterministic scene/goal-conditioned bounded residual over strongest causal baseline.",
        "",
        "Not true 3D. Not latent generative. Not SMC. Top-k goal diagnostics are deterministic candidate evaluations only.",
        "",
        f"Gate verdict: {gate.get('verdict', 'unknown')}",
        f"Expert audit score: {gate.get('expert_audit_score', 'unknown')}",
        "",
        "Official prediction form:",
        "",
        "`prediction = strongest_causal_baseline + alpha * bounded_residual(goal, scene, multi_agent_context)`",
        "",
        "Known limits: inferred goals, no verified pedestrian/drone t+50/t+100 in this run, primary-agent residual target, weak interaction trajectory evidence.",
    ]
    (REPORT_DIR / "model_card_stage8.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_data_card(data_audit: List[Dict], scene: Dict, multi: Dict) -> None:
    lines = [
        "# Stage 8 Data Card",
        "",
        "Data sources audited:",
        "",
    ]
    for r in data_audit:
        lines.append(f"- {r.get('dataset_name')}: status={r.get('download_status')}, unit={r.get('coordinate_unit')}, t50={r.get('actual_verified_t50')}, t100={r.get('actual_verified_t100')}, license={r.get('license')}")
    lines += [
        "",
        f"Scene packs: total={scene.get('total_scene_packs', 0)}, gold={scene.get('gold_scenes', 0)}, silver={scene.get('silver_scenes', 0)}, inferred_only={scene.get('inferred_only_scenes', 0)}",
        "",
        "Multi-agent episode summary:",
    ]
    for row in multi.get("datasets", []):
        lines.append(f"- {row.get('dataset_name')}: episodes={row.get('total_episodes')}, mean_agents={row.get('mean_agents_per_episode')}, t50={row.get('verified_t50_episodes')}, t100={row.get('verified_t100_episodes')}")
    (REPORT_DIR / "data_card_stage8.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_next_steps(gate: Dict) -> None:
    lines = [
        "# Stage 8 Next Steps",
        "",
        "1. Add local SDD/OpenTraj pedestrian/drone data and build verified t+50/t+100 episodes.",
        "2. Create human-confirmed Scene-Gold/Silver annotations for walkable areas, exits, and candidate goals.",
        "3. Train per-agent multi-agent residual heads and evaluate on official hard/failure subsets with confidence intervals.",
        "",
        f"Stage 5C allowed now: {gate.get('latent_stage5c_ready', False)}",
        f"SMC allowed now: {gate.get('smc_ready', False)}",
    ]
    (REPORT_DIR / "stage8_next_steps.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_outputs() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    (PACKAGE / "reports").mkdir(exist_ok=True)
    for path in REPORT_DIR.glob("*stage8*"):
        shutil.copy2(path, PACKAGE / "reports" / path.name)
    for name in ["report_stage8_final.md", "failure_analysis_stage8.md", "model_card_stage8.md", "data_card_stage8.md", "world_model_gate_stage8.md", "world_model_gate_stage8.json"]:
        src = REPORT_DIR / name
        if src.exists():
            shutil.copy2(src, PACKAGE / "reports" / src.name)
    for src_dir, dst_name in [
        (Path("data/scene_gold_packs"), "scene_gold_packs"),
        (Path("data/stage8_multiagent_episodes"), "stage8_multiagent_episodes"),
        (Path("data/stage8_goalbench_gold"), "stage8_goalbench_gold"),
        (Path("outputs/checkpoints/stage8"), "checkpoints_stage8"),
    ]:
        dst = PACKAGE / dst_name
        if dst.exists():
            shutil.rmtree(dst)
        if src_dir.exists():
            shutil.copytree(src_dir, dst)
    summary = REPORT_DIR / "report_stage8_final.md"
    if summary.exists():
        (PACKAGE / "STAGE8_EXECUTIVE_SUMMARY.md").write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
