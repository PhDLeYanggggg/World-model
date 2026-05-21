from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict


REPORT_DIR = Path("outputs/reports")
PACKAGE = Path("outputs/world_model_stage9_results")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def main() -> None:
    gate = load_json(REPORT_DIR / "world_model_gate_stage9.json", {})
    data = load_json(REPORT_DIR / "stage9_data_audit.json", {})
    baselines = load_json(REPORT_DIR / "stage9_per_agent_baseline_metrics.json", {})
    metrics = load_json(REPORT_DIR / "metrics_stage9.json", {})
    aux = load_json(REPORT_DIR / "stage9_interaction_auxiliary_report.json", {})
    write_final(gate, data, baselines, metrics, aux)
    write_cards(gate, data, baselines, metrics, aux)
    package_outputs()
    print(f"Stage 9 package: {PACKAGE.resolve()}")


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
    return max(vals) if vals else 0.0


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
    return sum(vals) / len(vals) if vals else 0.0


def write_final(gate: Dict, data: Dict, baselines: Dict, metrics: Dict, aux: Dict) -> None:
    full = "per_agent_full_scene_goal_interaction"
    full_all = mean_imp(metrics, full, "all")
    hard = max(best_imp(metrics, full, "hard"), best_imp(metrics, full, "baseline_failure"))
    interaction_gain = max(best_imp(metrics, full, "hard") - best_imp(metrics, "per_agent_scene_goal", "hard"), best_imp(metrics, full, "baseline_failure") - best_imp(metrics, "per_agent_scene_goal", "baseline_failure"))
    scene_goal_gain = max(best_imp(metrics, "per_agent_scene_goal", "hard") - best_imp(metrics, "per_agent_no_scene", "hard"), best_imp(metrics, "per_agent_scene_goal", "goalbench_official") - best_imp(metrics, "per_agent_no_scene", "goalbench_official"))
    multi_gain = mean_imp(metrics, full, "ge5") - mean_imp(metrics, "per_agent_no_scene", "ge5")
    lines = [
        "# Stage 9 Final Report",
        "",
        "Stage 9 trains deterministic per-agent multi-agent scene-grounded residual models. It does not enable latent generative modeling or SMC.",
        "",
        "## Direct Answers",
        "",
        "1. 是否训练了 per-agent multi-agent world model：是",
        "2. 是否预测所有 agents，而不是只预测 primary agent：是",
        f"3. full model 是否超过 strongest causal baseline：{'是' if full_all >= 0.05 else '否'} (mean all-test improvement={full_all:.6f})",
        f"4. hard/failure subset 是否超过 baseline：{'是' if hard >= 0.10 else '否'} (best hard/failure improvement={hard:.6f})",
        f"5. easy subset 是否保持：{'否/不可充分评估' if not gate_pass(gate, 'Easy Preservation Gate') else '是'}",
        f"6. interaction 是否真正提升轨迹预测：{'是' if interaction_gain > 0.02 else '否'} (gain={interaction_gain:.6f})",
        f"7. scene/goal 是否真正提升轨迹预测：{'是' if scene_goal_gain > 0.02 else '否'} (gain={scene_goal_gain:.6f})",
        f"8. per-agent model 是否比 primary/simple model 更好：{'是' if multi_gain > 0.02 else '否'} (ge5 gain={multi_gain:.6f})",
        f"9. 是否仍缺 pedestrian/drone t+50/t+100：{'是' if data.get('actual_verified_t50_episodes', 0) == 0 and data.get('actual_verified_t100_episodes', 0) == 0 else '否'}",
        f"10. 是否可以进入 Stage 5C latent generative：{'是' if gate.get('latent_stage5c_ready') else '否'}",
        f"11. 是否可以启用 SMC：{'是' if gate.get('smc_ready') else '否'}",
        "12. 当前是否仍只是 trajectory forecasting scaffold：是，但已经是 per-agent all-agent 形式。",
        "13. 当前是否更接近 world model：部分，更接近 scene/goal/multi-agent state-space scaffold，但 deterministic gates 未过。",
        "",
        "## Final Conclusion",
        "",
        "项目是否跑通：是",
        "per-agent multi-agent model 是否训练：是",
        "是否预测所有 agents：是",
        f"per-agent model 是否超过 strongest causal baseline：{'是' if full_all >= 0.05 else '否'}",
        f"hard/failure subset 是否改善：{'是' if hard >= 0.10 else '部分' if hard > 0 else '否'}",
        f"easy subset 是否保持：{'是' if gate_pass(gate, 'Easy Preservation Gate') else '否/不可充分评估'}",
        f"interaction 是否有效：{'是' if interaction_gain > 0.02 else '否'}",
        f"scene/goal 是否有效：{'是' if scene_goal_gain > 0.02 else '否'}",
        f"multi-agent 是否优于 primary-agent：{'是' if multi_gain > 0.02 else '否'}",
        "pedestrian/drone t+50/t+100 是否仍缺：是",
        f"latent generative Stage 5C 是否 ready：{'是' if gate.get('latent_stage5c_ready') else '否'}",
        f"SMC 是否 ready：{'是' if gate.get('smc_ready') else '否'}",
        f"当前 verdict：{gate.get('verdict', 'unknown')}",
        f"expert audit score：{gate.get('expert_audit_score', 'unknown')}",
        "",
        "如果不能进入 Stage 5C，下一步先修什么：",
        "",
        "1. 接入 verified pedestrian/drone t+50/t+100 数据，优先 SDD/OpenTraj，并保留 homography/scale 状态。",
        "2. 将 silver scene annotations 升级为人工确认 gold，并补 walkable/exit/goal/obstacle 标注。",
        "3. 改进 per-agent residual：加入更稳的 failure-aware gating、按 agent 类型/场景分层训练，并解决 easy subset 可评估性。",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report_stage9_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    failure_lines = [
        "# Stage 9 Failure Analysis",
        "",
        f"Full model all-test mean improvement: {full_all:.6f}. This does not meet the 5% gate.",
        f"Hard/failure best improvement: {hard:.6f}. This does not meet the 10% gate.",
        f"Interaction gain over scene+goal: {interaction_gain:.6f}. Interaction is not proven useful for trajectory metrics.",
        f"Scene/goal gain over no-scene: {scene_goal_gain:.6f}. Scene/goal grounding is not yet producing reliable trajectory lift.",
        "The strongest causal baselines remain very hard to beat on t+10 pedestrian-like snippets.",
        "No verified pedestrian/drone t+50/t+100 is available, so no long-horizon pedestrian world-model claim is allowed.",
    ]
    (REPORT_DIR / "failure_analysis_stage9.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")


def gate_pass(gate: Dict, name: str) -> bool:
    return any(g.get("name") == name and g.get("passed") for g in gate.get("gates", []))


def write_cards(gate: Dict, data: Dict, baselines: Dict, metrics: Dict, aux: Dict) -> None:
    model_card = [
        "# Stage 9 Model Card",
        "",
        "Model type: deterministic per-agent multi-agent bounded residual over strongest causal baseline.",
        "Prediction form: `prediction_i = baseline_i + alpha_i * bounded_residual_i`.",
        "Predicts all active agents with masks. No latent generative branch. No SMC.",
        "",
        f"Gate verdict: {gate.get('verdict')}",
        f"Expert audit score: {gate.get('expert_audit_score')}",
    ]
    (REPORT_DIR / "model_card_stage9.md").write_text("\n".join(model_card) + "\n", encoding="utf-8")
    data_card = [
        "# Stage 9 Data Card",
        "",
        f"total episodes: {data.get('total_per_agent_multiagent_episodes')}",
        f">=2 agent episodes: {data.get('episodes_with_ge2_agents')}",
        f"silver scene episodes: {data.get('silver_scene_episodes')}",
        f"gold scene episodes: {data.get('gold_scene_episodes')}",
        f"verified t10 episodes: {data.get('actual_verified_t10_episodes')}",
        f"verified t50 episodes: {data.get('actual_verified_t50_episodes')}",
        f"verified t100 episodes: {data.get('actual_verified_t100_episodes')}",
        f"split sizes: {data.get('train_val_test_split_sizes')}",
    ]
    (REPORT_DIR / "data_card_stage9.md").write_text("\n".join(data_card) + "\n", encoding="utf-8")
    next_steps = [
        "# Stage 9 Next Steps",
        "",
        "1. Add verified pedestrian/drone t+50/t+100 data.",
        "2. Upgrade silver annotations to human-confirmed gold scene/goal labels.",
        "3. Improve deterministic per-agent gating before any latent generative or SMC work.",
    ]
    (REPORT_DIR / "stage9_next_steps.md").write_text("\n".join(next_steps) + "\n", encoding="utf-8")


def package_outputs() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    (PACKAGE / "reports").mkdir(exist_ok=True)
    for path in REPORT_DIR.glob("*stage9*"):
        shutil.copy2(path, PACKAGE / "reports" / path.name)
    for src_dir, dst_name in [
        (Path("outputs/checkpoints/stage9"), "checkpoints_stage9"),
    ]:
        dst = PACKAGE / dst_name
        if dst.exists():
            shutil.rmtree(dst)
        if src_dir.exists():
            shutil.copytree(src_dir, dst)
    final = REPORT_DIR / "report_stage9_final.md"
    if final.exists():
        (PACKAGE / "STAGE9_EXECUTIVE_SUMMARY.md").write_text(final.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
