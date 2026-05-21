#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.evaluation.stage7_benchmark import target_rows
from src.evaluation.stage7_gates import evaluate_gates, write_report


REPORT_DIR = Path("outputs/reports")
RESULT_DIR = Path("outputs/world_model_stage7_results")


def load(name, default):
    p = REPORT_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def markdown_table(rows):
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def write_all():
    gates = evaluate_gates()
    write_report(gates)
    scene_audit = load("stage7_scene_data_audit.json", [])
    scene_pack = load("stage7_scene_pack_report.json", {"scene_packs": []})
    goalbench = load("goalbench_summary_stage7.json", {"datasets": {}})
    goal_metrics = load("goal_predictor_metrics_stage7.json", {})
    failure_cmp = load("stage7_failure_predictor_comparison.json", {})
    interaction_aux = load("stage7_interaction_auxiliary_report.json", {})
    metrics = load("metrics_stage7.json", {"variants": []})
    rows = target_rows(metrics)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "data_card_stage7.md").write_text(data_card(scene_audit, scene_pack, goalbench), encoding="utf-8")
    (REPORT_DIR / "model_card_stage7.md").write_text(model_card(gates, goal_metrics, failure_cmp, interaction_aux), encoding="utf-8")
    (REPORT_DIR / "failure_analysis_stage7.md").write_text(failure_analysis(gates, rows, goal_metrics, failure_cmp, interaction_aux), encoding="utf-8")
    (REPORT_DIR / "stage7_next_steps.md").write_text(next_steps(), encoding="utf-8")
    (REPORT_DIR / "report_stage7_final.md").write_text(final_report(gates, scene_audit, scene_pack, goalbench, goal_metrics, failure_cmp, interaction_aux, rows), encoding="utf-8")
    package_outputs()
    return gates


def data_card(scene_audit, scene_pack, goalbench):
    audit_rows = [
        {
            "dataset": r["dataset_name"],
            "local": r["actual_downloaded_or_user_path_verified"],
            "unit": r["coordinate_unit"],
            "homography": r["homography_available"],
            "t50": r["max_verified_t50"],
            "t100": r["max_verified_t100"],
            "metric_eval": r["suitable_for_metric_evaluation"],
        }
        for r in scene_audit
    ]
    pack_rows = [
        {
            "dataset": p["dataset_name"],
            "scene": p["scene_id"],
            "goals": len(p["candidate_goal_regions"]),
            "walkable": p["annotation_quality"]["walkable_area"],
            "goal_source": p["annotation_quality"]["goals"],
        }
        for p in scene_pack.get("scene_packs", [])
    ]
    gb_rows = [{"dataset": k, **v} for k, v in goalbench.get("datasets", {}).items()]
    return "\n".join(
        [
            "# Stage 7 Data Card",
            "",
            "Stage 7 adds inferred scene packs and candidate goals. Inferred goals are not true observed goals.",
            "",
            "## Scene Data Audit",
            markdown_table(audit_rows),
            "## Scene Packs",
            markdown_table(pack_rows),
            "## GoalBench",
            markdown_table(gb_rows),
        ]
    ) + "\n"


def model_card(gates, goal_metrics, failure_cmp, interaction_aux):
    test_goal = goal_metrics.get("test", {})
    best_fp = max((v.get("test", {}).get("AUROC", 0.0) for v in failure_cmp.get("variants", {}).values()), default=0.0)
    return "\n".join(
        [
            "# Stage 7 Model Card",
            "",
            "Model family: deterministic scene/goal-grounded baseline-aware world-state predictor.",
            "",
            "Not enabled: latent generative modeling, CVAE/diffusion, SMC.",
            "",
            "Prediction form: `strongest_causal_baseline + alpha * goal_conditioned_bounded_residual`.",
            "",
            f"Goal predictor test top3: `{test_goal.get('top3_goal_accuracy')}` vs majority `{test_goal.get('majority_top3')}`.",
            f"Best Stage 7 failure predictor AUROC: `{best_fp}`.",
            f"Interaction auxiliary trajectory lift claimed: `{interaction_aux.get('metrics', {}).get('improves_hard_failure_trajectory_performance')}`.",
            f"Gate result: `{gates['passed']} / {gates['total']}`.",
            f"Verdict: `{gates['verdict']}`.",
        ]
    ) + "\n"


def failure_analysis(gates, rows, goal_metrics, failure_cmp, interaction_aux):
    failure_rows = [r for r in rows if r["subset"] == "baseline_failure"]
    hard_rows = [r for r in rows if r["subset"] == "hard"]
    return "\n".join(
        [
            "# Stage 7 Failure Analysis",
            "",
            "Remaining blockers:",
            "",
            "1. No verified pedestrian/drone t+50/t+100 source is available locally.",
            "2. Candidate goals are inferred from training endpoints, not true annotated destinations.",
            "3. If GoalBench top3 is saturated by majority baseline, goal prediction is not a strong signal.",
            "4. Interaction auxiliary labels remain weak because most converted episodes are one-primary-agent windows.",
            "5. Do not enter Stage 5C unless failure correction and hardbench gates pass.",
            "",
            "## BaselineFailureBench Rows",
            markdown_table(failure_rows[:40]),
            "## HardBench Rows",
            markdown_table(hard_rows[:40]),
            f"Current verdict: `{gates['verdict']}`.",
        ]
    ) + "\n"


def final_report(gates, scene_audit, scene_pack, goalbench, goal_metrics, failure_cmp, interaction_aux, rows):
    ped_long = any(r.get("eligible_for_pedestrian_drone_long_horizon_gate") for r in scene_audit)
    scene_grounded = len(scene_pack.get("scene_packs", [])) > 0
    goal_ok = gates["gates"][2]["passed"]
    fp_ok = gates["gates"][3]["passed"]
    failure_ok = gates["gates"][4]["passed"]
    hard_ok = gates["gates"][5]["passed"]
    easy_ok = gates["gates"][6]["passed"]
    interaction_ok = gates["gates"][7]["passed"]
    long_ok = gates["gates"][8]["passed"]
    return "\n".join(
        [
            "# Stage 7 Final Report",
            "",
            "Stage 7 upgraded the scaffold from pure trajectory residuals toward scene/goal-grounded deterministic prediction.",
            "",
            "## Honest Current State",
            "",
            "1. 当前不是 true 3D world model.",
            "2. 当前不是 large-scale foundation world model.",
            "3. 当前仍是 multi-source trajectory world-state benchmark scaffold.",
            "4. Stage 7 不启用 latent generative，不启用 SMC.",
            "5. traffic t+100 不能包装成 pedestrian world model.",
            "",
            "## What Changed",
            "",
            "- Built inferred scene packs with walkable bbox, boundary SDF, candidate goals, and route hypotheses.",
            "- Built GoalBench from scene-level candidate goals and future endpoint labels for training/evaluation only.",
            "- Trained a causal goal/intent predictor.",
            "- Trained goal/scene-conditioned baseline failure predictors.",
            "- Trained deterministic goal-conditioned gated residual variants.",
            "- Added interaction auxiliary diagnostics without claiming graph-interaction success.",
            "",
            "## Key Benchmark Rows",
            markdown_table(rows[:80]),
            "## Direct Answers",
            "",
            f"pedestrian/drone long horizon 是否补上：{'是' if ped_long else '否'}.",
            f"scene packs 是否建立：{'是' if scene_grounded else '否'}；数量={len(scene_pack.get('scene_packs', []))}.",
            f"candidate goals 是否建立：{'是' if scene_grounded else '否'}.",
            f"GoalBench 是否有意义：{'是' if any(v.get('goal_prediction_meaningful') for v in goalbench.get('datasets', {}).values()) else '部分/弱'} .",
            f"goal predictor 是否超过 majority baseline：{'是' if goal_ok else '否/部分'} .",
            f"goal/scene-conditioned failure predictor 是否超过 Stage 6：{'是' if fp_ok else '否'} .",
            f"goal-conditioned residual 是否在 BaselineFailureBench 上赢：{'是' if failure_ok else '否'} .",
            f"goal-conditioned residual 是否在 HardBench-v1 上赢：{'是' if hard_ok else '否'} .",
            f"easy subset 是否没有被破坏：{'是' if easy_ok else '否'} .",
            f"interaction auxiliary tasks 是否有效：{'是' if interaction_ok else '否/diagnostic only'} .",
            f"verified long-horizon 是否改善：{'是' if long_ok else '否'} .",
            "是否可以进入 latent generative Stage 5C：否." if not gates["latent_stage5c_ready"] else "是否可以进入 latent generative Stage 5C：是.",
            "是否可以启用 SMC：否.",
            "当前是否仍只是 trajectory forecasting scaffold：是，但现在加入了 scene/goal grounding.",
            "当前是否更接近 world model：部分，更接近 scene/goal-grounded state-space model，但不是 true 3D.",
            "",
            "## Final Verdict",
            "",
            "项目是否跑通：是",
            f"scene/goal grounding 是否建立：{'是' if scene_grounded else '否'}",
            f"pedestrian/drone long-horizon 是否补上：{'是' if ped_long else '否'}",
            f"GoalBench 是否可靠：{'是' if goal_ok else '部分'}",
            f"goal predictor 是否有效：{'是' if goal_ok else '部分/弱'}",
            f"goal-conditioned failure predictor 是否有效：{'是' if fp_ok else '部分/否'}",
            f"goal-conditioned residual 是否有效：{'是' if failure_ok and hard_ok else '部分/否'}",
            f"BaselineFailureBench 是否改善：{'是' if failure_ok else '否'}",
            f"HardBench 是否改善：{'是' if hard_ok else '否'}",
            f"verified long-horizon 是否改善：{'是' if long_ok else '否'}",
            f"latent generative Stage 5C 是否 ready：{'是' if gates['latent_stage5c_ready'] else '否'}",
            "SMC 是否 ready：否",
            f"当前 verdict：{gates['verdict']}",
            f"expert audit score：{gates['expert_audit_score']}",
            "如果不能进入 Stage 5C，下一步先修什么：真实 pedestrian/drone scene+homography+t50/t100；人工/半自动 walkable/exit/goal 标注；多智能体 episodes 而不是 single-primary-agent windows.",
        ]
    ) + "\n"


def next_steps():
    return """# Stage 7 Next Steps

1. Add a real pedestrian/drone dataset with verified t+50/t+100, scene image, and homography or calibrated scale.
2. Replace inferred bbox walkable areas and inferred endpoint goals with annotated walkable polygons, exits, obstacles, and route regions.
3. Rebuild multi-agent episodes so interaction auxiliary tasks use true neighboring trajectories rather than weak single-agent proxies.
"""


def package_outputs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "reports").mkdir(exist_ok=True)
    for pattern in ["*stage7*", "goalbench_summary_stage7.*", "metrics_stage7.*", "world_model_gate_stage7.*"]:
        for p in REPORT_DIR.glob(pattern):
            shutil.copy2(p, RESULT_DIR / "reports" / p.name)
    for name in ["data_card_stage7.md", "model_card_stage7.md", "failure_analysis_stage7.md", "report_stage7_final.md", "stage7_next_steps.md"]:
        p = REPORT_DIR / name
        if p.exists():
            shutil.copy2(p, RESULT_DIR / "reports" / p.name)
    scene_src = Path("data/scene_packs")
    if scene_src.exists():
        shutil.copytree(scene_src, RESULT_DIR / "scene_packs", dirs_exist_ok=True)
    goal_src = Path("data/goalbench")
    if goal_src.exists():
        shutil.copytree(goal_src, RESULT_DIR / "goalbench", dirs_exist_ok=True)
    ckpt = Path("outputs/checkpoints/stage7")
    if ckpt.exists():
        shutil.copytree(ckpt, RESULT_DIR / "checkpoints" / "stage7", dirs_exist_ok=True)
    summary = REPORT_DIR / "report_stage7_final.md"
    if summary.exists():
        (RESULT_DIR / "STAGE7_EXECUTIVE_SUMMARY.md").write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(write_all(), indent=2))

