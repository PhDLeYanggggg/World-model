from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.evaluation.stage5b6_gates import evaluate_gates, write_report


REPORT_DIR = Path("outputs/reports")
RESULT_DIR = Path("outputs/world_model_stage5b6_results")


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


def official_target_rows(metrics):
    official = {"gated_residual_all_data", "gated_residual_hard_weighted", "gated_residual_failure_classifier_aux"}
    rows = []
    for variant in metrics.get("variants", []):
        if variant.get("variant") not in official:
            continue
        for dataset, drow in variant.get("datasets", {}).items():
            for subset in ["all", "hard", "easy"]:
                srow = drow.get("subsets", {}).get(subset)
                if not srow:
                    continue
                horizons = srow.get("horizons", {})
                if not horizons:
                    continue
                target = "100" if "100" in horizons else max(horizons, key=lambda h: int(h))
                hrow = horizons[target]
                rows.append(
                    {
                        "model": variant["variant"],
                        "dataset": dataset,
                        "subset": subset,
                        "target": target,
                        "FDE": hrow["FDE"],
                        "baseline_FDE": hrow["baseline_FDE"],
                        "improvement": hrow["improvement_over_strongest"],
                        "episodes": srow["episodes"],
                        "alpha": srow["alpha_mean"],
                    }
                )
    return rows


def best_by_dataset(metrics):
    rows = official_target_rows(metrics)
    best = {}
    for row in rows:
        if row["subset"] != "all":
            continue
        key = row["dataset"]
        if key not in best or float(row["FDE"]) < float(best[key]["FDE"]):
            best[key] = row
    return list(best.values())


def write_cards_and_reports():
    gates = evaluate_gates()
    write_report(gates)
    reliability = load("stage5b6_hard_reliability_audit.json", [])
    horizon = load("stage5b6_pedestrian_drone_horizon_report.json", [])
    metrics = load("metrics_stage5b6.json", {"variants": []})
    oracle = load("stage5b6_baseline_failure_oracle.json", {"summary": {}})
    ablation = load("stage5b6_interaction_ablation.json", [])
    target_rows = official_target_rows(metrics)
    best_rows = best_by_dataset(metrics)
    (REPORT_DIR / "data_card_stage5b6.md").write_text(data_card(reliability, horizon), encoding="utf-8")
    (REPORT_DIR / "model_card_stage5b6.md").write_text(model_card(gates, oracle, ablation), encoding="utf-8")
    (REPORT_DIR / "failure_analysis_stage5b6.md").write_text(failure_analysis(gates, reliability, horizon, best_rows, ablation), encoding="utf-8")
    (REPORT_DIR / "stage5b6_next_steps.md").write_text(next_steps(), encoding="utf-8")
    (REPORT_DIR / "report_stage5b6_final.md").write_text(final_report(gates, reliability, horizon, target_rows, best_rows, oracle, ablation), encoding="utf-8")
    package()
    return gates


def data_card(reliability, horizon):
    return "\n".join(
        [
            "# Stage 5B.6 Data Card",
            "",
            "Stage 5B.6 uses the existing actual converted Stage 5B sources: TGSIM, TGSIM-I90, TrajNet fallback, and ETH/UCY fallback. It does not count registry-only data as converted data.",
            "",
            "## Hard Reliability",
            markdown_table(
                [
                    {
                        "dataset": r["dataset_name"],
                        "hard": r["hard_episode_count"],
                        "verified_t100_hard": r["hard_count_by_actual_verified_t100"],
                        "reliability": r["hard_reliability_label"],
                        "gate_eligible": r["hard_subset_is_gate_eligible"],
                    }
                    for r in reliability
                ]
            ),
            "## Pedestrian / Drone Horizon",
            markdown_table(
                [
                    {
                        "dataset": r["dataset_name"],
                        "unit": r["coordinate_unit"],
                        "max_raw_horizon": r["max_raw_horizon"],
                        "t50": r["t50_verified"],
                        "t100": r["t100_verified"],
                        "official_gate": r["suitable_for_official_gate"],
                    }
                    for r in horizon
                ]
            ),
            "No new verified pedestrian/drone t+50/t+100 source was added.",
        ]
    ) + "\n"


def model_card(gates, oracle, ablation):
    summary = oracle.get("summary", {})
    return "\n".join(
        [
            "# Stage 5B.6 Model Card",
            "",
            "Model type: baseline-aware deterministic gated residual over each dataset's strongest causal baseline.",
            "",
            "Prediction form: `prediction = strongest_causal_baseline + alpha * bounded_residual`.",
            "",
            "The model is not latent generative, not SMC, and not a true 3D world model. It remains a 2.5D / trajectory world-state benchmark model.",
            "",
            "Implemented variants:",
            "",
            "- `gated_residual_all_data`",
            "- `gated_residual_hard_weighted`",
            "- `gated_residual_failure_classifier_aux`",
            "- interaction ablations: no interaction, nearest-neighbor scalar, graph interaction, graph temporal history",
            "",
            f"Alpha calibration: corr={summary.get('alpha_vs_baseline_failure_correlation')}, easy_alpha={summary.get('easy_alpha_mean')}, hard_alpha={summary.get('hard_alpha_mean')}.",
            "",
            "Interaction result: graph interaction did not beat no-interaction in the quick hard benchmark. This is a failure, not a success.",
            "",
            f"Gate result: {gates['passed']} / {gates['total']}, verdict `{gates['verdict']}`.",
        ]
    ) + "\n"


def failure_analysis(gates, reliability, horizon, best_rows, ablation):
    return "\n".join(
        [
            "# Stage 5B.6 Failure Analysis",
            "",
            "Main failures:",
            "",
            "1. Hard benchmark reliability is insufficient: no dataset has >=50 hard episodes, so hard wins remain diagnostic.",
            "2. Real pedestrian/drone long horizon is still missing: no verified t+50/t+100 pedestrian/drone source was added.",
            "3. Official gated residual variants beat the strongest causal baseline on only one dataset target horizon.",
            "4. Graph interaction features did not improve hard-subset performance over no-interaction ablation.",
            "5. Verified t+100 improvement is not achieved by the official gated residual variants.",
            "",
            "Best official gated residual by dataset:",
            markdown_table(best_rows),
            "Interaction ablation:",
            markdown_table(ablation),
            f"Current verdict: `{gates['verdict']}`.",
        ]
    ) + "\n"


def next_steps():
    return """# Stage 5B.6 Next Steps

1. Add a legal real pedestrian/drone source with verified t+50/t+100, preferably SDD after license acceptance or full OpenTraj/ETH-UCY with longer raw tracks.
2. Build multi-agent episodes instead of single-primary-agent windows so the interaction encoder influences actual trajectory prediction, not only diagnostic features.
3. Increase hard subset size to at least 50 episodes per official hard gate and retrain the gated residual with reliable hard validation.
"""


def final_report(gates, reliability, horizon, target_rows, best_rows, oracle, ablation):
    hard_official = sum(1 for r in reliability if r["hard_subset_is_gate_eligible"])
    ped_t50 = sum(1 for r in horizon if r["t50_verified"])
    ped_t100 = sum(1 for r in horizon if r["t100_verified"])
    official_wins = sum(1 for r in best_rows if float(r["improvement"]) >= 0.05)
    hard_reliable = "是" if hard_official >= 2 else ("部分" if hard_official else "否")
    alpha_summary = oracle.get("summary", {})
    alpha_ok = alpha_summary.get("easy_alpha_mean") is not None and alpha_summary.get("hard_alpha_mean") is not None and alpha_summary["easy_alpha_mean"] < alpha_summary["hard_alpha_mean"]
    interaction_rows = {r["ablation"]: r for r in ablation}
    graph = interaction_rows.get("graph attention interaction", {}).get("mean_hard_target_improvement", 0.0)
    no = interaction_rows.get("no interaction", {}).get("mean_hard_target_improvement", 0.0)
    interaction_ok = graph > no
    return "\n".join(
        [
            "# Stage 5B.6 Final Report",
            "",
            "Stage 5B.6 repaired the benchmark reliability logic and trained baseline-aware gated residual models. It did not make the system ready for Stage 5C.",
            "",
            "## Current State",
            "",
            "1. 当前不是 true 3D world model.",
            "2. 当前不是 large-scale foundation world model.",
            "3. 当前仍是 multi-source trajectory world-state benchmark scaffold.",
            "4. Stage 5B.5 hard subsets 已建立，Stage 5B.6 进一步证明其统计可靠性不足.",
            "5. PyTorch deterministic temporal-interaction models 已经跑通，但 deterministic gate 仍失败.",
            "6. actual verified t+100 pedestrian/drone source 数量仍为 0.",
            "7. latent generative Stage 5C 和 SMC 仍不 ready.",
            "",
            "## Official Gated Residual Results",
            "",
            markdown_table(target_rows),
            "## Best Official Model By Dataset",
            "",
            markdown_table(best_rows),
            "## Interaction Ablation",
            "",
            markdown_table(ablation),
            "## Direct Answers",
            "",
            f"hard benchmark 是否可靠：{hard_reliable}；official hard-gate eligible datasets={hard_official}.",
            "hard subsets 样本量是否足够：否；所有 hard subsets 都低于 30 或 50 的强 gate 阈值.",
            "新增 pedestrian/drone 长时程数据了吗：否.",
            f"是否有 pedestrian/drone verified t+50：{'是' if ped_t50 else '否'}.",
            f"是否有 pedestrian/drone verified t+100：{'是' if ped_t100 else '否'}.",
            "gated residual model 是否训练成功：是，7 个 official/ablation checkpoint 已生成.",
            f"alpha gate 是否学会什么时候介入：{'部分' if alpha_ok else '否'}；easy_alpha={alpha_summary.get('easy_alpha_mean')}, hard_alpha={alpha_summary.get('hard_alpha_mean')}, corr={alpha_summary.get('alpha_vs_baseline_failure_correlation')}.",
            f"interaction encoder 是否真的带来提升：{'是' if interaction_ok else '否'}；graph hard improvement={graph}, no-interaction={no}.",
            f"all-test 是否超过 strongest causal baseline：部分；official gated residual target wins={official_wins}.",
            "hard-test 是否超过 strongest causal baseline：否，no official hard subset is reliable enough for gate.",
            "verified t+100 是否超过 strongest causal baseline：否，official gated variants did not beat verified t+100 by 5%.",
            "是否可以进入 Stage 5C latent generative：否.",
            "是否可以启用 SMC：否.",
            f"当前 expert audit score：{gates['expert_audit_score']}.",
            "",
            "## Final Verdict",
            "",
            "项目是否跑通：是",
            f"hard benchmark 是否可靠：{hard_reliable}",
            f"真实 pedestrian/drone long horizon 是否补上：{'是' if ped_t50 or ped_t100 else '否'}",
            f"gated residual 是否超过 strongest causal baseline：{'部分' if official_wins else '否'}",
            f"interaction encoder 是否有效：{'是' if interaction_ok else '否'}",
            f"alpha gate 是否有效：{'部分' if alpha_ok else '否'}",
            "verified long-horizon 是否改善：否",
            f"latent generative Stage 5C 是否 ready：{'是' if gates['latent_stage5c_ready'] else '否'}",
            f"SMC 是否 ready：{'是' if gates['smc_ready'] else '否'}",
            f"当前 verdict：{gates['verdict']}",
            f"expert audit score：{gates['expert_audit_score']}",
            "如果不能进入 Stage 5C，下一步先修什么：真实 pedestrian/drone t+50/t+100；>=50 hard episodes per dataset；真正多智能体 interaction episode 输入。",
        ]
    ) + "\n"


def package():
    if RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    (RESULT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    for path in REPORT_DIR.glob("*stage5b6*"):
        if path.is_file():
            shutil.copy2(path, RESULT_DIR / "reports" / path.name)
    for name in ["data_card_stage5b6.md", "model_card_stage5b6.md", "failure_analysis_stage5b6.md"]:
        path = REPORT_DIR / name
        if path.exists():
            shutil.copy2(path, RESULT_DIR / "reports" / path.name)
    summary = REPORT_DIR / "report_stage5b6_final.md"
    if summary.exists():
        (RESULT_DIR / "STAGE5B6_EXECUTIVE_SUMMARY.md").write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")
    ckpt_src = Path("outputs/checkpoints/stage5b6")
    if ckpt_src.exists():
        shutil.copytree(ckpt_src, RESULT_DIR / "checkpoints" / "stage5b6", dirs_exist_ok=True)


def main() -> int:
    gates = write_cards_and_reports()
    print(json.dumps({"result_dir": str(RESULT_DIR.resolve()), "gates": f"{gates['passed']}/{gates['total']}", "score": gates["expert_audit_score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
