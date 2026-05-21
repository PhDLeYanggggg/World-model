from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.evaluation.stage6_gates import evaluate_gates, write_report


REPORT_DIR = Path("outputs/reports")
RESULT_DIR = Path("outputs/world_model_stage6_results")


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


def target_rows(metrics):
    rows = []
    for variant in metrics.get("variants", []):
        for dataset, drow in variant.get("datasets", {}).items():
            for subset in ["all", "easy", "hard", "baseline_failure", "verified_t50", "verified_t100"]:
                srow = drow.get("subsets", {}).get(subset)
                if not srow:
                    continue
                horizons = srow.get("horizons", {})
                if not horizons:
                    continue
                target = "100" if "100" in horizons else max(horizons, key=lambda h: int(h))
                h = horizons[target]
                rows.append(
                    {
                        "model": variant["variant"],
                        "dataset": dataset,
                        "subset": subset,
                        "target": target,
                        "FDE": h["FDE"],
                        "baseline_FDE": h["baseline_FDE"],
                        "improvement": h["improvement_over_strongest"],
                        "episodes": srow["episodes"],
                        "alpha": srow["alpha_mean"],
                        "intervention": srow["intervention_rate"],
                    }
                )
    return rows


def best_rows(metrics, subset):
    best = {}
    for row in target_rows(metrics):
        if row["subset"] != subset:
            continue
        key = row["dataset"]
        if key not in best or float(row["FDE"]) < float(best[key]["FDE"]):
            best[key] = row
    return list(best.values())


def write_all():
    gates = evaluate_gates()
    write_report(gates)
    ped = load("stage6_pedestrian_drone_audit.json", [])
    hardbench = load("hardbench_v1_summary.json", {})
    failure_bench = load("baseline_failure_bench_summary.json", {})
    predictor = load("baseline_failure_predictor_metrics.json", {})
    metrics = load("metrics_stage6.json", {"variants": []})
    interaction = load("stage6_interaction_ablation.json", [])
    rows_all = target_rows(metrics)
    (REPORT_DIR / "data_card_stage6.md").write_text(data_card(ped, hardbench, failure_bench), encoding="utf-8")
    (REPORT_DIR / "model_card_stage6.md").write_text(model_card(gates, predictor, interaction), encoding="utf-8")
    (REPORT_DIR / "failure_analysis_stage6.md").write_text(failure_analysis(gates, best_rows(metrics, "baseline_failure"), best_rows(metrics, "verified_t100"), interaction), encoding="utf-8")
    (REPORT_DIR / "stage6_next_steps.md").write_text(next_steps(), encoding="utf-8")
    (REPORT_DIR / "report_stage6_final.md").write_text(final_report(gates, ped, hardbench, failure_bench, predictor, rows_all, interaction), encoding="utf-8")
    package()
    return gates


def data_card(ped, hardbench, failure_bench):
    ped_rows = [
        {
            "dataset": r["dataset_name"],
            "downloaded_or_path": r["actual_downloaded_or_user_path_verified"],
            "unit": r["coordinate_unit"],
            "t50": r["max_verified_t50"],
            "t100": r["max_verified_t100"],
            "gate": r["eligible_for_pedestrian_drone_long_horizon_gate"],
        }
        for r in ped
    ]
    failure_rows = [
        {"dataset": k, **v}
        for k, v in failure_bench.get("baseline_failure_rate_by_dataset", {}).items()
    ]
    return "\n".join(
        [
            "# Stage 6 Data Card",
            "",
            "Stage 6 counts only actual local converted/user-path verified sources. Registry-only datasets are not benchmark data.",
            "",
            "## Pedestrian/Drone Audit",
            markdown_table(ped_rows),
            "## HardBench-v1",
            markdown_table(
                [
                    {"field": "total_hard_episodes", "value": hardbench.get("total_hard_episodes")},
                    {"field": "gate_eligibility", "value": hardbench.get("gate_eligibility")},
                    {"field": "pedestrian_drone_hard_episodes", "value": hardbench.get("pedestrian_drone_hard_episodes")},
                    {"field": "traffic_hard_episodes", "value": hardbench.get("traffic_hard_episodes")},
                ]
            ),
            "## BaselineFailureBench",
            markdown_table(failure_rows),
        ]
    ) + "\n"


def model_card(gates, predictor, interaction):
    test = predictor.get("test", {})
    return "\n".join(
        [
            "# Stage 6 Model Card",
            "",
            "Model family: deterministic baseline-failure-aware trajectory world-state model.",
            "",
            "Not enabled: latent generative modeling, diffusion, CVAE, SMC.",
            "",
            "Components:",
            "",
            "- Baseline failure predictor from causal past-window features.",
            "- Failure-aware gated residual model: `baseline + alpha * bounded_residual`.",
            "- Interaction ablations: no interaction, scalar interaction, graph interaction.",
            "",
            f"Failure predictor test AUROC: `{test.get('AUROC')}`.",
            f"Failure predictor test AUPRC: `{test.get('AUPRC')}`.",
            "",
            "Graph interaction did not pass the interaction gate. Scalar/graph features are kept for diagnostics, not claimed as a solved interaction model.",
            "",
            f"Gate result: `{gates['passed']} / {gates['total']}`.",
            f"Verdict: `{gates['verdict']}`.",
        ]
    ) + "\n"


def failure_analysis(gates, failure_rows, t100_rows, interaction):
    return "\n".join(
        [
            "# Stage 6 Failure Analysis",
            "",
            "Failures that still block Stage 5C:",
            "",
            "1. No real pedestrian/drone verified t+50/t+100 source is available.",
            "2. Failure-aware gated residual does not improve BaselineFailureBench by the required 10%.",
            "3. Verified long-horizon improvement still fails the >=5% gate.",
            "4. Interaction features do not produce a reliable gain over no-interaction.",
            "5. Traffic long-horizon results cannot be presented as pedestrian world-model success.",
            "",
            "## BaselineFailureBench Best Rows",
            markdown_table(failure_rows),
            "## Verified t+100 Best Rows",
            markdown_table(t100_rows),
            "## Interaction Ablation",
            markdown_table(interaction),
            f"Current verdict: `{gates['verdict']}`.",
        ]
    ) + "\n"


def next_steps():
    return """# Stage 6 Next Steps

1. Add an actual legal pedestrian/drone long-horizon source, preferably SDD or full OpenTraj/ETH-UCY with verified t+50/t+100.
2. Convert multi-agent episodes instead of one-primary-agent windows so interaction encoders model real neighboring trajectories.
3. Train the failure-aware residual only on reliable BaselineFailureBench folds and require >=10% failure-subset improvement before any Stage 5C latent generative work.
"""


def final_report(gates, ped, hardbench, failure_bench, predictor, rows_all, interaction):
    ped_long = sum(1 for r in ped if r.get("eligible_for_pedestrian_drone_long_horizon_gate"))
    predictor_ok = gates["gates"][3]["passed"]
    alpha_ok = gates["gates"][4]["passed"]
    failure_ok = gates["gates"][5]["passed"]
    easy_ok = gates["gates"][6]["passed"]
    interaction_ok = gates["gates"][8]["passed"]
    long_ok = gates["gates"][7]["passed"]
    hard_ok = gates["gates"][1]["passed"]
    failure_bench_ok = gates["gates"][2]["passed"]
    return "\n".join(
        [
            "# Stage 6 Final Report",
            "",
            "Stage 6 built HardBench-v1, BaselineFailureBench, a causal baseline-failure predictor, and a deterministic failure-aware gated residual model. It still does not unlock Stage 5C.",
            "",
            "## Honest Current State",
            "",
            "1. 当前不是 true 3D world model.",
            "2. 当前不是 large-scale foundation world model.",
            "3. 当前仍是 multi-source trajectory world-state benchmark scaffold.",
            "4. 不启用 latent generative，不启用 SMC.",
            "5. traffic t+100 不能包装成 pedestrian world model.",
            "",
            "## Key Metrics",
            "",
            markdown_table(rows_all),
            "## Interaction Ablation",
            "",
            markdown_table(interaction),
            "## Direct Answers",
            "",
            f"是否补上 pedestrian/drone verified t+50/t+100：{'是' if ped_long else '否'}.",
            f"HardBench-v1 是否可靠：{'是' if hard_ok else '否'}；hard episodes={hardbench.get('total_hard_episodes')}, eligibility={hardbench.get('gate_eligibility')}.",
            f"BaselineFailureBench 是否建立：{'是' if failure_bench_ok else '否'}；failure samples={failure_bench.get('failure_samples')}.",
            f"failure predictor 是否有效：{'是' if predictor_ok else '否'}；test AUROC={predictor.get('test', {}).get('AUROC')}.",
            f"alpha 是否学会何时介入：{'是' if alpha_ok else '部分/否'}.",
            f"failure-aware model 是否在 baseline failure cases 上赢：{'是' if failure_ok else '否'}.",
            f"easy cases 是否没有被破坏：{'是' if easy_ok else '否'}.",
            f"interaction encoder 是否有效：{'是' if interaction_ok else '否'}.",
            f"verified long-horizon 是否改善：{'是' if long_ok else '否'}.",
            f"是否可以进入 latent generative Stage 5C：{'是' if gates['latent_stage5c_ready'] else '否'}.",
            f"是否可以启用 SMC：{'是' if gates['smc_ready'] else '否'}.",
            "当前是否仍只是 trajectory forecasting scaffold：是.",
            "当前是否更接近 world model：部分，更接近 failure-aware benchmarked world-state model，但不是 true world model.",
            "",
            "## Final Verdict",
            "",
            "项目是否跑通：是",
            f"pedestrian/drone long-horizon 是否补上：{'是' if ped_long else '否'}",
            f"HardBench-v1 是否可靠：{'是' if hard_ok else '否'}",
            f"BaselineFailureBench 是否可靠：{'是' if failure_bench_ok else '否'}",
            f"failure predictor 是否有效：{'是' if predictor_ok else '否'}",
            f"failure-aware gated residual 是否有效：{'是' if failure_ok else '部分' if alpha_ok else '否'}",
            f"interaction encoder 是否有效：{'是' if interaction_ok else '否'}",
            f"verified long-horizon 是否改善：{'是' if long_ok else '否'}",
            f"latent generative Stage 5C 是否 ready：{'是' if gates['latent_stage5c_ready'] else '否'}",
            f"SMC 是否 ready：{'是' if gates['smc_ready'] else '否'}",
            f"当前 verdict：{gates['verdict']}",
            f"expert audit score：{gates['expert_audit_score']}",
            "如果不能进入 Stage 5C，下一步先修什么：真实 pedestrian/drone verified t+50/t+100；真正多智能体 interaction episodes；failure-aware model 在 BaselineFailureBench 上稳定超过 baseline 10%。",
        ]
    ) + "\n"


def package():
    if RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    (RESULT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    for pattern in ["*stage6*", "hardbench_v1_summary.*", "baseline_failure_bench_summary.*", "baseline_failure_predictor_*"]:
        for path in REPORT_DIR.glob(pattern):
            if path.is_file():
                shutil.copy2(path, RESULT_DIR / "reports" / path.name)
    for name in ["data_card_stage6.md", "model_card_stage6.md", "failure_analysis_stage6.md", "world_model_gate_stage6.md", "world_model_gate_stage6.json"]:
        path = REPORT_DIR / name
        if path.exists():
            shutil.copy2(path, RESULT_DIR / "reports" / path.name)
    ckpt = Path("outputs/checkpoints/stage6")
    if ckpt.exists():
        shutil.copytree(ckpt, RESULT_DIR / "checkpoints" / "stage6", dirs_exist_ok=True)
    summary = REPORT_DIR / "report_stage6_final.md"
    if summary.exists():
        (RESULT_DIR / "STAGE6_EXECUTIVE_SUMMARY.md").write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    gates = evaluate_gates()
    write_report(gates)
    write_reports = write_report
    del write_reports
    ped = load("stage6_pedestrian_drone_audit.json", [])
    hardbench = load("hardbench_v1_summary.json", {})
    failure_bench = load("baseline_failure_bench_summary.json", {})
    predictor = load("baseline_failure_predictor_metrics.json", {})
    metrics = load("metrics_stage6.json", {"variants": []})
    interaction = load("stage6_interaction_ablation.json", [])
    (REPORT_DIR / "data_card_stage6.md").write_text(data_card(ped, hardbench, failure_bench), encoding="utf-8")
    (REPORT_DIR / "model_card_stage6.md").write_text(model_card(gates, predictor, interaction), encoding="utf-8")
    (REPORT_DIR / "failure_analysis_stage6.md").write_text(failure_analysis(gates, best_rows(metrics, "baseline_failure"), best_rows(metrics, "verified_t100"), interaction), encoding="utf-8")
    (REPORT_DIR / "stage6_next_steps.md").write_text(next_steps(), encoding="utf-8")
    (REPORT_DIR / "report_stage6_final.md").write_text(final_report(gates, ped, hardbench, failure_bench, predictor, target_rows(metrics), interaction), encoding="utf-8")
    package()
    print(json.dumps({"result_dir": str(RESULT_DIR.resolve()), "gates": f"{gates['passed']}/{gates['total']}", "score": gates["expert_audit_score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

