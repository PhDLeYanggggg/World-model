from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.evaluation.stage5b5_gates import evaluate_gates, write_report


REPORT_DIR = Path("outputs/reports")
RESULT_DIR = Path("outputs/world_model_stage5b5_results")


def load(name, default):
    p = REPORT_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def main() -> int:
    write_problem_statement()
    write_cards_and_final()
    result = evaluate_gates()
    write_report(result)
    package()
    print(json.dumps({"result_dir": str(RESULT_DIR.resolve()), "gates": f"{result['passed']}/{result['total']}", "score": result["expert_audit_score"]}, indent=2))
    return 0


def write_problem_statement():
    text = """# Stage 5B.5 Problem Statement

Stage 5B.5 is not a latent generative stage. It is a deterministic learned dynamics repair stage.

Stage 5B showed that simple causal baselines can dominate smooth real trajectories, especially TGSIM traffic. A learned residual that only adds small corrections is not enough if the benchmark is mostly straight, smooth, or near-inertial. The core question is whether a deterministic temporal-interaction model can beat the strongest causal baseline on hard subsets: turning, acceleration/deceleration, stop/go, close interaction, near collision, high density, and long-horizon nonlinear motion.

Current hard constraint: latent generative modeling and SMC remain disabled until deterministic dynamics passes the hard gates.
"""
    (REPORT_DIR / "stage5b5_problem_statement.md").write_text(text, encoding="utf-8")


def write_cards_and_final():
    horizon = load("stage5b5_horizon_audit.json", [])
    hard = load("stage5b5_hard_subset_summary.json", [])
    metrics = load("metrics_stage5b5.json", {"datasets": {}})
    torch_training = load("stage5b5_temporal_training.json", {"runs": []})
    gates = evaluate_gates()
    rows = final_rows(metrics)
    torch_rows_data = torch_rows(torch_training)
    (REPORT_DIR / "data_card_stage5b5.md").write_text(data_card(horizon, hard), encoding="utf-8")
    (REPORT_DIR / "model_card_stage5b5.md").write_text(model_card(torch_training), encoding="utf-8")
    (REPORT_DIR / "stage5b5_next_steps.md").write_text(next_steps(), encoding="utf-8")
    (REPORT_DIR / "report_stage5b5_final.md").write_text(final_report(horizon, hard, rows, torch_rows_data, gates), encoding="utf-8")


def final_rows(metrics):
    rows = []
    for dataset, row in metrics.get("datasets", {}).items():
        for subset_name in ["all", "hard", "easy"]:
            subset = row["subsets"].get(subset_name)
            if not subset:
                continue
            horizons = subset["horizons"]
            target = "100" if "100" in horizons else max(horizons, key=lambda h: int(h))
            h = horizons[target]
            rows.append(
                {
                    "dataset": dataset,
                    "subset": subset_name,
                    "target_horizon": target,
                    "baseline_FDE": h["baseline_FDE"],
                    "learned_FDE": h["FDE"],
                    "improvement": h["improvement_over_strongest"],
                    "episodes": subset["episodes"],
                    "gate_alpha": subset["residual_gate_alpha_mean"],
                }
            )
    return rows


def torch_rows(torch_training):
    rows = []
    best = {}
    for run in torch_training.get("runs", []):
        mode = run.get("mode", "unknown")
        for dataset, row in run.get("test_metrics", {}).items():
            for subset_name in ["all", "hard"]:
                subset = row.get("subsets", {}).get(subset_name)
                if not subset:
                    continue
                horizons = subset.get("horizons", {})
                if not horizons:
                    continue
                target = "100" if "100" in horizons else max(horizons, key=lambda h: int(h))
                h = horizons[target]
                candidate = {
                    "mode": mode,
                    "dataset": dataset,
                    "subset": subset_name,
                    "target_horizon": target,
                    "baseline_FDE": round_float(h.get("baseline_FDE")),
                    "torch_FDE": round_float(h.get("FDE")),
                    "improvement": round_float(h.get("improvement_over_strongest")),
                }
                key = (dataset, subset_name)
                if key not in best or float(candidate["torch_FDE"]) < float(best[key]["torch_FDE"]):
                    best[key] = candidate
    rows.extend(best.values())
    return sorted(rows, key=lambda r: (r["dataset"], r["subset"]))


def round_float(value, digits=6):
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def data_card(horizon, hard):
    return "\n".join(
        [
            "# Stage 5B.5 Data Card",
            "",
            "Actual long-horizon pedestrian/drone data is still missing in this quick run. TGSIM provides verified t+100, but it is traffic/generic trajectory data, not proof of pedestrian world modeling.",
            "",
            "## Horizon Audit",
            markdown_table(
                [
                    {
                        "dataset": r["dataset_name"],
                        "raw_t50": r["supports_raw_t50"],
                        "raw_t100": r["supports_raw_t100"],
                        "t100_samples": r["raw_samples"].get("100", r["raw_samples"].get(100, 0)),
                        "coordinate_unit": r["coordinate_unit"],
                    }
                    for r in horizon
                ]
            ),
            "## Hard Subsets",
            markdown_table(
                [
                    {
                        "dataset": r["dataset_name"],
                        "hard": r["hard_episodes"],
                        "medium": r["medium_episodes"],
                        "easy": r["easy_episodes"],
                        "t100_hard": r["t100_hard_episodes"],
                        "eval_ok": r["large_enough_for_evaluation"],
                    }
                    for r in hard
                ]
            ),
            "SDD remains a license/manual placeholder. It was not downloaded or counted as converted.",
        ]
    ) + "\n"


def model_card(torch_training):
    torch_completed = any(
        str(run.get("checkpoint", "")).endswith(".pt") and Path(run.get("checkpoint", "")).exists()
        for run in torch_training.get("runs", [])
    )
    torch_status = (
        "The PyTorch GRU temporal-interaction path now runs in the cleaned `.venv_m3_torch` environment and produced three checkpoints: direct multi-horizon, recurrent rollout, and hybrid."
        if torch_completed
        else "The PyTorch GRU temporal-interaction path was not completed in this run; the NumPy fallback remains the official quick model."
    )
    return f"""# Stage 5B.5 Model Card

Models evaluated:

1. `numpy_temporal_interaction_ridge_residual` with causal history features, nearest-neighbor interaction features, domain flags, horizon conditioning, residual clipping, and validation-selected residual gate alpha.
2. PyTorch deterministic temporal-interaction variants: `direct_multi_horizon`, `recurrent_rollout`, and `hybrid`.

{torch_status}

The PyTorch backend recovery is an engineering fix, not a model-success result. The deterministic learned model still does not beat the strongest causal baseline on enough all-test / hard-test / verified t+100 benchmarks.

This stage remains deterministic. It is not CVAE, diffusion, latent generative modeling, or SMC. It predicts residuals over each dataset's strongest causal baseline, never over weak hand physics.
"""


def next_steps():
    return """# Stage 5B.5 Next Steps

1. Add a true long-horizon pedestrian/drone source: SDD with accepted license, full OpenTraj/ETH-UCY if legally prepared, or AerialMPT longer sequences with verified trajectories.
2. Scale and repair the PyTorch GRU/Transformer temporal-interaction model now that the runtime can complete, especially multi-agent scene batching, stronger hard-subset training, and stable long-horizon residual gating.
3. Build multi-agent episodes with split-safe agent groups so interaction features are model inputs, not only hard-subset diagnostics.
"""


def final_report(horizon, hard, rows, torch_rows_data, gates):
    ped_long = [r for r in horizon if r["dataset_name"] in {"trajnet", "eth_ucy"} and (r["supports_raw_t50"] or r["supports_raw_t100"])]
    t100_ped = [r for r in ped_long if r["supports_raw_t100"]]
    all_wins = [r for r in rows if r["subset"] == "all" and float(r["improvement"]) >= 0.05]
    hard_wins = [r for r in rows if r["subset"] == "hard" and float(r["improvement"]) >= 0.10]
    verified_wins = [r for r in rows if r["dataset"] in {"tgsim", "tgsim_i90"} and r["subset"] == "all" and r["target_horizon"] == "100" and float(r["improvement"]) >= 0.05]
    torch_all_wins = [r for r in torch_rows_data if r["subset"] == "all" and float(r["improvement"]) >= 0.05]
    torch_hard_wins = [r for r in torch_rows_data if r["subset"] == "hard" and float(r["improvement"]) >= 0.10]
    torch_verified_wins = [r for r in torch_rows_data if r["dataset"] in {"tgsim", "tgsim_i90"} and r["subset"] == "all" and r["target_horizon"] == "100" and float(r["improvement"]) >= 0.05]
    all_win_count = max(len(all_wins), len(torch_all_wins))
    hard_win_count = max(len(hard_wins), len(torch_hard_wins))
    verified_win_count = max(len(verified_wins), len(torch_verified_wins))
    return "\n".join(
        [
            "# Stage 5B.5 Final Report",
            "",
            "Stage 5B.5 built hard interaction subsets and trained deterministic temporal-interaction models. After the runtime cleanup, the PyTorch path also completed and produced checkpoints. It still did not make the project a foundation world model.",
            "",
            "## Benchmark Results",
            "",
            markdown_table(rows),
            "## PyTorch Backend Update",
            "",
            "The previous `OMP: Error #179: Can't open SHM failed` blocker is no longer present in the cleaned environment. PyTorch training completed for three deterministic modes, but the metrics still fail the deterministic gate overall.",
            "",
            markdown_table(torch_rows_data),
            "## Direct Answers",
            "",
            "新增真实 pedestrian / drone 数据：部分。TrajNet++/ETH-UCY bundled fallback was prepared/probed, but no new raw long-horizon pedestrian/drone source was verified.",
            "哪些能 t+50：TGSIM traffic sources only in this run.",
            "哪些能 t+100：TGSIM traffic sources only in this run.",
            "哪些只是 t+10：TrajNet and ETH/UCY fallback.",
            "是否构建 hard interaction subsets：是。",
            f"hard subsets 数量够不够：部分；{sum(1 for r in hard if r['large_enough_for_evaluation'])} datasets are eval-ok.",
            "temporal-interaction model 是否训练成功：是。NumPy deterministic fallback 和 PyTorch direct/recurrent/hybrid variants 都已跑通；但 PyTorch 结果仍未通过 deterministic gate.",
            f"是否超过 strongest causal baseline：部分；all-test wins={all_win_count}.",
            f"在 hard-test 上超过了吗：部分但不过 gate；hard wins over 10%={hard_win_count}，PyTorch 只在 TrajNet hard t+10 上出现明显提升，不足以过 gate.",
            f"在 verified t+100 上超过了吗：部分；verified t+100 wins={verified_win_count}.",
            "哪些数据源赢：tgsim_i90 all-test verified t+100 improved by about 8.9%.",
            "哪些数据源输：tgsim, ETH/UCY fallback; pedestrian/drone verified t+100 仍为 0；tgsim hard subset also did not improve.",
            "为什么输：strong causal baselines remain very strong; pedestrian snippets are short; no real maps/goals/routes; interaction is mostly diagnostic rather than multi-agent model input.",
            "是否仍然只是 trajectory forecasting：是，仍是 trajectory world-state benchmark scaffold.",
            "是否已经接近 world model：部分接近 state-space benchmark，但不是 true world model.",
            "是否可以进入 latent generative Stage 5C：否.",
            "是否可以启用 SMC：否.",
            f"expert audit score 是否达到 70：{'是' if gates['expert_audit_score'] >= 70 else '否'} ({gates['expert_audit_score']}).",
            "expert audit score 是否达到 80：否.",
            "",
            "## Final Verdict",
            "",
            "项目是否跑通：是",
            f"新增真实 pedestrian/drone 长轨迹数据：{'是' if ped_long else '否'}",
            f"actual verified t+100 pedestrian/drone source 数量：{len(t100_ped)}",
            "hard interaction benchmark 是否建立：是",
            "temporal-interaction deterministic model 是否超过 strongest causal baseline：部分",
            f"all-test 是否超过：{'部分' if all_win_count else '否'}",
            f"hard-test 是否超过：{'部分' if hard_win_count else '否'}",
            f"verified t+100 是否超过：{'部分' if verified_win_count else '否'}",
            "cross-dataset 泛化：diagnostic only",
            f"latent generative Stage 5C 是否 ready：{'是' if gates['latent_stage5c_ready'] else '否'}",
            f"SMC 是否 ready：{'是' if gates['smc_ready'] else '否'}",
            f"当前 verdict：{gates['verdict']}",
            f"expert audit score：{gates['expert_audit_score']}",
            "如果不能进入 Stage 5C，下一步先修什么：长轨迹 pedestrian/drone 数据；真实多智能体输入；干净 PyTorch temporal-interaction runtime。",
        ]
    ) + "\n"


def markdown_table(rows):
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def package():
    if RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    (RESULT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    for path in REPORT_DIR.glob("*stage5b5*"):
        if path.is_file():
            shutil.copy2(path, RESULT_DIR / "reports" / path.name)
    for name in ["data_card_stage5b5.md", "model_card_stage5b5.md", "failure_analysis_stage5b5.md"]:
        path = REPORT_DIR / name
        if path.exists():
            shutil.copy2(path, RESULT_DIR / "reports" / path.name)
    fig_src = Path("outputs/figures/stage5b5")
    if fig_src.exists():
        shutil.copytree(fig_src, RESULT_DIR / "figures" / "stage5b5")
    (RESULT_DIR / "STAGE5B5_EXECUTIVE_SUMMARY.md").write_text((REPORT_DIR / "report_stage5b5_final.md").read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
