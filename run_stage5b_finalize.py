from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.evaluation.stage5b_gates import evaluate_stage5b_gates, write_gate_report


REPORT_DIR = Path("outputs/reports")
RESULT_DIR = Path("outputs/world_model_stage5b_results")


def load_json(name: str, default):
    path = REPORT_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def md_table(rows):
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    summaries = [load_json(path.name, {}) for path in sorted(REPORT_DIR.glob("stage5b_episode_summary_*.json"))]
    summaries = [row for row in summaries if row and row.get("dataset_name")]
    baselines = load_json("stage5b_baseline_metrics.json", {"datasets": {}})
    learned = load_json("stage5b_deterministic_metrics.json", {})
    leakage = load_json("leakage_audit_stage5b.json", [])
    downloads = load_json("stage5b_download_records.json", [])
    actual_converted = [s for s in summaries if s.get("train_episodes", 0) + s.get("val_episodes", 0) + s.get("test_episodes", 0) > 0]
    t100_sources = [s for s in actual_converted if s.get("actual_verified_t100")]
    rows = dataset_rows(summaries, baselines, learned)
    write_data_card(summaries, downloads)
    write_model_card(baselines, learned)
    write_failure_analysis(rows)
    write_next_steps()
    gates = evaluate_stage5b_gates()
    write_gate_report(gates)
    write_final_report(rows, summaries, leakage, gates)
    package_results()
    print(json.dumps({"result_dir": str(RESULT_DIR.resolve()), "actual_converted": len(actual_converted), "actual_t100": len(t100_sources), "gates": f"{gates['passed']}/{gates['total']}"}, indent=2))
    return 0


def dataset_rows(summaries, baselines, learned):
    rows = []
    for summary in summaries:
        dataset = summary["dataset_name"]
        baseline_row = baselines.get("datasets", {}).get(dataset, {})
        strongest = baseline_row.get("strongest_causal_baseline", "n/a")
        target_h = str(baseline_row.get("target_horizon_for_strongest", 0))
        base_fde = baseline_row.get("all_baselines", {}).get(strongest, {}).get("horizons", {}).get(target_h, {}).get("FDE", "n/a")
        best_name, best_fde, best_improvement = best_learned_for_dataset(dataset, target_h, base_fde, learned)
        rows.append(
            {
                "dataset": dataset,
                "domain": summary.get("domain"),
                "actual_verified_t100": summary.get("actual_verified_t100"),
                "official_horizons": summary.get("official_eval_horizons"),
                "target_horizon": target_h,
                "strongest_causal_baseline": strongest,
                "baseline_FDE_target": base_fde,
                "best_learned": best_name,
                "learned_FDE_target": best_fde,
                "learned_improvement": best_improvement,
                "learned_beats_5pct": isinstance(best_improvement, float) and best_improvement >= 0.05,
            }
        )
    return rows


def best_learned_for_dataset(dataset, target_h, base_fde, learned):
    best = ("none", "n/a", "n/a")
    if not isinstance(base_fde, (float, int)):
        return best
    for variant in ["one_step", "multistep"]:
        metrics = learned.get(variant, {}).get("learned_metrics", {}).get(dataset, {})
        fde = metrics.get("horizons", {}).get(str(target_h), {}).get("FDE")
        if fde is None:
            continue
        improvement = (float(base_fde) - float(fde)) / max(abs(float(base_fde)), 1e-9)
        if best[1] == "n/a" or fde < best[1]:
            best = (f"deterministic_residual_{variant}", fde, round(improvement, 6))
    return best


def write_data_card(summaries, downloads):
    lines = [
        "# Stage 5B Data Card",
        "",
        "Actual converted datasets are separated from registry-only or gated datasets. Official benchmark inputs use causal finite-difference velocity.",
        "",
        "## Converted Datasets",
        "",
        md_table(
            [
                {
                    "dataset": s["dataset_name"],
                    "domain": s.get("domain"),
                    "coordinate_unit": s.get("coordinate_unit"),
                    "metric": s.get("is_metric"),
                    "samples_t100": s.get("samples_t100"),
                    "actual_verified_t100": s.get("actual_verified_t100"),
                    "train/val/test": f"{s.get('train_episodes')}/{s.get('val_episodes')}/{s.get('test_episodes')}",
                }
                for s in summaries
            ]
        ),
        "## Download / Access Records",
        "",
        md_table([{"dataset": d["dataset"], "status": d["status"], "kind": d["kind"], "executed": d["executed"], "notes": d["notes"]} for d in downloads]),
        "",
        "TrajNet++ was cloned from its public GitHub repository. The ETH/UCY fallback in this run is the BIWI/ETH-style file bundled in that TrajNet++ original-data tree, not a separate full official ETH/UCY conversion.",
    ]
    (REPORT_DIR / "data_card_stage5b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_card(baselines, learned):
    lines = [
        "# Stage 5B Deterministic Model Card",
        "",
        "Model: `deterministic_linear_residual_over_strongest_causal_baseline`.",
        "",
        "This is a gated deterministic pretraining test, not a latent generative model, not SMC, and not a large-scale foundation model. It learns a small linear residual over each dataset's strongest causal baseline.",
        "",
        "Official inputs: causal finite-difference position-derived velocity and past states only.",
        "",
        "Known limitation: the residual model is dataset-specific in this quick run; true leave-one-dataset-out transfer is diagnostic only.",
    ]
    (REPORT_DIR / "model_card_stage5b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_analysis(rows):
    failed = [row for row in rows if not row["learned_beats_5pct"]]
    lines = [
        "# Stage 5B Failure Analysis",
        "",
        "The deterministic learned residual did not pass the learned dynamics gate. It beat the strongest causal baseline by at least 5% only on the short-horizon ETH/UCY fallback subset, not on two actual verified t+100 sources.",
        "",
        "Main failure modes:",
        "",
        "1. Traffic trajectories are very smooth under causal constant velocity, so a small residual head easily over-corrects.",
        "2. The converted pedestrian sources in this quick run are short TrajNet-format snippets and cannot verify t+100.",
        "3. No real scene maps, lane graphs, goals, or interaction labels were available in the official converted quick benchmark.",
        "4. The residual model is too small and dataset-specific to be a foundation world model.",
        "",
        "Failed or insufficient datasets:",
        "",
        md_table(failed),
    ]
    (REPORT_DIR / "failure_analysis_stage5b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_next_steps():
    lines = [
        "# Stage 5B Next Steps",
        "",
        "1. Add a real pedestrian/drone source with verified t+100, preferably full SDD or a full ETH/UCY/OpenTraj conversion with longer tracks and scene homographies.",
        "2. Replace the quick linear residual with a real deterministic temporal-interaction model, but keep residual-over-strongest-baseline as the training target.",
        "3. Add real scene geometry or lane/map constraints for datasets that support it; do not report off-road or obstacle metrics where maps are absent.",
        "4. Run true leave-one-dataset-out training only after the deterministic model beats strongest causal baselines in-domain.",
        "5. Keep latent generative and SMC disabled until Stage 5B Gates 1-7 pass.",
    ]
    (REPORT_DIR / "stage5b_next_steps.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(rows, summaries, leakage, gates):
    actual_converted = [s for s in summaries if s.get("train_episodes", 0) + s.get("val_episodes", 0) + s.get("test_episodes", 0) > 0]
    actual_t100 = [s for s in actual_converted if s.get("actual_verified_t100")]
    leakage_pass = all(row.get("passed") for row in leakage) if leakage else False
    learned_wins = sum(1 for row in rows if row["learned_beats_5pct"])
    lines = [
        "# Stage 5B Final Report",
        "",
        "Stage 5B moved the project from a registry-heavy data lake scaffold to an actually executable multi-source benchmark. It still does not produce a large-scale foundation world model.",
        "",
        "## Honest Current State",
        "",
        "1. The project is not a true 3D world model.",
        "2. It is not a large-scale foundation world model.",
        "3. It is a partial but usable real-trajectory data lake plus deterministic residual pretraining gate.",
        "4. Latent generative modeling remains disabled.",
        "5. SMC remains disabled because deterministic learned proposals are not strong enough.",
        "",
        "## Actual Converted Sources",
        "",
        md_table(
            [
                {
                    "dataset": s["dataset_name"],
                    "domain": s.get("domain"),
                    "actual_verified_t100": s.get("actual_verified_t100"),
                    "horizons": s.get("official_eval_horizons"),
                    "episodes": f"{s.get('train_episodes')}/{s.get('val_episodes')}/{s.get('test_episodes')}",
                    "metric": s.get("is_metric"),
                }
                for s in summaries
            ]
        ),
        "## Baseline vs Learned",
        "",
        md_table(rows),
        "## Direct Answers",
        "",
        f"实际下载或接入真实数据源：{len(actual_converted)}",
        f"实际转换真实数据源：{len(actual_converted)}",
        f"actual verified t+100 数据源：{len(actual_t100)}",
        "registry-estimated 数据不计入 actual verified t+100。",
        "失败/placeholder：SDD、OpenDD、NGSIM 在本轮保持 license/manual placeholder，未作为 converted dataset。",
        "deterministic learned model 已训练：是，quick linear residual one-step 和 multistep 两个版本。",
        f"deterministic learned model 超过 strongest causal baseline 的数据源数量：{learned_wins}",
        "cross-dataset generalization：已执行诊断矩阵，但未完成真正 leave-one-dataset-out learned transfer。",
        f"no-leakage audit：{'pass' if leakage_pass else 'fail'}",
        "现在是否可以称为 large-scale world model：否。",
        "现在是否仍只是 trajectory forecasting model：是，更准确说是 multi-source trajectory world-state benchmark scaffold。",
        "是否可以进入 latent generative Stage 5C：否。",
        "是否可以启用 SMC：否。",
        f"当前 expert audit score：{gates['expert_audit_score']} / 100",
        f"当前 verdict：{gates['verdict']}",
        "",
        "## Final Verdict",
        "",
        "项目是否跑通：是",
        "数据湖是否从 partial 变成 usable：部分",
        f"实际转换真实数据源数量：{len(actual_converted)}",
        f"actual verified t+100 数据源数量：{len(actual_t100)}",
        f"no-leakage audit：{'pass' if leakage_pass else 'fail'}",
        "strongest causal baselines：见上表逐数据源结果",
        "best learned deterministic model：deterministic_residual_multistep on ETH/UCY fallback for t+10, but not enough for Stage 5C",
        f"learned model 是否超过 strongest causal baseline：{'部分' if learned_wins else '否'}",
        "跨数据集泛化：弱 / diagnostic only",
        "是否启用 latent generative：否",
        "是否启用 SMC：否",
        f"当前 verdict：{gates['verdict']}",
        f"expert audit score：{gates['expert_audit_score']}",
        f"是否达到 70：{'是' if gates['expert_audit_score'] >= 70 else '否'}",
        f"是否达到 80：{'是' if gates['expert_audit_score'] >= 80 else '否'}",
        "是否进入 Stage 5C latent generative：否",
        "如果否，下一步先修什么：补长轨迹真实行人/无人机数据；训练真正 deterministic temporal-interaction model；加入真实 scene/map/goal geometry。",
    ]
    (REPORT_DIR / "report_stage5b_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_results():
    if RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    (RESULT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    for path in REPORT_DIR.glob("*stage5b*"):
        if path.is_file():
            shutil.copy2(path, RESULT_DIR / "reports" / path.name)
    for name in ["data_card_stage5b.md", "model_card_stage5b.md", "failure_analysis_stage5b.md"]:
        path = REPORT_DIR / name
        if path.exists():
            shutil.copy2(path, RESULT_DIR / "reports" / path.name)
    fig_src = Path("outputs/figures/stage5b")
    if fig_src.exists():
        shutil.copytree(fig_src, RESULT_DIR / "figures" / "stage5b")
    summary = RESULT_DIR / "STAGE5B_EXECUTIVE_SUMMARY.md"
    summary.write_text((REPORT_DIR / "report_stage5b_final.md").read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
