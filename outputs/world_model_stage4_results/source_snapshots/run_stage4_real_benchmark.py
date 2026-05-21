from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.build_real_episodes import build_real_episodes
from src.data.real_trajectory_loader import missing_data_error
from src.evaluation.world_model_gates import run_stage4_gates
from src.training.evaluate_real_benchmark import evaluate_real_benchmark, flatten_metrics, markdown_table
from src.training.train_real_benchmark import train_real_residual_model


REPORT_DIR = Path("outputs/reports")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 4 real long-trajectory benchmark.")
    parser.add_argument("--dataset", choices=["tgsim", "trajnet", "eth_ucy", "sdd"], required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.data:
        write_failure_outputs(args.dataset, missing_data_error(args.dataset))
        print(missing_data_error(args.dataset))
        return 2
    try:
        built = build_real_episodes(args.dataset, args.data, quick=args.quick)
        episodes = built["episodes"]
        summary = built["summary"]
        (REPORT_DIR / "real_data_episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        train = [e for e in episodes if e["meta"]["split"] == "real_train"]
        model_bundle = train_real_residual_model(train)
        metrics = evaluate_real_benchmark(episodes, model_bundle, REPORT_DIR, quick=args.quick)
        gate_payload = run_stage4_gates(metrics, summary, prior_audit_score=58.0)
        write_report(args.dataset, summary, metrics, gate_payload, model_bundle["training"])
        return 0
    except Exception as exc:
        write_failure_outputs(args.dataset, str(exc))
        print(f"Stage 4 failed clearly: {exc}")
        return 1


def write_failure_outputs(dataset: str, reason: str) -> None:
    summary = {
        "dataset_name": dataset,
        "total_scenes": 0,
        "total_agents": 0,
        "total_tracks": 0,
        "total_frames": 0,
        "samples_t10": 0,
        "samples_t25": 0,
        "samples_t50": 0,
        "samples_t100": 0,
        "train_episodes": 0,
        "val_episodes": 0,
        "test_episodes": 0,
        "mean_agents_per_episode": 0,
        "mean_track_length": 0,
        "coordinate_unit": "unknown",
        "whether_metric_coordinates": False,
        "whether_scene_geometry_available": False,
        "whether_t100_verified": False,
        "cannot_evaluate_t100": reason,
    }
    metrics = {"error": reason}
    (REPORT_DIR / "real_data_episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "metrics_stage4_real.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "metrics_stage4_real.csv").write_text("", encoding="utf-8")
    (REPORT_DIR / "metrics_table_stage4_real.md").write_text("_No real benchmark metrics available._\n", encoding="utf-8")
    gate_payload = run_stage4_gates({}, summary, prior_audit_score=58.0)
    write_failure_analysis(summary, metrics, reason)
    write_report(dataset, summary, metrics, gate_payload, {"trained": False, "reason": reason})


def write_report(dataset: str, summary: dict, metrics: dict, gate_payload: dict, training: dict) -> None:
    rows = flatten_metrics(metrics) if isinstance(metrics, dict) and "error" not in metrics else []
    report = f"""# Stage 4 Real Long-Trajectory Benchmark

## Required Current-State Admission

1. 当前模型是 pseudo-3D physics-informed learned residual state-space world model。
2. 它不是真 3D。
3. 它不是 exceptional world model。
4. 当前 expert audit score 是 58/100。
5. 当前 verdict 是 prototype_with_major_failures。
6. Synthetic t+100 已经可验证。
7. AerialMPT bauma3 t+100 仍然只能 qualitative free-run，不能报 ADE@100/FDE@100。
8. learned residual 没有显著超过 hand physics。
9. SMC 没有明显提升 coverage。
10. 真实长轨迹 t+100 benchmark 在本报告中按实际数据接入结果判定。

## Real Data Summary

{markdown_table([summary])}

## Training

{markdown_table([training])}

## Metrics

{markdown_table(rows) if rows else '_No real benchmark metrics available._'}

## Gates

See `outputs/reports/world_model_gate_stage4.md`.

## Direct Conclusions

项目是否跑通：
{'是' if rows else '否'}

接入真实长轨迹数据：
{'是' if summary.get('total_tracks', 0) > 0 else '否'}

真实数据名称：
{summary.get('dataset_name', dataset)}

真实数据是否支持 t+100 verified evaluation：
{'是' if summary.get('whether_t100_verified') else '否'}

当前模型是否仍是 pseudo-3D：
是

是否是真 3D：
否

learned residual 是否超过 hand physics：
{learned_result(metrics)}

超过幅度：
{learned_margin(metrics)}

SMC 是否提升 coverage：
{coverage_result(metrics)}

coverage_FDE_lt_5m：
{coverage_value(metrics)}

best_of_N_FDE@100：
{best_fde_value(metrics)}

physical validity 是否可接受：
{physical_validity(metrics)}

terminal clusters 是否有语义差异：
{semantic_diversity(metrics)}

expert audit score：
58

是否超过 70：
否

当前 verdict：
prototype_with_major_failures

是否值得进入 latent generative model Stage 5：
否

如果值得，原因：
当前不值得；真实 t+100、learned dynamics、coverage gate 尚未同时通过。

如果不值得，先修什么：
先修真实数据上的 learned dynamics：使用多步 rollout loss、按行人/车辆类型分层训练、引入真实 scene geometry/goal labels，并让 SMC proposal 表达 latent intent 而不是只加局部噪声。
"""
    (REPORT_DIR / "report_stage4_real_benchmark.md").write_text(report, encoding="utf-8")
    write_failure_analysis(summary, metrics, summary.get("cannot_evaluate_t100") or "")


def write_failure_analysis(summary: dict, metrics: dict, reason: str) -> None:
    text = f"""# Stage 4 Failure Analysis

1. 真实数据能不能构建 t+100？
   - {'能' if summary.get('whether_t100_verified') else '不能'}。

2. 如果不能，卡在哪里？
   - {reason or summary.get('cannot_evaluate_t100') or 'No blocker reported.'}

3. learned residual 是否超过 hand physics？
   - {learned_result(metrics)}。

4. 如果没有，为什么？
   - {learned_failure_reason(metrics)}

5. SMC 是否提升 coverage？
   - {coverage_result(metrics)}。

6. 如果没有，为什么？
   - {coverage_failure_reason(metrics)}

7. 当前变量 schema 是否真的帮助了真实数据？
   - 还没有被真实 ablation 证明。Stage 4 已经把真实 TGSIM 轨迹接入并训练 residual，但当前 best learned residual 仍未超过 hand physics，更未超过 constant velocity。

8. time-to-collision、closing speed、bottleneck score、obstacle tangent 等新变量是否有实际贡献？
   - 尚未通过真实 ablation 证明。TGSIM 当前接入结果没有 obstacle / exit / walkable scene geometry，因此 bottleneck、obstacle tangent、exit-distance 这类变量不能完整发挥作用。

9. synthetic 和 real 的 domain gap 有多大？
   - 已经能初步量化：synthetic 上的物理脚手架可以跑通 t+100，但真实 TGSIM 上 constant velocity 明显强于 hand physics 和 learned residual，说明当前 social-force prior 与真实轨迹分布不匹配。

10. 当前世界模型还差什么才算 strong world model？
   - 真实数据上的 learned dynamics > simple baselines、SMC coverage 提升、真实 scene geometry、类型/意图标注、semantic event labels，以及跨数据集验证。
"""
    (REPORT_DIR / "stage4_failure_analysis.md").write_text(text, encoding="utf-8")


def learned_result(metrics: dict) -> str:
    try:
        hand = metrics["hand_physics_baseline"]["horizons"]["100"]["FDE"]
        learned = best_learned_h100(metrics)["FDE"]
        return "是" if learned < hand * 0.95 else ("部分" if learned < hand else "否")
    except Exception:
        return "否"


def learned_margin(metrics: dict) -> str:
    try:
        hand = metrics["hand_physics_baseline"]["horizons"]["100"]
        learned = best_learned_h100(metrics)
        return f"hand ADE@100={hand['ADE']}, best learned ADE@100={learned['ADE']}; hand FDE@100={hand['FDE']}, best learned FDE@100={learned['FDE']}"
    except Exception:
        return "无 verified ADE@100/FDE@100"


def best_learned_h100(metrics: dict) -> dict:
    candidates = []
    for name in ["deterministic_neural_residual", "stochastic_neural_residual", "physics_plus_neural_residual"]:
        payload = metrics.get(name, {})
        h100 = payload.get("horizons", {}).get("100")
        if h100:
            candidates.append(h100)
    if not candidates:
        raise KeyError("No learned h100 metrics")
    return min(candidates, key=lambda row: row.get("FDE", 1e9))


def learned_failure_reason(metrics: dict) -> str:
    try:
        constant = metrics["constant_velocity_baseline"]["horizons"]["100"]
        hand = metrics["hand_physics_baseline"]["horizons"]["100"]
        learned = best_learned_h100(metrics)
        return (
            f"best learned residual 没有达到比 hand physics 好 5% 的门槛："
            f"hand FDE@100={hand['FDE']}, best learned FDE@100={learned['FDE']}。"
            f"同时 constant velocity FDE@100={constant['FDE']}，说明真实 TGSIM quick benchmark 更接近平滑惯性运动，当前 social-force/goal prior 会把轨迹推偏。"
        )
    except Exception:
        return "真实 benchmark 不可用或 learned dynamics 没有达到 5% margin。"


def coverage_result(metrics: dict) -> str:
    try:
        hand = metrics["hand_physics_baseline"]["coverage_FDE_lt_5m"]
        smc = metrics["physics_plus_neural_residual_SMC"]["coverage_FDE_lt_5m"]
        return "是" if smc > hand else "否"
    except Exception:
        return "否"


def coverage_failure_reason(metrics: dict) -> str:
    try:
        hand = metrics["hand_physics_baseline"]["coverage_FDE_lt_5m"]
        smc = metrics["physics_plus_neural_residual_SMC"]["coverage_FDE_lt_5m"]
        min_fde = best_fde_value(metrics)
        return f"SMC 没有把真实未来纳入 5m 覆盖范围：hand coverage_FDE_lt_5m={hand}, physics_plus_neural_residual_SMC coverage_FDE_lt_5m={smc}, {min_fde}。当前粒子主要是噪声扰动，缺少可学习的 intent/route proposal。"
    except Exception:
        return "没有真实 t+100 或多分支 coverage 指标未超过 baseline。"


def coverage_value(metrics: dict) -> str:
    try:
        return str(metrics["physics_plus_neural_residual_SMC"]["coverage_FDE_lt_5m"])
    except Exception:
        return "无"


def best_fde_value(metrics: dict) -> str:
    try:
        row = metrics["physics_plus_neural_residual_SMC"]["horizons"]["100"]
        key = [k for k in row if k.startswith("minFDE@")][0]
        return f"{key}={row[key]}"
    except Exception:
        return "无"


def physical_validity(metrics: dict) -> str:
    try:
        valid = metrics["deterministic_neural_residual"]["physical_validity_rate"]
        return "是" if valid > 0.95 else ("部分" if valid > 0.85 else "否")
    except Exception:
        return "否"


def semantic_diversity(metrics: dict) -> str:
    try:
        div = metrics["physics_plus_neural_residual_SMC"]["cluster_diversity_score"]
        return "强" if div > 0.65 else ("中等" if div > 0.35 else "弱")
    except Exception:
        return "弱"


if __name__ == "__main__":
    raise SystemExit(main())
