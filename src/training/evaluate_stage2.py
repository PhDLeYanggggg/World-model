from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.data.synthetic_dataset import dataset_summary, split_episodes
from src.inference.cluster_futures import terminal_world_clustering
from src.inference.smc_rollout import rollout_learned_single, rollout_smc
from src.models.baselines import constant_velocity_rollout, hand_physics_rollout
from src.utils.figures import generate_stage2_figures
from src.utils.metrics import aggregate_metric_dicts, trajectory_metrics


MODEL_NAMES = [
    "constant_velocity_baseline",
    "hand_physics_baseline",
    "deterministic_neural_residual",
    "stochastic_neural_residual",
    "hand_physics_SMC",
    "learned_neural_SMC",
    "physics_plus_neural_residual_SMC",
]


def evaluate_stage2(episodes: List[Dict], model_bundle: Dict, cfg: Dict, quick: bool = True) -> Dict:
    test_episodes_all = split_episodes(episodes, "test")
    eval_count = int(cfg["inference"]["quick_eval_episodes"] if quick else len(test_episodes_all))
    test_episodes = select_eval_episodes(test_episodes_all, eval_count)
    horizons = [int(h) for h in cfg["inference"]["horizons"]]
    particles = int(cfg["inference"]["quick_particles"] if quick else cfg["inference"]["particles"])
    history_steps = int(cfg["training"]["history_steps"])
    horizon = 100
    results_by_model: Dict[str, List[Dict]] = {name: [] for name in MODEL_NAMES}
    clusters_by_model: Dict[str, List[Dict]] = {name: [] for name in MODEL_NAMES}
    sample_predictions: Dict[str, Dict] = {}
    sample_episode_payload: Dict | None = None

    for episode_index, episode in enumerate(test_episodes):
        states = episode["states"]
        scene = episode["scene"]
        start_t = history_steps
        history = states[start_t - history_steps + 1 : start_t + 1]
        true_future = states[start_t : start_t + horizon + 1]
        true_event = episode["meta"]["event_label"]
        rollouts = run_episode_rollouts(episode, history, cfg, model_bundle, particles, horizon)

        for name, rollout in rollouts.items():
            metrics = trajectory_metrics(
                rollout["trajectories"],
                rollout["weights"],
                true_future,
                scene,
                horizons,
                sigma_m=float(cfg["inference"]["obs_likelihood_sigma_m"]),
            )
            cluster_payload = terminal_world_clustering(rollout["trajectories"], rollout["weights"], scene, true_future, true_event)
            metrics["cluster_diversity_score"] = cluster_payload["cluster_diversity_score"]
            metrics["semantic_event_accuracy"] = cluster_payload["semantic_event_accuracy"]
            metrics["true_event"] = true_event
            metrics["top_cluster_label"] = cluster_payload["top_cluster_label"]
            metrics["episode_id"] = int(episode["meta"]["episode_id"])
            results_by_model[name].append(metrics)
            clusters_by_model[name].append(cluster_payload)
            if episode_index == 0:
                sample_predictions[name] = rollout
        if episode_index == 0:
            sample_episode_payload = {"states": true_future, "scene": scene, "meta": episode["meta"]}

    summary = {}
    cluster_summary = {}
    for name in MODEL_NAMES:
        summary[name] = aggregate_metric_dicts(results_by_model[name], horizons)
        summary[name]["semantic_event_accuracy"] = round(
            float(np.mean([r["semantic_event_accuracy"] for r in results_by_model[name] if r.get("semantic_event_accuracy") is not None])),
            5,
        )
        summary[name]["cluster_diversity_score"] = round(
            float(np.mean([r["cluster_diversity_score"] for r in results_by_model[name] if r.get("cluster_diversity_score") is not None])),
            5,
        )
        cluster_summary[name] = aggregate_cluster_payloads(clusters_by_model[name])

    metrics_rows = flatten_metrics(summary)
    write_metrics_artifacts(cfg, summary, metrics_rows)
    if sample_episode_payload:
        generate_stage2_figures(
            Path(cfg["output_dir"]) / "figures" / "stage2",
            sample_episode_payload,
            sample_predictions,
            cluster_summary,
            metrics_rows,
            model_bundle["training"],
        )
    report_payload = {
        "dataset": dataset_summary(episodes, cfg["synthetic_dir"]),
        "evaluation_meta": {
            "full_test_episodes": len(test_episodes_all),
            "evaluated_test_episodes": len(test_episodes),
            "evaluated_episode_ids": [int(ep["meta"]["episode_id"]) for ep in test_episodes],
            "evaluated_event_labels": [ep["meta"]["event_label"] for ep in test_episodes],
            "horizon": horizon,
            "particles_requested": int(cfg["inference"]["particles"]),
            "particles_used_in_this_run": particles,
            "quick_mode": bool(quick),
            "eval_selection": "event-balanced quick subset, not smallest-agent-only selection",
            "future_leakage": "No future states are used by proposals or weights; ground truth is used only for metrics and cluster scoring.",
        },
        "summary": summary,
        "metrics_rows": metrics_rows,
        "clusters": cluster_summary,
        "aerialmpt": load_aerialmpt_limitations(),
        "training": model_bundle["training"],
    }
    Path(cfg["reports"]["report_path"]).write_text(build_report_stage2(report_payload), encoding="utf-8")
    return report_payload


def select_eval_episodes(episodes: List[Dict], count: int) -> List[Dict]:
    """Pick a small event-balanced test subset instead of only the smallest-agent episodes."""
    by_event: Dict[str, List[Dict]] = {}
    for episode in sorted(episodes, key=lambda item: (item["meta"]["agents"], item["meta"]["episode_id"])):
        by_event.setdefault(episode["meta"]["event_label"], []).append(episode)
    selected: List[Dict] = []
    while len(selected) < min(count, len(episodes)):
        progressed = False
        for event in sorted(by_event):
            if by_event[event] and len(selected) < count:
                selected.append(by_event[event].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) < min(count, len(episodes)):
        remaining = [episode for episode in episodes if episode not in selected]
        selected.extend(sorted(remaining, key=lambda item: item["meta"]["agents"])[: count - len(selected)])
    return selected


def run_episode_rollouts(episode: Dict, history: np.ndarray, cfg: Dict, model_bundle: Dict, particles: int, horizon: int) -> Dict[str, Dict]:
    scene = episode["scene"]
    dt = float(cfg["world"]["dt"])
    max_speed = float(cfg["world"]["max_speed_mps"])
    max_accel = float(cfg["world"]["max_accel_mps2"])
    deterministic_model = model_bundle["deterministic_model"]
    stochastic_model = model_bundle["stochastic_model"]
    normalization = model_bundle["normalization"]
    seed = int(cfg["seed"]) + int(episode["meta"]["episode_id"]) * 37
    cv = constant_velocity_rollout(history, scene, horizon, dt, max_speed, max_accel)[None]
    physics = hand_physics_rollout(history, scene, horizon, dt, max_speed, max_accel)[None]
    deterministic = rollout_learned_single(history, scene, cfg, deterministic_model, normalization, horizon, stochastic=False, seed=seed)[None]
    stochastic_branches = []
    stochastic_count = min(8, particles)
    for k in range(stochastic_count):
        stochastic_branches.append(rollout_learned_single(history, scene, cfg, stochastic_model, normalization, horizon, stochastic=True, seed=seed + 100 + k))
    stochastic = np.stack(stochastic_branches, axis=0)
    hand_smc = rollout_smc(history, scene, cfg, None, None, "hand_physics_proposal", particles, horizon, seed=seed + 200)
    neural_smc = rollout_smc(
        history,
        scene,
        cfg,
        stochastic_model,
        normalization,
        "learned_neural_proposal",
        particles,
        horizon,
        stochastic=True,
        seed=seed + 300,
    )
    hybrid_smc = rollout_smc(
        history,
        scene,
        cfg,
        stochastic_model,
        normalization,
        "physics_plus_neural_residual_proposal",
        particles,
        horizon,
        stochastic=True,
        seed=seed + 400,
    )
    return {
        "constant_velocity_baseline": {"trajectories": cv, "weights": np.ones(1), "particles": 1},
        "hand_physics_baseline": {"trajectories": physics, "weights": np.ones(1), "particles": 1},
        "deterministic_neural_residual": {"trajectories": deterministic, "weights": np.ones(1), "particles": 1},
        "stochastic_neural_residual": {"trajectories": stochastic, "weights": np.ones(stochastic.shape[0]) / stochastic.shape[0], "particles": stochastic.shape[0]},
        "hand_physics_SMC": hand_smc,
        "learned_neural_SMC": neural_smc,
        "physics_plus_neural_residual_SMC": hybrid_smc,
    }


def aggregate_cluster_payloads(payloads: List[Dict]) -> Dict:
    if not payloads:
        return {"clusters": [], "cluster_diversity_score": 0.0}
    by_label: Dict[str, List[Dict]] = {}
    for payload in payloads:
        for cluster in payload["clusters"]:
            by_label.setdefault(cluster["semantic_label"], []).append(cluster)
    rows = []
    for idx, (label, clusters) in enumerate(sorted(by_label.items())):
        probability_across_episodes = []
        for payload in payloads:
            matches = [cluster for cluster in payload["clusters"] if cluster["semantic_label"] == label]
            probability_across_episodes.append(sum(float(cluster["probability_mass"]) for cluster in matches))
        rows.append(
            {
                "cluster_id": idx,
                "semantic_label": label,
                "probability_mass": round(float(np.mean(probability_across_episodes)), 5),
                "representative_trajectory_id": int(clusters[0]["representative_trajectory_id"]),
                "mean_ADE@100": mean_optional(clusters, "mean_ADE@100"),
                "mean_FDE@100": mean_optional(clusters, "mean_FDE@100"),
                "mean_collision_rate": mean_optional(clusters, "mean_collision_rate"),
                "mean_obstacle_violation_rate": mean_optional(clusters, "mean_obstacle_violation_rate"),
                "mean_boundary_violation_rate": mean_optional(clusters, "mean_boundary_violation_rate"),
                "mean_goal_reached_rate": mean_optional(clusters, "mean_goal_reached_rate"),
                "mean_jam_duration": mean_optional(clusters, "mean_jam_duration"),
                "confidence": mean_optional(clusters, "confidence"),
                "is_credible": bool(np.mean([c["is_credible"] for c in clusters]) >= 0.5),
                "explanation": clusters[0]["explanation"],
            }
        )
    diversity = float(np.mean([payload["cluster_diversity_score"] for payload in payloads]))
    return {"clusters": sorted(rows, key=lambda row: row["probability_mass"], reverse=True), "cluster_diversity_score": round(diversity, 5)}


def mean_optional(rows: List[Dict], key: str):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return round(float(np.mean(values)), 5) if values else None


def flatten_metrics(summary: Dict) -> List[Dict]:
    rows = []
    for model, metrics in summary.items():
        row = {
            "model": model,
            "ADE@1": metrics["horizons"]["1"]["ADE_m"],
            "ADE@10": metrics["horizons"]["10"]["ADE_m"],
            "ADE@25": metrics["horizons"]["25"]["ADE_m"],
            "ADE@50": metrics["horizons"]["50"]["ADE_m"],
            "ADE@100": metrics["horizons"]["100"]["ADE_m"],
            "FDE@1": metrics["horizons"]["1"]["FDE_m"],
            "FDE@10": metrics["horizons"]["10"]["FDE_m"],
            "FDE@25": metrics["horizons"]["25"]["FDE_m"],
            "FDE@50": metrics["horizons"]["50"]["FDE_m"],
            "FDE@100": metrics["horizons"]["100"]["FDE_m"],
            "best_of_64_ADE@100": metrics["horizons"]["100"]["best_of_64_ADE_m"],
            "best_of_64_FDE@100": metrics["horizons"]["100"]["best_of_64_FDE_m"],
            "branch_count": metrics["branch_count"],
            "collision_violation_rate": metrics["collision_violation_rate"],
            "obstacle_violation_rate": metrics["obstacle_violation_rate"],
            "boundary_violation_rate": metrics["boundary_violation_rate"],
            "max_speed_violation_rate": metrics["max_speed_violation_rate"],
            "acceleration_violation_rate": metrics["acceleration_violation_rate"],
            "trajectory_smoothness": metrics["trajectory_smoothness"],
            "coverage@64": metrics["coverage@64"],
            "NLL_endpoint_t100": metrics["NLL_endpoint_t100"],
            "cluster_diversity_score": metrics["cluster_diversity_score"],
            "semantic_event_accuracy": metrics["semantic_event_accuracy"],
        }
        rows.append(row)
    return rows


def write_metrics_artifacts(cfg: Dict, summary: Dict, rows: List[Dict]) -> None:
    Path(cfg["reports"]["metrics_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["reports"]["metrics_json"]).write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    with Path(cfg["reports"]["metrics_csv"]).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    Path(cfg["reports"]["metrics_table"]).write_text(markdown_table(rows), encoding="utf-8")


def load_aerialmpt_limitations() -> Dict:
    path = Path("experiments/outputs/pseudo3d_world_model/summary.json")
    if not path.exists():
        return {
            "available": False,
            "verified_horizons": [1, 5, 10, 12],
            "t100_status": "qualitative free-run only; selected AerialMPT bauma3 slice has no t+100 ground truth.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "dataset": payload.get("dataset", {}),
        "verified_horizons": [1, 5, 10, 12],
        "t100_status": "qualitative free-run only; selected AerialMPT bauma3 slice has no t+100 ground truth.",
        "calibration_quality": payload.get("calibration_quality", {}),
    }


def build_report_stage2(payload: Dict) -> str:
    summary = payload["summary"]
    rows = payload["metrics_rows"]
    clusters = payload["clusters"]
    aerial = payload["aerialmpt"]
    best_hybrid = summary["physics_plus_neural_residual_SMC"]["horizons"]["100"]["ADE_m"]
    physics = summary["hand_physics_baseline"]["horizons"]["100"]["ADE_m"]
    deterministic_ade = summary["deterministic_neural_residual"]["horizons"]["100"]["ADE_m"]
    hybrid_smc = summary["physics_plus_neural_residual_SMC"]["coverage@64"]
    hybrid_beats_physics = best_hybrid < physics
    cluster_diversity = summary["physics_plus_neural_residual_SMC"]["cluster_diversity_score"]
    quality = quality_label(summary["physics_plus_neural_residual_SMC"]["horizons"]["100"]["FDE_m"])
    if hybrid_beats_physics:
        neural_vs_physics = "是"
    elif deterministic_ade < physics:
        neural_vs_physics = "部分超过"
    else:
        neural_vs_physics = "否"
    smc_coverage = "是" if hybrid_smc > summary["deterministic_neural_residual"]["coverage@64"] else "否"
    physics_effect = "强" if summary["physics_plus_neural_residual_SMC"]["obstacle_violation_rate"] < 0.02 and summary["physics_plus_neural_residual_SMC"]["boundary_violation_rate"] < 0.02 else "中等"
    terminal_quality = "强" if cluster_diversity > 0.65 else ("中等" if cluster_diversity > 0.35 else "弱")

    return f"""# Stage 2: Learned 2.5D Crowd Physics World Model

## 1. What Changed

上一阶段是 pseudo-3D physics-informed SMC scaffold。Stage 2 增加了 SyntheticPhysicalCrowd2.5D 长轨迹环境、episode-level train/val/test split、learned neural residual transition、stochastic latent residual variant、SMC proposal 对比、t+100 真值评估、物理违规指标和 semantic terminal clustering。

这个版本仍然不是 true 3D；它是 Z=0 ground-plane 上的 2.5D / pseudo-3D 人群轨迹世界模型。

## 2. SyntheticPhysicalCrowd2.5D

{markdown_table([payload['dataset']])}

为什么需要 synthetic data：AerialMPT bauma3 当前片段只有 16 帧，从 start frame 4 出发最多真实评估到 t+12。Synthetic 数据有 160/200 帧，因此 t+100 有真实状态，不需要把 free-run 当作准确预测。

## 3. Learned Transition

训练目标是 world-coordinate residual acceleration：

```text
A_residual = A_true_next - A_hand_physics
A_final = A_hand_physics + A_neural_residual
S_next = project_constraints(integrate(S_t, A_final))
```

训练日志包含 position、velocity、acceleration、heading、goal direction、collision surrogate、obstacle/boundary surrogate、speed、acceleration、smoothness、stochastic diversity 和 KL 项。几何违规的最终可信指标以 rollout evaluation 为准。

## 4. Strict Synthetic t+100 Evaluation

Evaluation meta:

{markdown_table([payload['evaluation_meta']])}

Metrics:

{markdown_table(rows)}

注意：quick demo 使用的 branch count 可能小于 64；表里的 best-of-64 字段在 quick mode 中表示“当前实际分支数下的 best-of-N”，并由 branch_count 显式标出。

## 5. Semantic Terminal Clustering

{cluster_sections(clusters)}

如果多个 cluster 仍落在相同语义，结论按弱多样性处理；不会把普通多分支采样包装成语义丰富预测。

## 6. AerialMPT Re-Application Limits

Synthetic data: verified t+100.

AerialMPT bauma3: verified only up to t+12: `{aerial.get('verified_horizons')}`.

AerialMPT t+20/t+50/t+100: no ground truth in the selected sequence, qualitative free-run only. 不能报告 ADE@100/FDE@100。

## 7. Failure Cases

- 最大失败风险：learned residual 在 quick synthetic split 上可能没有超过 hand-coded physics，尤其是 long-horizon FDE。
- terminal clusters 仍可能被 collision_risk / jam 类标签吸收，说明 branch semantics 还不够强。
- synthetic dynamics 和 AerialMPT 的真实摄像机、身份跟踪、场景标注之间仍有明显 domain gap。

## Final Conclusions

项目是否跑通：是

当前模型类型：
pseudo-3D physics-informed learned residual state-space world model

是否是真 3D：
否，是 2.5D / pseudo-3D

是否是游戏预测：
否，是真实人物物理轨迹世界模型

是否已经从 scaffold 变成 learned world model：
部分是

Synthetic t+100 是否可验证：
是

Synthetic t+100 预测质量：
{quality}

AerialMPT t+12 预测质量：
弱

AerialMPT t+100 预测质量：
只能 qualitative free-run / 不可评估

learned neural residual 是否超过 hand physics：
{neural_vs_physics}

SMC 是否提升多分支 coverage：
{smc_coverage}

物理约束是否有效：
{physics_effect}

terminal clusters 是否有语义差异：
{terminal_quality}

当前最大局限：
1. AerialMPT 当前片段没有 t+100 真值，真实数据长预测不可验证。
2. homography / camera calibration 仍是弱标定，不是真 3D。
3. neural residual 训练数据来自合成物理，真实 domain transfer 还没有被证明。

下一步最值得做：
1. 接入 Stanford Drone / TrajNet++ / ETH-UCY 等更长真实轨迹。
2. 为真实场景补 ground-plane homography、walkable/obstacle polygon 和 exit/goal 标注。
3. 用真实长轨迹做 supervised residual fine-tuning，并报告 t+100 verified metrics。
"""


def quality_label(fde100: float) -> str:
    if fde100 < 2.0:
        return "强"
    if fde100 < 5.0:
        return "中等"
    if fde100 < 10.0:
        return "弱"
    return "不可靠"


def cluster_sections(clusters: Dict) -> str:
    parts = []
    for model, payload in clusters.items():
        parts.append(f"### {model}\n\n{markdown_table(payload['clusters'])}")
    return "\n\n".join(parts)


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._"
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                value = round(value, 5)
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value[:10])
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
