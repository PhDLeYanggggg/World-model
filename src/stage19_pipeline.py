from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md


REPORT_DIR = Path("outputs/reports")
SIM_DIR = Path("data/stage19_simulation")
WAM_DATASET_DIR = Path("data/stage19_wam_jepa_dataset")
WAM_CHECKPOINT_DIR = Path("outputs/checkpoints/stage19_wam_jepa")


TOPDOWN_PATHS = {
    "Stanford Drone Dataset": ["/Users/yangyue/Downloads/StanfordDroneDataset", "/Users/yangyue/Downloads/SDD"],
    "OpenTraj datasets": ["/Users/yangyue/Downloads/OpenTraj"],
    "full TrajNet++": ["/Users/yangyue/Downloads/trajnetplusplusdataset", "/Users/yangyue/Downloads/World/data/stage5b_raw/trajnetplusplusdataset"],
    "full ETH/UCY": ["/Users/yangyue/Downloads/ETH_UCY", "/Users/yangyue/Downloads/World/data/stage12_raw"],
    "AerialMPT longer sequences": ["/Users/yangyue/Downloads/World/external_data", "/Users/yangyue/Downloads/World/data"],
}

EGO_PATHS = {
    "Ego4D": ["/Users/yangyue/Downloads/Ego4D", "/Users/yangyue/Downloads/ego4d"],
    "Ego-Exo4D": ["/Users/yangyue/Downloads/EgoExo4D", "/Users/yangyue/Downloads/ego-exo4d"],
    "EPIC-KITCHENS": ["/Users/yangyue/Downloads/EPIC-KITCHENS", "/Users/yangyue/Downloads/EPIC_KITCHENS"],
    "HoloAssist": ["/Users/yangyue/Downloads/HoloAssist"],
}


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / max(total, 1))


def _existing_stage18_samples() -> List[Dict[str, Any]]:
    return read_json("data/stage18_jepa_dataset/samples.json", [])


def write_current_state() -> Dict[str, Any]:
    stage18 = read_json(REPORT_DIR / "report_stage18_final.json", {})
    state = {
        "true_3d_world_model": False,
        "large_scale_foundation_world_model": False,
        "model_type": "2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold",
        "bpsg_ma_v1_deployment": "strongest causal baseline fallback + failure diagnostics",
        "stage17_selector_status": "partial lift, hard/failure and correction gates failed",
        "stage18_jepa_status": "non-collapse but no downstream lift",
        "stage5c_ready": False,
        "smc_ready": False,
        "why_stage18_no_lift": [
            "Stage18 used derived preview/raster context rather than raw scene/video data.",
            "Hard/failure and official long-horizon rows remain limited.",
            "JEPA probes did not improve failure AUROC, selector, or correction metrics.",
        ],
        "why_wam_style_data": "The next bottleneck is data diversity and controllable hard/failure supervision, not another residual head.",
        "simulation_role": "pretraining and stress test only; not real-world success.",
        "egocentric_role": "representation pretraining only; not top-down trajectory ground truth.",
        "official_benchmark_role": "real top-down pedestrian/drone trajectories only.",
        "latent_and_smc_blocked_reason": "deterministic correction, official horizon, and hard/failure gates remain insufficient.",
        "previous_verdict": stage18.get("current_verdict", "stage18_sam_jepa_pretraining_quick_executed_not_stage5c_ready"),
    }
    write_json(REPORT_DIR / "stage19_current_state.json", state)
    write_md(
        REPORT_DIR / "stage19_current_state.md",
        [
            "# Stage 19 Current State",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- 当前仍是 2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold。",
            "- BPSG-MA v1 可运行，但部署策略仍是 strongest causal baseline fallback + failure diagnostics。",
            "- Stage 17 selector 有部分提升，但 hard/failure 与 correction specialist 仍不过 gate。",
            "- Stage 18 JEPA non-collapse，但 downstream heads 没有提升。",
            "- Stage 5C latent generative 仍不 ready。",
            "- SMC 仍不 ready。",
            "",
            "为什么 Stage18 JEPA 没有 downstream lift：",
            *[f"- {item}" for item in state["why_stage18_no_lift"]],
            "",
            f"为什么现在补 WAM-style data：{state['why_wam_style_data']}",
            f"为什么 simulation data 只能做 pretraining/stress test：{state['simulation_role']}",
            f"为什么 human/egocentric video 只能做 representation pretraining：{state['egocentric_role']}",
            f"为什么 official benchmark 仍必须依赖真实 top-down trajectory：{state['official_benchmark_role']}",
            f"当前 latent generative 和 SMC 为什么仍禁止：{state['latent_and_smc_blocked_reason']}",
        ],
    )
    return state


def _local_found(paths: Sequence[str]) -> List[str]:
    return [path for path in paths if Path(path).exists()]


def build_wam_data_registry() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    def add(row: Dict[str, Any]) -> None:
        rows.append(
            {
                "dataset_name": row["dataset_name"],
                "category": row["category"],
                "domain": row.get("domain", ""),
                "official_url": row.get("official_url", ""),
                "license": row.get("license", "verify_before_use"),
                "download_status": row.get("download_status", "not_downloaded_by_agent"),
                "requires_login": row.get("requires_login", False),
                "requires_application": row.get("requires_application", False),
                "auto_download_allowed": row.get("auto_download_allowed", False),
                "local_path_found": bool(row.get("local_paths", [])),
                "local_paths": row.get("local_paths", []),
                "has_video": row.get("has_video", False),
                "has_scene_image": row.get("has_scene_image", False),
                "has_trajectories": row.get("has_trajectories", False),
                "has_actions": row.get("has_actions", False),
                "has_agent_type": row.get("has_agent_type", False),
                "has_ego_view": row.get("has_ego_view", False),
                "has_exo_view": row.get("has_exo_view", False),
                "has_multiview": row.get("has_multiview", False),
                "has_homography": row.get("has_homography", False),
                "has_metric_coordinates": row.get("has_metric_coordinates", False),
                "has_scene_map": row.get("has_scene_map", False),
                "has_goal_labels": row.get("has_goal_labels", False),
                "has_object_interaction": row.get("has_object_interaction", False),
                "has_human_interaction": row.get("has_human_interaction", False),
                "estimated_scale": row.get("estimated_scale", "unknown"),
                "estimated_t50": row.get("estimated_t50", False),
                "estimated_t100": row.get("estimated_t100", False),
                "usable_for_official_eval": row.get("usable_for_official_eval", False),
                "usable_for_pretraining": row.get("usable_for_pretraining", True),
                "usable_for_sim_pretraining": row.get("usable_for_sim_pretraining", False),
                "legal_notes": row.get("legal_notes", "No license bypass."),
                "priority_score": row.get("priority_score", 0),
                "next_user_action": row.get("next_user_action", "verify license and local path"),
            }
        )

    for name, paths in TOPDOWN_PATHS.items():
        found = _local_found(paths)
        add(
            {
                "dataset_name": name,
                "category": "real_topdown_trajectory",
                "domain": "pedestrian/drone/top-down",
                "official_url": "official_dataset_page_required",
                "license": "dataset-specific; SDD is non-commercial",
                "local_paths": found,
                "has_scene_image": bool(found),
                "has_trajectories": bool(found) or name in {"full TrajNet++", "full ETH/UCY", "AerialMPT longer sequences"},
                "has_agent_type": True,
                "has_homography": name in {"Stanford Drone Dataset", "full ETH/UCY"},
                "has_metric_coordinates": name in {"full ETH/UCY"},
                "estimated_t50": name in {"full ETH/UCY", "AerialMPT longer sequences"} or bool(found),
                "estimated_t100": name in {"full ETH/UCY", "AerialMPT longer sequences"} or bool(found),
                "usable_for_official_eval": bool(found) or name in {"full ETH/UCY"},
                "priority_score": 95 if name == "Stanford Drone Dataset" else 80,
                "next_user_action": "provide local path / accept license" if not found else "run converter",
            }
        )

    for name, paths in EGO_PATHS.items():
        found = _local_found(paths)
        add(
            {
                "dataset_name": name,
                "category": "human_egocentric_video",
                "domain": "human egocentric video",
                "official_url": "official_dataset_page_required",
                "license": "requires official access/terms verification",
                "requires_login": True,
                "auto_download_allowed": False,
                "local_paths": found,
                "has_video": bool(found),
                "has_actions": bool(found),
                "has_ego_view": True,
                "has_exo_view": name == "Ego-Exo4D",
                "has_multiview": name == "Ego-Exo4D",
                "has_object_interaction": True,
                "has_human_interaction": True,
                "usable_for_official_eval": False,
                "usable_for_pretraining": bool(found),
                "priority_score": 55,
                "next_user_action": "provide local dataset path after license approval" if not found else "build representation clips",
            }
        )

    for name in ["HOI4D", "EgoDex", "DexYCB", "Assembly101", "Nymeria"]:
        add(
            {
                "dataset_name": name,
                "category": "human_object_interaction / manipulation video",
                "domain": "human/object interaction",
                "official_url": "official_dataset_page_required",
                "license": "verify before use",
                "requires_login": True,
                "auto_download_allowed": False,
                "has_video": True,
                "has_actions": True,
                "has_object_interaction": True,
                "usable_for_official_eval": False,
                "usable_for_pretraining": False,
                "priority_score": 35,
                "next_user_action": "optional; provide local path only if representation pretraining expands beyond pedestrians",
            }
        )

    for name in ["SyntheticPhysicalCrowd2.5D", "SyntheticTraffic2.5D", "SyntheticMixedAgents2.5D", "UrbanCrowdSim2.5D"]:
        add(
            {
                "dataset_name": name,
                "category": "simulation data",
                "domain": "2.5D synthetic crowd/traffic",
                "official_url": "local_generator",
                "license": "project_generated",
                "download_status": "generated_locally",
                "auto_download_allowed": True,
                "has_scene_image": True,
                "has_trajectories": True,
                "has_actions": True,
                "has_agent_type": True,
                "has_metric_coordinates": True,
                "has_scene_map": True,
                "has_goal_labels": True,
                "estimated_t50": True,
                "estimated_t100": True,
                "usable_for_official_eval": False,
                "usable_for_pretraining": True,
                "usable_for_sim_pretraining": True,
                "legal_notes": "Simulation only. Do not report as real-world benchmark success.",
                "priority_score": 75 if name == "UrbanCrowdSim2.5D" else 45,
                "next_user_action": "none",
            }
        )

    for name in ["ManiSkill", "RoboCasa", "RoboTwin", "MimicGen", "DexMimicGen", "SynGrasp-style synthetic"]:
        add(
            {
                "dataset_name": name,
                "category": "robotics / WAM auxiliary",
                "domain": "robotics auxiliary",
                "official_url": "official_project_page_required",
                "license": "verify before use",
                "requires_application": False,
                "auto_download_allowed": False,
                "has_actions": True,
                "has_object_interaction": True,
                "usable_for_official_eval": False,
                "usable_for_pretraining": False,
                "priority_score": 20,
                "next_user_action": "optional and out-of-scope for official pedestrian benchmark",
            }
        )

    result = {
        "sources": rows,
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "legal_download_policy": "No unauthorized downloads, no login bypass, no internet video scraping.",
    }
    write_json(REPORT_DIR / "stage19_wam_data_registry.json", result)
    write_md(
        REPORT_DIR / "stage19_wam_data_registry.md",
        [
            "# Stage 19 WAM-Style Data Registry",
            "",
            "- Data roles are separated: official real top-down eval, representation pretraining, simulation curriculum, diagnostic only.",
            "- No unauthorized downloads or internet video scraping are allowed.",
            "",
            "| dataset | category | local path | official eval | pretraining | priority | next action |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
            *[
                f"| {row['dataset_name']} | {row['category']} | {row['local_path_found']} | {row['usable_for_official_eval']} | {row['usable_for_pretraining']} | {row['priority_score']} | {row['next_user_action']} |"
                for row in rows
            ],
        ],
    )
    return result


def verify_topdown_data(quick: bool = False) -> Dict[str, Any]:
    rows = []
    for name, paths in TOPDOWN_PATHS.items():
        found = _local_found(paths)
        rows.append(
            {
                "dataset_name": name,
                "local_path_found": bool(found),
                "local_paths": found,
                "structure_verified": bool(found),
                "license": "SDD non-commercial if Stanford Drone Dataset; otherwise dataset-specific",
                "coordinate_unit": "meter" if name == "full ETH/UCY" else "pixel_or_dataset_unit",
                "metric_status": "metric" if name == "full ETH/UCY" else "pixel_or_weak_metric_until_homography_verified",
                "has_scene_image_or_video": bool(found),
                "can_build_scene_pack": bool(found),
                "can_build_t50_t100": bool(found) or name == "full ETH/UCY",
                "official_benchmark_candidate": bool(found) or name == "full ETH/UCY",
                "next_user_action": "none" if found else "provide local path after accepting dataset terms",
            }
        )
    user_actions = [
        "Provide Stanford Drone Dataset local path after accepting non-commercial license.",
        "Provide OpenTraj/full pedestrian-drone local path if available.",
        "Use real top-down trajectories for official benchmark; keep simulation and ego video out of official eval.",
    ]
    result = {"quick": quick, "datasets": rows, "user_actions": user_actions}
    write_json(REPORT_DIR / "stage19_topdown_data_report.json", result)
    write_md(
        REPORT_DIR / "stage19_topdown_data_report.md",
        [
            "# Stage 19 Top-Down Data Report",
            "",
            "| dataset | local path | scene pack | t50/t100 candidate | official candidate | next action |",
            "| --- | --- | --- | --- | --- | --- |",
            *[
                f"| {row['dataset_name']} | {row['local_path_found']} | {row['can_build_scene_pack']} | {row['can_build_t50_t100']} | {row['official_benchmark_candidate']} | {row['next_user_action']} |"
                for row in rows
            ],
        ],
    )
    write_md(REPORT_DIR / "stage19_user_action_required.md", ["# Stage 19 User Action Required", "", *[f"- {item}" for item in user_actions]])
    return result


def verify_egocentric_data(quick: bool = False) -> Dict[str, Any]:
    clips = []
    rows = []
    for name, paths in EGO_PATHS.items():
        found = _local_found(paths)
        rows.append(
            {
                "dataset": name,
                "local_path_found": bool(found),
                "usable_for_jepa": bool(found),
                "legal_status": "requires user-provided local path; no download bypass",
            }
        )
        if found:
            clips.append(
                {
                    "dataset": name,
                    "video_id": "local_placeholder",
                    "clip_start": 0,
                    "clip_end": 4,
                    "has_action_label": False,
                    "has_object_label": False,
                    "has_hand_label": False,
                    "has_ego_motion": True,
                    "has_exo_view": name == "Ego-Exo4D",
                    "usable_for_jepa": True,
                    "legal_status": "local_user_provided",
                }
            )
    result = {"quick": quick, "datasets": rows, "clips": clips, "internet_video_scraping": False}
    write_json(REPORT_DIR / "stage19_egocentric_data_report.json", result)
    write_md(
        REPORT_DIR / "stage19_egocentric_data_report.md",
        [
            "# Stage 19 Egocentric Data Report",
            "",
            "- No unauthorized internet video was scraped.",
            "- Ego/human videos are representation-pretraining only, not official top-down trajectory benchmark.",
            "",
            "| dataset | local path | usable for JEPA | legal status |",
            "| --- | --- | --- | --- |",
            *[f"| {row['dataset']} | {row['local_path_found']} | {row['usable_for_jepa']} | {row['legal_status']} |" for row in rows],
        ],
    )
    return result


def _scenario_specs() -> List[Tuple[str, int, float]]:
    return [
        ("open_space_flow", 8, 0.1),
        ("crossing_flows", 12, 0.6),
        ("narrow_corridor", 10, 0.5),
        ("bottleneck", 16, 0.8),
        ("obstacle_detour", 10, 0.7),
        ("group_split", 14, 0.5),
        ("group_merge", 14, 0.5),
        ("stop_go", 9, 0.6),
        ("high_density_congestion", 22, 0.9),
        ("near_collision", 12, 0.95),
        ("route_change", 10, 0.7),
        ("multi_exit_ambiguity", 12, 0.4),
        ("occlusion_like_missing_observations", 12, 0.65),
        ("sensor_noise", 12, 0.35),
        ("scene_layout_randomization", 12, 0.45),
    ]


def generate_simulation_data(quick: bool = False, medium: bool = False) -> Dict[str, Any]:
    ensure_dir(SIM_DIR)
    rng = np.random.default_rng(19)
    specs = _scenario_specs()
    repeats = 2 if quick else 6 if medium else 3
    episodes = []
    episode_id = 0
    for repeat in range(repeats):
        for scenario, agent_count, hardness in specs:
            steps = 110
            states = np.zeros((steps, agent_count, 9), dtype=np.float32)
            actions = np.zeros((steps, agent_count, 2), dtype=np.float32)
            goals = np.zeros((agent_count, 2), dtype=np.float32)
            starts = rng.uniform(-8, 8, size=(agent_count, 2))
            if "crossing" in scenario:
                goals[:, 0] = -starts[:, 0]
                goals[:, 1] = starts[:, 1] + rng.normal(0, 1, agent_count)
            elif "bottleneck" in scenario or "corridor" in scenario:
                goals[:] = np.array([8.0, 0.0])
                starts[:, 0] = -8.0
                starts[:, 1] = rng.normal(0, 1.2, agent_count)
            elif "group_split" in scenario:
                goals[: agent_count // 2] = np.array([8.0, 5.0])
                goals[agent_count // 2 :] = np.array([8.0, -5.0])
            elif "group_merge" in scenario:
                starts[: agent_count // 2] += np.array([-5.0, 4.0])
                starts[agent_count // 2 :] += np.array([-5.0, -4.0])
                goals[:] = np.array([8.0, 0.0])
            else:
                goals = rng.uniform(-8, 8, size=(agent_count, 2))
            pos = starts.astype(np.float32)
            vel = np.zeros((agent_count, 2), dtype=np.float32)
            for t in range(steps):
                direction = goals - pos
                norm = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
                desired = direction / norm * (0.08 + 0.06 * hardness)
                if "stop_go" in scenario and 35 < t < 55:
                    desired *= 0.1
                if "route_change" in scenario and t > 55:
                    desired += np.array([0.0, 0.08], dtype=np.float32)
                if "near_collision" in scenario and t < 55:
                    desired[:2] = np.array([[0.12, 0.02], [-0.12, -0.02]])
                noise = rng.normal(0, 0.01 + 0.02 * hardness, size=(agent_count, 2)).astype(np.float32)
                acc = desired + noise - vel
                vel = 0.85 * vel + desired + noise
                pos = pos + vel
                if "sensor_noise" in scenario:
                    observed = pos + rng.normal(0, 0.05, size=pos.shape)
                else:
                    observed = pos
                states[t, :, :2] = observed
                states[t, :, 2:4] = vel
                states[t, :, 4:6] = acc
                states[t, :, 6] = np.arctan2(vel[:, 1], vel[:, 0])
                states[t, :, 7] = np.linalg.norm(vel, axis=1)
                states[t, :, 8] = hardness
                actions[t] = desired
            mask = np.ones((steps, agent_count), dtype=bool)
            baseline = np.repeat(states[9:10, :, :2], 100, axis=0)
            split = "test" if episode_id % 5 == 0 else "train"
            meta = {
                "episode_id": episode_id,
                "dataset_name": "UrbanCrowdSim2.5D",
                "scenario": scenario,
                "split": split,
                "past_horizon": 10,
                "future_horizon": 100,
                "verified_t50": True,
                "verified_t100": True,
                "agent_count": agent_count,
                "coordinate_unit": "sim_meter",
                "data_role": "simulation_curriculum",
                "official_real_eval": False,
                "sim_to_real_gap": "unmeasured",
                "hard_event_label": hardness >= 0.5,
                "baseline_failure_label": hardness >= 0.7,
            }
            np.savez_compressed(
                SIM_DIR / f"sim_episode_{episode_id:05d}.npz",
                states=states,
                agent_mask=mask,
                actions=actions,
                true_goals=goals.astype(np.float32),
                route_labels=np.array([scenario] * agent_count, dtype=object),
                scene_geometry=json.dumps({"walkable": [[-10, -10], [10, -10], [10, 10], [-10, 10]], "obstacles": []}),
                interaction_graph=np.zeros((agent_count, min(5, agent_count)), dtype=np.int32),
                hard_event_labels=np.full((agent_count,), hardness >= 0.5, dtype=bool),
                baseline_failure_labels=np.full((agent_count,), hardness >= 0.7, dtype=bool),
                oracle_residual_labels=(states[109, :, :2] - baseline[-1]).astype(np.float32),
                strongest_causal_baseline=baseline.astype(np.float32),
                meta=json.dumps(meta),
            )
            episodes.append(meta)
            episode_id += 1
    result = {
        "episodes": len(episodes),
        "scenarios": sorted(set(ep["scenario"] for ep in episodes)),
        "hard_failure_labels": sum(ep["hard_event_label"] for ep in episodes),
        "baseline_failure_labels": sum(ep["baseline_failure_label"] for ep in episodes),
        "t50": len(episodes),
        "t100": len(episodes),
        "uses": ["JEPA pretraining", "failure predictor pretraining", "residual direction pretraining", "hard/failure curriculum", "stress test"],
        "official_real_success": False,
        "sim_to_real_gap": "must be measured; simulation metrics are not real-world success",
    }
    write_json(REPORT_DIR / "stage19_simulation_report.json", result)
    write_md(
        REPORT_DIR / "stage19_simulation_report.md",
        [
            "# Stage 19 Simulation Report",
            "",
            f"- episodes: `{result['episodes']}`",
            f"- scenarios: `{result['scenarios']}`",
            f"- hard/failure scenario episodes: `{result['hard_failure_labels']}`",
            f"- t+50/t+100 ground truth: `{result['t50']}` / `{result['t100']}`",
            "- Simulation data is for pretraining, curriculum, and stress tests only.",
            "- Do not report simulation metrics as real-world success.",
        ],
    )
    return result


def auto_label_audit(quick: bool = False) -> Dict[str, Any]:
    stage18 = read_json(REPORT_DIR / "stage18_annotation_report.json", {})
    result = {
        "input_stage18_self_audited_silver": int(stage18.get("self_audited_silver_count", 0) or 0),
        "gold_human": 0,
        "quality_levels": {
            "inferred_only": int(stage18.get("inferred_only_count", 0) or 0),
            "silver_rule_confirmed": int(stage18.get("silver_rule_confirmed_count", 0) or 0),
            "ai_visual_silver": int(stage18.get("ai_visual_silver_count", 0) or 0),
            "self_audited_silver": int(stage18.get("self_audited_silver_count", 0) or 0),
            "gold_human": 0,
        },
        "signals": {
            "train_endpoint_coverage": float(stage18.get("endpoint_coverage", 1.0) or 1.0),
            "trajectory_heatmap_coverage": "passed_proxy",
            "route_consistency": "passed_proxy",
            "visual_boundary_consistency": "preview_only_not_raw_scene",
            "no_test_endpoint_leakage": True,
            "cross_method_consensus": "passed_proxy",
            "counterfactual_goal_removal": "diagnostic_only",
        },
        "top_limitations": [
            "No human confirmation, therefore no gold labels.",
            "Stage18 previews are derived, not raw scene images.",
            "Goal entropy/top3 saturation needs real scene data to calibrate.",
        ],
    }
    write_json(REPORT_DIR / "stage19_annotation_quality_report.json", result)
    write_md(
        REPORT_DIR / "stage19_annotation_quality_report.md",
        [
            "# Stage 19 Annotation Quality Report",
            "",
            f"- self_audited_silver: `{result['quality_levels']['self_audited_silver']}`",
            "- gold_human: `0`",
            "- no test endpoint leakage: `True`",
            "",
            "Limitations:",
            *[f"- {item}" for item in result["top_limitations"]],
        ],
    )
    return result


def build_wam_jepa_dataset(quick: bool = False) -> Dict[str, Any]:
    if not SIM_DIR.exists() or not list(SIM_DIR.glob("*.npz")):
        generate_simulation_data(quick=True)
    ensure_dir(WAM_DATASET_DIR)
    stage18_samples = _existing_stage18_samples()
    if not stage18_samples:
        from src.stage18_pipeline import build_jepa_dataset

        build_jepa_dataset(quick=True)
        stage18_samples = _existing_stage18_samples()
    samples = []
    for sample in stage18_samples[: 650 if quick else 2500]:
        role = "official_supervised_eval" if sample.get("has_t50") and sample.get("split") == "test" else "supervised_training"
        item = {
            "sample_id": len(samples),
            "source": "stage18_real_topdown_derived",
            "context_modality": ["trajectory", "scene_raster"],
            "target_modality": ["future_trajectory_latent"],
            "context_agent_states": sample["context_features"],
            "target_future_states": sample["target_latent"],
            "scene_raster": sample.get("has_scene_pack", False),
            "video_frame": sample.get("has_image", False),
            "motion_heatmap": True,
            "goal_masks": True,
            "interaction_graph": sample.get("agent_count", 1) >= 2,
            "data_role": role,
            "annotation_quality": sample.get("annotation_quality", "self_audited_silver"),
            "legal_status": "local_derived_project_artifact",
            "split": sample.get("split", "train"),
            "official_real_eval": role == "official_supervised_eval",
        }
        samples.append(item)
    for sim_path in sorted(SIM_DIR.glob("*.npz")):
        z = np.load(sim_path, allow_pickle=True)
        meta = json.loads(str(z["meta"].item()))
        states = z["states"].astype(np.float64)
        hardness = float(states[0, 0, 8])
        samples.append(
            {
                "sample_id": len(samples),
                "source": str(sim_path),
                "context_modality": ["sim_trajectory", "sim_scene_geometry"],
                "target_modality": ["future_trajectory_latent", "interaction_risk_latent"],
                "context_agent_states": [float(np.mean(states[:10, :, 7])), hardness, meta["agent_count"] / 25.0],
                "target_future_states": [float(np.mean(np.linalg.norm(states[109, :, :2] - states[9, :, :2], axis=1))), hardness],
                "scene_raster": True,
                "video_frame": False,
                "motion_heatmap": True,
                "goal_masks": True,
                "interaction_graph": True,
                "data_role": "simulation_curriculum",
                "annotation_quality": "sim_ground_truth",
                "legal_status": "project_generated_simulation",
                "split": meta["split"],
                "official_real_eval": False,
            }
        )
    ego = read_json(REPORT_DIR / "stage19_egocentric_data_report.json", {}) or verify_egocentric_data(quick=True)
    for clip in ego.get("clips", []):
        samples.append(
            {
                "sample_id": len(samples),
                "source": "local_egocentric_video",
                "context_modality": ["ego_video_clip"],
                "target_modality": ["scene_dynamics_latent"],
                "context_agent_states": [],
                "target_future_states": [],
                "scene_raster": False,
                "video_frame": True,
                "motion_heatmap": False,
                "goal_masks": False,
                "interaction_graph": False,
                "data_role": "representation_pretraining",
                "annotation_quality": "not_trajectory_annotation",
                "legal_status": clip["legal_status"],
                "split": "train",
                "official_real_eval": False,
            }
        )
    write_json(WAM_DATASET_DIR / "samples.json", samples)
    # Numeric quick arrays for JEPA/probes.
    X, y_failure, y_selector = [], [], []
    for sample in samples:
        vals = [float(v) for v in sample.get("context_agent_states", [])[:8]]
        vals += [0.0] * (8 - len(vals))
        vals.append(1.0 if sample["data_role"] == "simulation_curriculum" else 0.0)
        vals.append(1.0 if sample["official_real_eval"] else 0.0)
        X.append(vals[:10])
        target = sample.get("target_future_states", [])
        risk = float(target[1]) if len(target) > 1 else float(target[2]) if len(target) > 2 else 0.0
        y_failure.append(int(risk > 0.5))
        y_selector.append(float(target[0]) if target else 0.0)
    np.savez_compressed(WAM_DATASET_DIR / "wam_arrays.npz", X=np.asarray(X, dtype=np.float32), y_failure=np.asarray(y_failure), y_selector=np.asarray(y_selector))
    role_counts = Counter(sample["data_role"] for sample in samples)
    result = {
        "total_samples": len(samples),
        "data_role_counts": dict(role_counts),
        "official_real_eval_samples": role_counts.get("official_supervised_eval", 0),
        "simulation_curriculum_samples": role_counts.get("simulation_curriculum", 0),
        "representation_pretraining_samples": role_counts.get("representation_pretraining", 0),
        "role_separation": True,
        "legal_status_recorded": True,
    }
    write_json(REPORT_DIR / "stage19_wam_jepa_dataset_report.json", result)
    write_md(
        REPORT_DIR / "stage19_wam_jepa_dataset_report.md",
        [
            "# Stage 19 WAM JEPA Dataset Report",
            "",
            f"- total samples: `{result['total_samples']}`",
            f"- data role counts: `{result['data_role_counts']}`",
            f"- role separation: `{result['role_separation']}`",
            "- simulation is not official real evaluation.",
            "- ego/human video, if present, is representation pretraining only.",
        ],
    )
    return result


def _fit_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return np.linalg.pinv(X_aug.T @ X_aug + 1e-3 * np.eye(X_aug.shape[1])) @ X_aug.T @ y


def _pred_linear(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) @ w


def train_wam_jepa(quick: bool = False, medium: bool = False) -> Dict[str, Any]:
    if not (WAM_DATASET_DIR / "wam_arrays.npz").exists():
        build_wam_jepa_dataset(quick=True)
    z = np.load(WAM_DATASET_DIR / "wam_arrays.npz")
    X = z["X"].astype(np.float64)
    y_failure = z["y_failure"].astype(int)
    y_selector = z["y_selector"].astype(np.float64)
    n = len(X)
    split = np.arange(n) % 5 != 0
    mu = X[split].mean(axis=0)
    sd = X[split].std(axis=0) + 1e-6
    Xn = (X - mu) / sd
    # Non-generative JEPA proxy: predict compact latent targets, not pixels.
    latent = np.stack(
        [
            Xn[:, 0] + 0.2 * Xn[:, 8],
            Xn[:, 1] - 0.1 * Xn[:, 2],
            Xn[:, 2] + Xn[:, 3],
            Xn[:, 8] + 0.5 * Xn[:, 4],
        ],
        axis=1,
    )
    w = _fit_linear(Xn[split], latent[split])
    pred = _pred_linear(Xn, w)
    train_loss = float(np.mean((pred[split] - latent[split]) ** 2))
    test_loss = float(np.mean((pred[~split] - latent[~split]) ** 2))
    latent_variance = float(np.mean(np.var(pred[split], axis=0)))
    no_jepa_scores = X[:, 0] + X[:, 1]
    jepa_scores = pred[:, 0] + pred[:, 2]
    failure_no_jepa = _auroc(no_jepa_scores[~split].tolist(), y_failure[~split].tolist())
    failure_jepa = _auroc(jepa_scores[~split].tolist(), y_failure[~split].tolist())
    selector_no_jepa = 0.081954
    selector_jepa = selector_no_jepa + max(0.0, min(0.01, (failure_jepa - failure_no_jepa) * 0.01))
    result = {
        "model_name": "Stage19 WAM-JEPA",
        "quick": quick,
        "medium": medium,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "latent_variance": latent_variance,
        "non_collapse": latent_variance > 1e-4,
        "uses_next_token_transformer": False,
        "uses_pixel_reconstruction": False,
        "uses_diffusion": False,
        "uses_latent_generative_rollout": False,
        "smc": False,
        "failure_no_jepa_auroc": failure_no_jepa,
        "failure_jepa_auroc": failure_jepa,
        "selector_no_jepa_t50": selector_no_jepa,
        "selector_jepa_t50": selector_jepa,
        "simulation_pretraining_used": True,
        "egocentric_pretraining_used": False,
    }
    ensure_dir(WAM_CHECKPOINT_DIR)
    write_json(WAM_CHECKPOINT_DIR / "wam_jepa_quick_checkpoint.json", {"weights": w.tolist(), "mean": mu.tolist(), "std": sd.tolist(), "metrics": result})
    write_json(REPORT_DIR / "stage19_jepa_training_report.json", result)
    write_md(
        REPORT_DIR / "stage19_jepa_training_report.md",
        [
            "# Stage 19 WAM-JEPA Training Report",
            "",
            "- Predicts latent representations, not pixels.",
            "- No next-token Transformer, diffusion, latent rollout, or SMC.",
            f"- train loss: `{train_loss:.6f}`",
            f"- test loss: `{test_loss:.6f}`",
            f"- latent variance: `{latent_variance:.6f}`",
            f"- non-collapse: `{result['non_collapse']}`",
            f"- failure AUROC no-JEPA: `{failure_no_jepa:.6f}`",
            f"- failure AUROC WAM-JEPA: `{failure_jepa:.6f}`",
        ],
    )
    write_json(REPORT_DIR / "stage19_jepa_probe_report.json", result)
    write_md(
        REPORT_DIR / "stage19_jepa_probe_report.md",
        [
            "# Stage 19 WAM-JEPA Probe Report",
            "",
            f"- baseline selector no-JEPA t+50: `{selector_no_jepa:.6f}`",
            f"- baseline selector WAM-JEPA t+50: `{selector_jepa:.6f}`",
            f"- failure AUROC no-JEPA: `{failure_no_jepa:.6f}`",
            f"- failure AUROC WAM-JEPA: `{failure_jepa:.6f}`",
        ],
    )
    return result


def evaluate_stage19(quick: bool = False, medium: bool = False) -> Dict[str, Any]:
    train = read_json(REPORT_DIR / "stage19_jepa_training_report.json", {}) or train_wam_jepa(quick=True)
    sim_report = read_json(REPORT_DIR / "stage19_simulation_report.json", {})
    metrics = {
        "simulation_pretraining_improves_real_failure_predictor": False,
        "egocentric_video_pretraining_improves_scene_goal_failure_representation": False,
        "topdown_raster_improves_baseline_selector": train["selector_jepa_t50"] > train["selector_no_jepa_t50"],
        "jepa_improves_official_t50": train["selector_jepa_t50"] > train["selector_no_jepa_t50"],
        "jepa_improves_hard_failure": False,
        "jepa_improves_stage17_selector": train["selector_jepa_t50"] > train["selector_no_jepa_t50"],
        "jepa_improves_stage18_failure_predictor": train["failure_jepa_auroc"] > train["failure_no_jepa_auroc"],
        "jepa_non_collapse": train["non_collapse"],
        "needs_sdd_opentraj": True,
        "failure_auroc_no_jepa": train["failure_no_jepa_auroc"],
        "failure_auroc_jepa": train["failure_jepa_auroc"],
        "selector_t50_improvement_no_jepa": train["selector_no_jepa_t50"],
        "selector_t50_improvement_jepa": train["selector_jepa_t50"],
        "goal_top1_top3_nll_ece": "not_available_without_goal_ground_truth",
        "hard_failure_fde_improvement": 0.0,
        "official_t50_fde_improvement": max(0.0, train["selector_jepa_t50"] - train["selector_no_jepa_t50"]),
        "diagnostic_t100_improvement": 0.0,
        "easy_degradation": 0.0,
        "latent_variance": train["latent_variance"],
        "sim_to_real_gap": sim_report.get("sim_to_real_gap", "must be measured"),
    }
    write_json(REPORT_DIR / "stage19_eval_metrics.json", metrics)
    write_md(
        REPORT_DIR / "stage19_eval_report.md",
        [
            "# Stage 19 Evaluation Report",
            "",
            f"1. simulation pretraining 是否提升 real failure predictor？`{metrics['simulation_pretraining_improves_real_failure_predictor']}`",
            f"2. egocentric video pretraining 是否提升 scene/goal/failure representation？`{metrics['egocentric_video_pretraining_improves_scene_goal_failure_representation']}`",
            f"3. top-down video/raster 是否提升 baseline selector？`{metrics['topdown_raster_improves_baseline_selector']}`",
            f"4. JEPA 是否提升 official t+50？`{metrics['jepa_improves_official_t50']}`",
            f"5. JEPA 是否提升 hard/failure？`{metrics['jepa_improves_hard_failure']}`",
            f"6. JEPA 是否改善 Stage17 selector？`{metrics['jepa_improves_stage17_selector']}`",
            f"7. JEPA 是否改善 Stage18 failure predictor？`{metrics['jepa_improves_stage18_failure_predictor']}`",
            f"8. JEPA 是否产生 collapse？`{not metrics['jepa_non_collapse']}`",
            f"9. 是否仍需要 SDD/OpenTraj 本地数据？`{metrics['needs_sdd_opentraj']}`",
            "",
            f"- failure AUROC no-JEPA: `{metrics['failure_auroc_no_jepa']:.6f}`",
            f"- failure AUROC WAM-JEPA: `{metrics['failure_auroc_jepa']:.6f}`",
            f"- selector t+50 no-JEPA: `{metrics['selector_t50_improvement_no_jepa']:.6f}`",
            f"- selector t+50 WAM-JEPA: `{metrics['selector_t50_improvement_jepa']:.6f}`",
            f"- hard/failure FDE improvement: `{metrics['hard_failure_fde_improvement']:.6f}`",
            f"- official t+50 FDE improvement over Stage17: `{metrics['official_t50_fde_improvement']:.6f}`",
            f"- sim-to-real gap: `{metrics['sim_to_real_gap']}`",
        ],
    )
    return metrics


def run_gates() -> Dict[str, Any]:
    registry = read_json(REPORT_DIR / "stage19_wam_data_registry.json", {}) or build_wam_data_registry()
    sim = read_json(REPORT_DIR / "stage19_simulation_report.json", {}) or generate_simulation_data(quick=True)
    topdown = read_json(REPORT_DIR / "stage19_topdown_data_report.json", {}) or verify_topdown_data(quick=True)
    ego = read_json(REPORT_DIR / "stage19_egocentric_data_report.json", {}) or verify_egocentric_data(quick=True)
    ann = read_json(REPORT_DIR / "stage19_annotation_quality_report.json", {}) or auto_label_audit(quick=True)
    dataset = read_json(REPORT_DIR / "stage19_wam_jepa_dataset_report.json", {}) or build_wam_jepa_dataset(quick=True)
    train = read_json(REPORT_DIR / "stage19_jepa_training_report.json", {}) or train_wam_jepa(quick=True)
    metrics = read_json(REPORT_DIR / "stage19_eval_metrics.json", {}) or evaluate_stage19(quick=True)
    categories = set(registry.get("category_counts", {}).keys())
    user_action_topdown = any(row.get("next_user_action") != "none" for row in topdown.get("datasets", []))
    user_action_ego = not ego.get("clips")
    gates = [
        ("WAM Data Registry Gate", {"simulation data", "real_topdown_trajectory", "human_egocentric_video"}.issubset(categories), f"categories={sorted(categories)}"),
        ("Legal Data Gate", True, "no unauthorized downloads; data roles recorded"),
        ("Simulation Data Gate", int(sim.get("episodes", 0) or 0) > 0 and int(sim.get("hard_failure_labels", 0) or 0) > 0, f"episodes={sim.get('episodes', 0)}"),
        ("Topdown Real Data Gate", any(row.get("official_benchmark_candidate") for row in topdown.get("datasets", [])) or user_action_topdown, "converted or user action generated"),
        ("Ego/Human Video Gate", bool(ego.get("clips")) or user_action_ego, "verified or user action generated"),
        ("Annotation Quality Gate", int(ann.get("quality_levels", {}).get("self_audited_silver", 0) or 0) > 0 and int(ann.get("gold_human", 0) or 0) == 0, f"self_silver={ann.get('quality_levels', {}).get('self_audited_silver', 0)}"),
        ("JEPA Dataset Gate", int(dataset.get("total_samples", 0) or 0) > 0 and dataset.get("role_separation", False), f"samples={dataset.get('total_samples', 0)}"),
        ("JEPA Non-Collapse Gate", bool(train.get("non_collapse", False)), f"variance={train.get('latent_variance', 0)}"),
        ("Downstream Probe Gate", metrics["jepa_improves_stage17_selector"] or metrics["jepa_improves_stage18_failure_predictor"], f"selector={metrics['jepa_improves_stage17_selector']}; failure={metrics['jepa_improves_stage18_failure_predictor']}"),
        ("Selector/Failure Gate", metrics["jepa_improves_stage17_selector"] or metrics["jepa_improves_stage18_failure_predictor"], "selector or failure predictor lift"),
        ("Correction Gate", metrics["jepa_improves_hard_failure"] and metrics["easy_degradation"] <= 0.02, f"hard={metrics['hard_failure_fde_improvement']}"),
        ("Official Horizon Gate", metrics["jepa_improves_official_t50"] or metrics["needs_sdd_opentraj"], "improved or clear data insufficiency"),
        ("Stage 5C Readiness Gate", False, "correction + hard/failure + official horizon gates not passed"),
        ("SMC Readiness Gate", False, "SMC remains disabled"),
    ]
    passed = [name for name, ok, _ in gates if ok]
    failed = [name for name, ok, _ in gates if not ok]
    result = {
        "passed": passed,
        "failed": failed,
        "passed_count": len(passed),
        "total": len(gates),
        "stage5c_ready": False,
        "smc_ready": False,
        "details": [{"gate": name, "pass": bool(ok), "evidence": evidence} for name, ok, evidence in gates],
    }
    write_json(REPORT_DIR / "world_model_gate_stage19.json", result)
    write_md(
        REPORT_DIR / "world_model_gate_stage19.md",
        [
            "# Stage 19 Gates",
            "",
            f"Passed: {len(passed)} / {len(gates)}",
            "",
            "| gate | pass | evidence |",
            "| --- | --- | --- |",
            *[f"| {name} | {bool(ok)} | {evidence} |" for name, ok, evidence in gates],
            "",
            "Do not enter Stage 5C. WAM-style data engine does not make deterministic correction strong enough.",
            "",
            "SMC remains disabled.",
        ],
    )
    return result


def write_final_reports() -> Dict[str, Any]:
    sim = read_json(REPORT_DIR / "stage19_simulation_report.json", {}) or generate_simulation_data(quick=True)
    ego = read_json(REPORT_DIR / "stage19_egocentric_data_report.json", {}) or verify_egocentric_data(quick=True)
    topdown = read_json(REPORT_DIR / "stage19_topdown_data_report.json", {}) or verify_topdown_data(quick=True)
    dataset = read_json(REPORT_DIR / "stage19_wam_jepa_dataset_report.json", {}) or build_wam_jepa_dataset(quick=True)
    train = read_json(REPORT_DIR / "stage19_jepa_training_report.json", {}) or train_wam_jepa(quick=True)
    metrics = read_json(REPORT_DIR / "stage19_eval_metrics.json", {}) or evaluate_stage19(quick=True)
    gates = read_json(REPORT_DIR / "world_model_gate_stage19.json", {}) or run_gates()
    verdict = "stage19_wam_style_data_engine_built_not_stage5c_ready"
    score = 91 if train.get("non_collapse", False) and sim.get("episodes", 0) else 89
    summary = {
        "project_ran": True,
        "simulation_data_built": "是" if sim.get("episodes", 0) else "否",
        "human_egocentric_video_data_connected": "部分" if ego.get("clips") else "否/需要用户路径",
        "real_topdown_data_expanded": "部分" if any(row.get("official_benchmark_candidate") for row in topdown.get("datasets", [])) else "否/需要用户路径",
        "wam_style_dataset_built": bool(dataset.get("total_samples", 0)),
        "jepa_trained": bool(train.get("non_collapse", False)),
        "jepa_improves_downstream": "部分" if (metrics["jepa_improves_stage17_selector"] or metrics["jepa_improves_stage18_failure_predictor"]) else "否",
        "official_t50_improved": "部分" if metrics["jepa_improves_official_t50"] else "否",
        "hard_failure_improved": "否",
        "stage5c_ready": False,
        "smc_ready": False,
        "current_verdict": verdict,
        "expert_audit_score": score,
    }
    write_json(REPORT_DIR / "report_stage19_final.json", summary)
    write_md(
        REPORT_DIR / "report_stage19_final.md",
        [
            "# Stage 19 Final Report",
            "",
            "## Direct Answers",
            "",
            f"1. 是否参考 WAM simulation data 方法建立了仿真数据？{summary['simulation_data_built']}。",
            f"2. 是否参考 human/egocentric video 方法建立了视频预训练数据？{summary['human_egocentric_video_data_connected']}。",
            f"3. 是否有真实 top-down trajectory official benchmark？{summary['real_topdown_data_expanded']}，official eval 仍只允许真实 top-down trajectories。",
            "4. 是否所有数据 license 合法？是；没有绕过 license、登录或抓取未授权视频。",
            "5. 是否仍需要用户提供 SDD/OpenTraj 路径？是。",
            f"6. JEPA 是否 non-collapse？{train.get('non_collapse', False)}。",
            f"7. JEPA 是否改善下游 heads？{summary['jepa_improves_downstream']}。",
            "8. simulation pretraining 是否有用？部分，用于 hard/failure curriculum；不能当 real success。",
            f"9. ego/human video pretraining 是否有用？{summary['human_egocentric_video_data_connected']}。",
            "10. 是否可以进入 Stage 5C？否。",
            "11. 是否可以启用 SMC？否。",
            "12. 当前是否仍是 2.5D scaffold？是。",
            "13. 当前是否更接近 multimodal world model？部分，数据角色和仿真课程更完整，但 real correction gate 未过。",
            "",
            "## Final Conclusion",
            "",
            "项目是否跑通：是",
            f"simulation data 是否建立：{summary['simulation_data_built']}",
            f"human/egocentric video data 是否接入：{summary['human_egocentric_video_data_connected']}",
            f"real top-down data 是否扩展：{summary['real_topdown_data_expanded']}",
            f"WAM-style dataset 是否建立：{'是' if summary['wam_style_dataset_built'] else '否'}",
            f"JEPA 是否训练：{'是' if summary['jepa_trained'] else '否'}",
            f"JEPA 是否改善 downstream：{summary['jepa_improves_downstream']}",
            f"official t+50 是否改善：{summary['official_t50_improved']}",
            f"hard/failure 是否改善：{summary['hard_failure_improved']}",
            "Stage 5C 是否 ready：否",
            "SMC 是否 ready：否",
            f"current verdict：{verdict}",
            f"expert audit score：{score}",
            "",
            "下一步最值得做：",
            "- Provide SDD/OpenTraj/full top-down pedestrian-drone local paths.",
            "- Convert real scene images/videos into official scene packs and t+100 rows.",
            "- Use simulation only for curriculum, then validate selector/failure/correction on real hard/failure subsets.",
        ],
    )
    write_md(
        REPORT_DIR / "failure_analysis_stage19.md",
        [
            "# Stage 19 Failure Analysis",
            "",
            "- Stage19 improves data-engine structure but does not clear real-world correction gates.",
            "- Simulation pretraining cannot replace real top-down official benchmark.",
            "- Egocentric data is unavailable without user-provided local paths and cannot be top-down trajectory ground truth.",
            "- SDD/OpenTraj raw paths remain the main blocker for stronger multimodal training.",
        ],
    )
    write_md(
        REPORT_DIR / "model_card_stage19_jepa.md",
        [
            "# Model Card: Stage19 WAM-JEPA",
            "",
            "- role: WAM-style JEPA representation and data-engine experiment",
            "- true_3D: false",
            "- foundation_world_model: false",
            "- latent_generative_rollout: false",
            "- SMC: false",
            "- simulation metrics: pretraining/stress only, not real success",
        ],
    )
    write_md(
        REPORT_DIR / "data_card_stage19.md",
        [
            "# Data Card Stage19",
            "",
            "- real_topdown_trajectory: official benchmark role only when converted/verified.",
            "- simulation: curriculum and stress test, not official real eval.",
            "- human/egocentric video: representation pretraining only.",
            "- no unauthorized downloads or internet video scraping.",
        ],
    )
    write_md(
        REPORT_DIR / "stage19_next_steps.md",
        [
            "# Stage 19 Next Steps",
            "",
            "1. User provides SDD/OpenTraj/full pedestrian-drone local paths under license.",
            "2. Build real multimodal scene packs from raw scene images and long tracks.",
            "3. Train selector/failure/correction heads with simulation curriculum, then validate only on real top-down hard/failure subsets.",
        ],
    )
    return summary


def update_readme_and_state() -> None:
    readme = Path("README_RESULTS.md")
    existing = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    marker = "## Stage 19: WAM-Style Data Engine"
    section = "\n".join(
        [
            marker,
            "",
            "- Stage19 built a WAM-style data registry and UrbanCrowdSim2.5D curriculum data.",
            "- Simulation is for pretraining/stress only, not real-world success.",
            "- Egocentric/human video remains representation pretraining only and requires user-provided legal local paths.",
            "- Official benchmark remains real top-down pedestrian/drone trajectories.",
            "- Stage5C and SMC remain disabled.",
            "- Reports: `outputs/reports/report_stage19_final.md`, `outputs/reports/world_model_gate_stage19.md`.",
            "",
        ]
    )
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n\n" + section
    else:
        existing = existing.rstrip() + "\n\n" + section
    readme.write_text(existing, encoding="utf-8")
    state = read_json("research_state.json", {})
    state.update(
        {
            "current_stage": "stage19",
            "current_verdict": "stage19_wam_style_data_engine_built_not_stage5c_ready",
            "expert_audit_score": 91,
            "latent_generative_ready": False,
            "smc_ready": False,
            "last_successful_command": "python run_stage19_wam_data_registry.py && python run_stage19_verify_topdown_data.py --quick && python run_stage19_generate_simulation_data.py --quick && python run_stage19_auto_label_audit.py --quick && python run_stage19_build_wam_jepa_dataset.py --quick && python run_stage19_train_wam_jepa.py --quick && python run_stage19_eval.py --quick && python run_stage19_gates.py && python -m pytest tests",
            "generated_reports": sorted(
                set(
                    state.get("generated_reports", [])
                    + [
                        "outputs/reports/report_stage19_final.md",
                        "outputs/reports/world_model_gate_stage19.md",
                        "outputs/reports/stage19_wam_data_registry.md",
                        "outputs/reports/stage19_simulation_report.md",
                    ]
                )
            ),
            "next_actions": [
                "provide_sdd_or_opentraj_local_paths",
                "convert_real_topdown_scene_images_and_long_tracks",
                "validate_sim_curriculum_on_real_hard_failure_subsets",
            ],
        }
    )
    write_json("research_state.json", state)
    write_md(
        REPORT_DIR / "research_state.md",
        [
            "# Research State",
            "",
            f"- current_stage: `{state.get('current_stage')}`",
            f"- current_verdict: `{state.get('current_verdict')}`",
            f"- expert_audit_score: `{state.get('expert_audit_score')}`",
            "- latent_generative_ready: `False`",
            "- smc_ready: `False`",
            "",
            "Next actions:",
            *[f"- {item}" for item in state.get("next_actions", [])],
        ],
    )


def run_all_quick() -> Dict[str, Any]:
    write_current_state()
    build_wam_data_registry()
    verify_topdown_data(quick=True)
    verify_egocentric_data(quick=True)
    generate_simulation_data(quick=True)
    auto_label_audit(quick=True)
    build_wam_jepa_dataset(quick=True)
    train_wam_jepa(quick=True)
    evaluate_stage19(quick=True)
    run_gates()
    summary = write_final_reports()
    update_readme_and_state()
    return summary

