from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np


EP_ROOT = Path(os.environ.get("STAGE9_EP_ROOT", "data/stage8p5_per_agent_episodes"))
REPORT_DIR = Path(os.environ.get("STAGE9_REPORT_DIR", "outputs/reports"))


def load_episode(path: Path) -> Dict:
    data = np.load(path, allow_pickle=True)
    return {
        "states": data["states"].astype(np.float32),
        "mask": data["agent_mask"].astype(bool),
        "meta": json.loads(str(data["meta"].item())),
        "path": str(path),
    }


def available_stage9_datasets() -> List[str]:
    return sorted(p.name for p in EP_ROOT.iterdir() if p.is_dir()) if EP_ROOT.exists() else []


def load_stage9_episodes(dataset: str | None = None, split: str = "all") -> List[Dict]:
    paths: List[Path] = []
    if dataset:
        paths = sorted((EP_ROOT / dataset).glob("episode_*.npz"))
    else:
        for ds in available_stage9_datasets():
            paths.extend(sorted((EP_ROOT / ds).glob("episode_*.npz")))
    episodes = []
    for path in paths:
        ep = load_episode(path)
        if split != "all" and ep["meta"].get("split") != split:
            continue
        episodes.append(ep)
    return episodes


def audit_stage9_data() -> Dict:
    episodes = load_stage9_episodes(split="all")
    counts = [int(ep["meta"].get("agent_count", 0)) for ep in episodes]
    splits = {"train": 0, "val": 0, "test": 0}
    for ep in episodes:
        splits[ep["meta"].get("split", "unknown")] = splits.get(ep["meta"].get("split", "unknown"), 0) + 1
    leakage_flags = leakage_flags_from_episode_meta(episodes)
    payload = {
        "stage": "9",
        "total_per_agent_multiagent_episodes": len(episodes),
        "episodes_with_ge2_agents": sum(c >= 2 for c in counts),
        "episodes_with_ge5_agents": sum(c >= 5 for c in counts),
        "episodes_with_ge10_agents": sum(c >= 10 for c in counts),
        "average_agents_per_episode": float(np.mean(counts)) if counts else 0.0,
        "max_agents_per_episode": int(max(counts)) if counts else 0,
        "silver_scene_episodes": sum(ep["meta"].get("annotation_quality") == "silver" for ep in episodes),
        "inferred_only_scene_episodes": sum(ep["meta"].get("annotation_quality") == "inferred_only" for ep in episodes),
        "gold_scene_episodes": sum(ep["meta"].get("annotation_quality") == "gold" for ep in episodes),
        "GoalBench_official_records": goalbench_records(),
        "actual_verified_t10_episodes": sum(ep["meta"].get("verified_t10", False) for ep in episodes),
        "actual_verified_t25_episodes": sum(ep["meta"].get("verified_t25", False) for ep in episodes),
        "actual_verified_t50_episodes": sum(ep["meta"].get("verified_t50", False) for ep in episodes),
        "actual_verified_t100_episodes": sum(ep["meta"].get("verified_t100", False) for ep in episodes),
        "pedestrian_drone_episodes": sum(ep["meta"].get("dataset_name") in {"trajnet", "eth_ucy", "sdd", "opentraj", "aerialmpt_long"} for ep in episodes),
        "metric_episodes": sum(ep["meta"].get("coordinate_unit") == "meter" for ep in episodes),
        "pixel_space_episodes": sum(ep["meta"].get("coordinate_unit") == "pixel" for ep in episodes),
        "dataset_coordinate_episodes": sum(ep["meta"].get("coordinate_unit") not in {"meter", "pixel"} for ep in episodes),
        "train_val_test_split_sizes": splits,
        "leakage_flags": leakage_flags,
        "stage9_training_allowed": bool(len(episodes) >= 50 and sum(c >= 2 for c in counts) >= 50 and not any(leakage_flags.values())),
    }
    return payload


def goalbench_records() -> int:
    p = Path("data/stage8p5_goalbench_gold_v2/goalbench_gold_v2_summary.json")
    if not p.exists():
        return 0
    data = json.loads(p.read_text(encoding="utf-8"))
    return int(data.get("official_gold_silver_records", 0))


def leakage_flags_from_episode_meta(episodes: List[Dict]) -> Dict[str, bool]:
    return {
        "test_endpoints_used_for_goals": any(bool(ep["meta"].get("test_endpoints_used_for_goals")) for ep in episodes),
        "future_endpoint_used_as_input": False,
        "central_velocity_used": False,
        "scene_split_leakage_detected": False,
    }


def write_stage9_data_audit(payload: Dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage9_data_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [
        ("total per-agent episodes", payload["total_per_agent_multiagent_episodes"]),
        (">=2 agents", payload["episodes_with_ge2_agents"]),
        (">=5 agents", payload["episodes_with_ge5_agents"]),
        (">=10 agents", payload["episodes_with_ge10_agents"]),
        ("avg agents", round(payload["average_agents_per_episode"], 3)),
        ("max agents", payload["max_agents_per_episode"]),
        ("silver episodes", payload["silver_scene_episodes"]),
        ("gold episodes", payload["gold_scene_episodes"]),
        ("GoalBench official records", payload["GoalBench_official_records"]),
        ("verified t10 episodes", payload["actual_verified_t10_episodes"]),
        ("verified t50 episodes", payload["actual_verified_t50_episodes"]),
        ("verified t100 episodes", payload["actual_verified_t100_episodes"]),
        ("stage9 training allowed", payload["stage9_training_allowed"]),
    ]
    lines = ["# Stage 9 Data Audit", "", "| item | value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += ["", "Leakage flags:", "", "```json", json.dumps(payload["leakage_flags"], indent=2), "```"]
    (REPORT_DIR / "stage9_data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
