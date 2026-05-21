from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.scene.stage10_scene_pack_builder import load_stage10_scene_pack
from src.stage10_common import REPORT_DIR, ensure_dir, write_json, write_markdown_table


OUT_DIR = Path("data/stage10_multiagent_episodes")


def build_stage10_multiagent_episodes(src_root: str | Path = "data/stage8p5_per_agent_episodes") -> Dict:
    summaries = []
    for dataset_dir in sorted(Path(src_root).iterdir()) if Path(src_root).exists() else []:
        if dataset_dir.is_dir():
            summaries.append(build_dataset(dataset_dir.name, dataset_dir))
    payload = {"stage": "10", "datasets": summaries}
    write_episode_report(payload)
    return payload


def build_dataset(dataset: str, src_dir: Path) -> Dict:
    out_dir = ensure_dir(OUT_DIR / dataset)
    for stale in out_dir.glob("episode_*.npz"):
        stale.unlink()
    metas: List[Dict] = []
    for src in sorted(src_dir.glob("episode_*.npz")):
        data = np.load(src, allow_pickle=True)
        states = data["states"].astype(np.float32)
        mask = data["agent_mask"].astype(bool)
        meta = json.loads(str(data["meta"].item()))
        past = int(meta.get("past_horizon", 10))
        future = int(meta.get("future_horizon", states.shape[0] - past))
        baseline = constant_velocity_baseline(states, mask, past, future)
        fde = baseline_fde(baseline, states[past : past + future, :, 0:2], mask[past : past + future])
        pack = load_stage10_scene_pack(dataset, str(meta["scene_id"])) or {}
        new_meta = dict(meta)
        new_meta.update(
            {
                "stage": "10",
                "annotation_quality": pack.get("annotation_quality", meta.get("annotation_quality", "unknown")),
                "human_confirmed_scene": bool(pack.get("human_confirmed", False)),
                "scene_pack_available": bool(pack),
                "hard_label": bool(meta.get("hard_interaction") or min_pairwise_distance(states[:past], mask[:past]) < 5.0),
                "baseline_failure_label": bool(fde > baseline_failure_threshold(dataset, future)),
                "strongest_causal_baseline_name": "constant_velocity_causal_fd_proxy",
                "baseline_fde_proxy": float(fde),
                "candidate_goals_train_only": True,
                "future_endpoint_used_as_input": False,
                "central_velocity_used": False,
            }
        )
        out = out_dir / src.name
        np.savez_compressed(
            out,
            states=states,
            agent_mask=mask,
            agent_ids=data["agent_ids"],
            per_agent_goal_labels=data["per_agent_goal_labels"],
            neighbor_graph=data["neighbor_graph"],
            strongest_causal_baseline=baseline.astype(np.float32),
            scene_features=json.dumps({"annotation_quality": new_meta["annotation_quality"], "scene_pack_available": bool(pack)}),
            goal_candidates=data["goal_candidates"],
            meta=json.dumps(new_meta),
        )
        metas.append(new_meta)
    summary = summarize(dataset, metas)
    write_json(out_dir / "episode_summary.json", summary)
    return summary


def constant_velocity_baseline(states: np.ndarray, mask: np.ndarray, past: int, future: int) -> np.ndarray:
    last = states[past - 1, :, 0:2]
    vel = states[past - 1, :, 2:4]
    steps = np.arange(1, future + 1, dtype=np.float32)[:, None, None]
    return last[None, :, :] + steps * vel[None, :, :]


def baseline_fde(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    if pred.size == 0 or truth.size == 0:
        return 0.0
    valid = mask[-1]
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.linalg.norm(pred[-1, valid] - truth[-1, valid], axis=1)))


def baseline_failure_threshold(dataset: str, future: int) -> float:
    if dataset in {"trajnet", "eth_ucy"}:
        return 5.0 if future <= 10 else 10.0
    return 3.0


def min_pairwise_distance(states: np.ndarray, masks: np.ndarray) -> float:
    vals = []
    for frame, valid in zip(states, masks):
        pos = frame[valid, 0:2]
        if len(pos) < 2:
            continue
        d = np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=2)
        d[d == 0] = np.inf
        vals.append(float(np.min(d)))
    return min(vals) if vals else 999.0


def summarize(dataset: str, metas: List[Dict]) -> Dict:
    counts = [int(m.get("agent_count", 0)) for m in metas]
    return {
        "dataset_name": dataset,
        "total_episodes": len(metas),
        "episodes_ge2_agents": sum(c >= 2 for c in counts),
        "episodes_ge5_agents": sum(c >= 5 for c in counts),
        "episodes_ge10_agents": sum(c >= 10 for c in counts),
        "mean_agents_per_episode": float(np.mean(counts)) if counts else 0.0,
        "hard_episodes": sum(bool(m.get("hard_label")) for m in metas),
        "baseline_failure_episodes": sum(bool(m.get("baseline_failure_label")) for m in metas),
        "pedestrian_drone_hard_episodes": sum(bool(m.get("hard_label")) for m in metas) if dataset in {"trajnet", "eth_ucy", "sdd", "opentraj"} else 0,
        "verified_t50_episodes": sum(bool(m.get("verified_t50")) for m in metas),
        "verified_t100_episodes": sum(bool(m.get("verified_t100")) for m in metas),
        "gold_silver_scene_episodes": sum(m.get("annotation_quality") in {"gold_human", "silver_human_confirmed", "silver_rule_confirmed"} for m in metas),
        "human_confirmed_scene_episodes": sum(bool(m.get("human_confirmed_scene")) for m in metas),
        "inferred_only_episodes": sum(m.get("annotation_quality") == "inferred_only" for m in metas),
        "official_training_episodes": sum(m.get("annotation_quality") in {"gold_human", "silver_human_confirmed"} for m in metas),
        "diagnostic_only_episodes": sum(m.get("annotation_quality") not in {"gold_human", "silver_human_confirmed"} for m in metas),
    }


def write_episode_report(payload: Dict) -> None:
    write_json(REPORT_DIR / "stage10_multiagent_episode_report.json", payload)
    keys = [
        "dataset_name",
        "total_episodes",
        "episodes_ge2_agents",
        "episodes_ge5_agents",
        "episodes_ge10_agents",
        "mean_agents_per_episode",
        "hard_episodes",
        "baseline_failure_episodes",
        "verified_t50_episodes",
        "verified_t100_episodes",
        "gold_silver_scene_episodes",
        "human_confirmed_scene_episodes",
        "official_training_episodes",
        "diagnostic_only_episodes",
    ]
    write_markdown_table(REPORT_DIR / "stage10_multiagent_episode_report.md", "Stage 10 Multi-Agent Episode Report", [{k: r.get(k) for k in keys} for r in payload["datasets"]])


def main() -> None:
    payload = build_stage10_multiagent_episodes()
    print(json.dumps({"datasets": len(payload["datasets"]), "episodes": sum(r["total_episodes"] for r in payload["datasets"])}, indent=2))


if __name__ == "__main__":
    main()
