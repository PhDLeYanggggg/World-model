from __future__ import annotations

import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.annotation.stage11_aerialmpt_visual_annotator import ZIP_PATH, edge_goal_regions, parse_mot_rows, scene_entries


OUT_ROOT = Path("data/stage11_multiagent_episodes/aerialmpt")
REPORT_DIR = Path("outputs/reports")


VAL_SCENES = {"oac", "pasing1R"}


def split_for_scene(source_split: str, scene_id: str) -> str:
    if source_split == "test":
        return "test"
    if scene_id in VAL_SCENES:
        return "val"
    return "train"


def build_states(mot: List[Dict], frames: List[int], agents: List[str], width: int = 1, height: int = 1) -> tuple[np.ndarray, np.ndarray]:
    idx = {agent: i for i, agent in enumerate(agents)}
    fidx = {frame: i for i, frame in enumerate(frames)}
    states = np.zeros((len(frames), len(agents), 9), dtype=np.float32)
    mask = np.zeros((len(frames), len(agents)), dtype=bool)
    by = {(r["frame"], r["agent_id"]): r for r in mot}
    for frame in frames:
        for agent in agents:
            row = by.get((frame, agent))
            if row is None:
                continue
            t = fidx[frame]
            a = idx[agent]
            states[t, a, 0] = row["cx"]
            states[t, a, 1] = row["cy"]
            states[t, a, 8] = max(row["w"], row["h"]) / 2.0
            mask[t, a] = True
    for a in range(len(agents)):
        valid = np.where(mask[:, a])[0]
        for k, t in enumerate(valid):
            if k == 0:
                continue
            p = valid[k - 1]
            dt = max(float(frames[t] - frames[p]), 1.0)
            states[t, a, 2:4] = (states[t, a, 0:2] - states[p, a, 0:2]) / dt
            states[t, a, 4:6] = states[t, a, 2:4] - states[p, a, 2:4]
            states[t, a, 6] = math.atan2(float(states[t, a, 3]), float(states[t, a, 2]))
            states[t, a, 7] = float(np.linalg.norm(states[t, a, 2:4]))
        if len(valid):
            first = valid[0]
            if len(valid) > 1:
                states[first, a, 2:8] = states[valid[1], a, 2:8]
    return states, mask


def select_agents(mot: List[Dict], window_frames: List[int], max_agents: int = 64) -> List[str]:
    counts = defaultdict(int)
    frame_set = set(window_frames)
    for row in mot:
        if row["frame"] in frame_set:
            counts[row["agent_id"]] += 1
    return [agent for agent, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_agents]]


def neighbor_graph(states: np.ndarray, mask: np.ndarray, k: int = 5) -> np.ndarray:
    last_idx = max(0, states.shape[0] // 2 - 1)
    pos = states[last_idx, :, 0:2]
    valid = mask[last_idx]
    graph = np.full((states.shape[1], k), -1, dtype=np.int32)
    for i in range(states.shape[1]):
        if not valid[i]:
            continue
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[~valid] = np.inf
        d[i] = np.inf
        nn = np.argsort(d)[:k]
        graph[i, : len(nn)] = nn
    return graph


def build_aerialmpt_episodes(zip_path: Path = ZIP_PATH, past: int = 10, future: int = 10) -> Dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    episode_id = 0
    with zipfile.ZipFile(zip_path) as zf:
        for entry in scene_entries(zip_path):
            scene = entry["scene_id"]
            mot = parse_mot_rows(zf.read(entry["gts"]).decode("utf-8", errors="ignore"))
            frames = sorted({r["frame"] for r in mot})
            if len(frames) < past + 5:
                continue
            image_name = entry["frames"][0]
            from PIL import Image

            with zf.open(image_name) as fh:
                im = Image.open(fh)
                width, height = im.size
            target_future = min(future, len(frames) - past)
            if target_future < 5:
                continue
            starts = list(range(0, len(frames) - past - target_future + 1, max(1, target_future // 2)))
            split = split_for_scene(entry["split"], scene)
            goals = edge_goal_regions(width, height)
            for start in starts[:8]:
                window = frames[start : start + past + target_future]
                agents = select_agents(mot, window)
                if len(agents) < 2:
                    continue
                states, mask = build_states(mot, window, agents, width, height)
                meta = {
                    "episode_id": episode_id,
                    "dataset_name": "aerialmpt",
                    "scene_id": scene,
                    "split": split,
                    "past_horizon": past,
                    "future_horizon": target_future,
                    "official_eval_horizons": [h for h in [1, 5, 10] if h <= target_future],
                    "verified_t10": target_future >= 10,
                    "verified_t25": False,
                    "verified_t50": False,
                    "verified_t100": False,
                    "agent_ids": agents,
                    "agent_count": len(agents),
                    "dropped_agents": 0,
                    "coordinate_unit": "pixel",
                    "dt_s": 1.0,
                    "annotation_quality": "ai_visual_silver",
                    "candidate_goal_source": "image_boundary_prior_not_future_endpoint",
                    "test_endpoints_used_for_goals": False,
                    "candidate_goals_train_only": True,
                    "future_endpoint_used_as_input": False,
                    "central_velocity_used": False,
                    "frame_start": window[0],
                    "frame_end": window[-1],
                    "hard_interaction": len(agents) >= 10,
                    "baseline_failure_proxy": len(agents) >= 20,
                    "stage": "11",
                    "human_confirmed_scene": False,
                    "scene_pack_available": True,
                    "strongest_causal_baseline_name": "constant_velocity_causal_fd_proxy",
                }
                out_dir = OUT_ROOT
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_dir / f"episode_{episode_id:05d}.npz",
                    states=states,
                    agent_mask=mask,
                    agent_ids=np.asarray(agents, dtype=object),
                    per_agent_goal_labels=np.full((len(agents),), -1, dtype=np.int32),
                    neighbor_graph=neighbor_graph(states, mask),
                    strongest_causal_baseline=np.asarray([], dtype=np.float32),
                    scene_features=np.asarray([], dtype=np.float32),
                    goal_candidates=np.asarray(json.dumps(goals), dtype=object),
                    meta=np.asarray(json.dumps(meta), dtype=object),
                )
                rows.append({"episode_id": episode_id, "scene_id": scene, "split": split, "agents": len(agents), "future_horizon": target_future})
                episode_id += 1
    report = {
        "stage": "11",
        "dataset": "aerialmpt",
        "episodes": len(rows),
        "train": sum(r["split"] == "train" for r in rows),
        "val": sum(r["split"] == "val" for r in rows),
        "test": sum(r["split"] == "test" for r in rows),
        "verified_t10": sum(r["future_horizon"] >= 10 for r in rows),
        "verified_t50": 0,
        "verified_t100": 0,
        "coordinate_unit": "pixel",
        "annotation_quality": "ai_visual_silver",
        "records": rows,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage11_aerialmpt_episode_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    return report


def write_markdown(report: Dict) -> None:
    lines = [
        "# Stage 11 AerialMPT Episode Report",
        "",
        f"Episodes: `{report['episodes']}`",
        f"Train/val/test: `{report['train']}/{report['val']}/{report['test']}`",
        f"Verified t+10 episodes: `{report['verified_t10']}`",
        f"Verified t+50/t+100 episodes: `0/0`",
        f"Coordinate unit: `{report['coordinate_unit']}`",
        "",
        "These are pixel-space short-horizon pedestrian/drone episodes; they do not establish pedestrian t+50/t+100.",
    ]
    (REPORT_DIR / "stage11_aerialmpt_episode_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = build_aerialmpt_episodes()
    print(json.dumps({"episodes": report["episodes"], "report": "outputs/reports/stage11_aerialmpt_episode_report.md"}, indent=2))


if __name__ == "__main__":
    main()
