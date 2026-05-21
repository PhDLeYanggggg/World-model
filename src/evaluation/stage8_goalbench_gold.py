from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.scene.goal_region_builder import assign_goal
from src.scene.stage8_scene_gold_builder import load_scene_gold_pack
from src.scene.scene_sdf import goal_features


OUT_DIR = Path("data/stage8_goalbench_gold")
REPORT_DIR = Path("outputs/reports")


def load_multiagent_episodes(dataset: str, split: str = "all") -> List[Dict]:
    episodes = []
    for path in sorted((Path("data/stage8_multiagent_episodes") / dataset).glob("episode_*.npz")):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"].item()))
        if split != "all" and meta.get("split") != split:
            continue
        episodes.append({"states": data["states"].astype(np.float32), "agent_mask": data["agent_mask"].astype(bool), "meta": meta, "path": str(path)})
    return episodes


def available_stage8_datasets() -> List[str]:
    root = Path("data/stage8_multiagent_episodes")
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def build_goalbench_gold(datasets: List[str] | None = None) -> Dict:
    datasets = datasets or available_stage8_datasets()
    records = []
    for dataset in datasets:
        for ep in load_multiagent_episodes(dataset):
            meta = ep["meta"]
            pack = load_scene_gold_pack(dataset, str(meta.get("scene_id", dataset)))
            if not pack:
                continue
            records.append(record_from_episode(dataset, ep, pack))
    return summarize(records)


def record_from_episode(dataset: str, ep: Dict, pack: Dict) -> Dict:
    states = ep["states"]
    meta = ep["meta"]
    past = int(meta["past_horizon"])
    primary = 0
    last = states[past - 1, primary]
    endpoint = states[-1, primary, 0:2]
    goals = pack.get("goal_regions", [])
    label = assign_goal(endpoint, [{"center": g["center"]} for g in goals])
    gf = goal_features(last[0:2], last[2:4], [{"center": g["center"], "support_fraction": 1.0 / max(len(goals), 1), "radius": g.get("radius", 1.0)} for g in goals])
    return {
        "dataset": dataset,
        "scene_id": meta.get("scene_id"),
        "episode_id": int(meta["episode_id"]),
        "split": meta.get("split"),
        "horizon": int(meta["future_horizon"]),
        "agent_count": int(meta["agent_count"]),
        "goal_quality": pack.get("annotation_quality", "inferred_only"),
        "candidate_goal_count": len(goals),
        "true_endpoint_assignment": int(label),
        "future_endpoint_label_only": True,
        "distances_to_goals": gf[:, 0].tolist() if len(gf) else [],
        "angle_to_goals": gf[:, 1].tolist() if len(gf) else [],
        "route_distance_available": bool(pack.get("route_graph", {}).get("edges")),
        "ambiguity_score": 0.0,
        "goal_prediction_meaningful": len(goals) >= 2,
    }


def summarize(records: List[Dict]) -> Dict:
    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)
    datasets = {}
    for dataset, rows in by_dataset.items():
        labels = [r["true_endpoint_assignment"] for r in rows if r["true_endpoint_assignment"] >= 0]
        counts = Counter(labels)
        total = sum(counts.values())
        probs = np.asarray([v / max(total, 1) for v in counts.values()], dtype=float)
        entropy = float(-(probs * np.log(np.maximum(probs, 1e-9))).sum()) if len(probs) else 0.0
        top1 = counts.most_common(1)[0][1] / total if total else 0.0
        top3 = sum(v for _, v in counts.most_common(3)) / total if total else 0.0
        datasets[dataset] = {
            "episodes": len(rows),
            "goal_quality_counts": dict(Counter(r["goal_quality"] for r in rows)),
            "candidate_goal_count_mean": float(np.mean([r["candidate_goal_count"] for r in rows])) if rows else 0.0,
            "top1_majority_baseline": float(top1),
            "top3_majority_baseline": float(top3),
            "goal_entropy": entropy,
            "goal_ambiguity_score": float(entropy / np.log(max(len(counts), 2))) if counts else 0.0,
            "majority_saturation": bool(top3 >= 0.95),
            "whether_goal_prediction_meaningful": bool(len(counts) >= 2 and entropy > 0.2),
            "horizons": sorted({r["horizon"] for r in rows}),
        }
    return {"stage": "8", "records": records, "datasets": datasets, "total_records": len(records)}


def write_outputs(payload: Dict) -> Dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "goalbench_gold_records.json").write_text(json.dumps(payload["records"], indent=2), encoding="utf-8")
    summary = {k: v for k, v in payload.items() if k != "records"}
    (OUT_DIR / "goalbench_gold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "stage8_goalbench_gold_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = [{"dataset": k, **v} for k, v in payload["datasets"].items()]
    (REPORT_DIR / "stage8_goalbench_gold_report.md").write_text("# Stage 8 GoalBench-Gold Report\n\n" + markdown_table(rows), encoding="utf-8")
    return payload


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"

