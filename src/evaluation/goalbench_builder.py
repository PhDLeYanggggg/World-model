from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.evaluation.baseline_benchmark_stage5b import available_datasets, load_dataset_episodes
from src.scene.goal_region_builder import assign_goal
from src.scene.scene_pack_builder import build_scene_packs, load_scene_pack, write_report
from src.scene.scene_sdf import goal_features, walkability_sdf_sample


OUT_DIR = Path("data/goalbench")
REPORT_DIR = Path("outputs/reports")


def ensure_scene_packs() -> None:
    if not list(Path("data/scene_packs").glob("*/*/scene_pack.json")):
        write_report(build_scene_packs())


def goalbench_record(dataset: str, ep: Dict, pack: Dict) -> Dict:
    states = ep["states"]
    meta = ep["meta"]
    past = int(meta.get("past_horizon", 10))
    last = states[past - 1, 0]
    endpoint = states[-1, 0, 0:2]
    goals = pack.get("candidate_goal_regions", [])
    label = assign_goal(endpoint, goals)
    gf = goal_features(last[0:2], last[2:4], goals)
    sdf = walkability_sdf_sample(last[0:2], pack["boundary_summary"])
    distances = gf[:, 0].tolist() if len(gf) else []
    heading_cos = gf[:, 1].tolist() if len(gf) else []
    return {
        "dataset": dataset,
        "scene_id": meta.get("scene_id", dataset),
        "episode_id": int(meta.get("episode_id", -1)),
        "split": meta.get("split"),
        "domain": "traffic" if dataset.startswith("tgsim") else "pedestrian",
        "horizon": int(states.shape[0] - past),
        "coordinate_unit": meta.get("coordinate_unit", "unknown"),
        "candidate_goal_count": len(goals),
        "true_endpoint_cluster_label": label,
        "goal_source": "inferred_scene_goal",
        "future_endpoint_is_label_only": True,
        "last_position": [float(last[0]), float(last[1])],
        "last_velocity": [float(last[2]), float(last[3])],
        "distances_to_goals": distances,
        "heading_cos_to_goals": heading_cos,
        "goal_priors": [float(g.get("support_fraction", 0.0)) for g in goals],
        "boundary_distance": sdf["boundary_distance"],
        "inside_inferred_walkable_area": sdf["inside_inferred_walkable_area"],
    }


def build_goalbench(datasets: List[str] | None = None) -> Dict:
    ensure_scene_packs()
    datasets = datasets or available_datasets()
    records = []
    for dataset in datasets:
        for ep in load_dataset_episodes(dataset, split="all"):
            meta = ep["meta"]
            pack = load_scene_pack(dataset, str(meta.get("scene_id", dataset)))
            if not pack:
                continue
            records.append(goalbench_record(dataset, ep, pack))
    return summarize(records)


def summarize(records: List[Dict]) -> Dict:
    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)
    dataset_summaries = {}
    for dataset, rows in by_dataset.items():
        labels = [r["true_endpoint_cluster_label"] for r in rows if r["true_endpoint_cluster_label"] >= 0]
        counts = Counter(labels)
        total = sum(counts.values())
        probs = np.asarray([v / max(total, 1) for v in counts.values()], dtype=float)
        entropy = float(-(probs * np.log(np.maximum(probs, 1e-9))).sum()) if len(probs) else 0.0
        majority = counts.most_common(1)[0][0] if counts else -1
        top3 = {k for k, _ in counts.most_common(3)}
        dataset_summaries[dataset] = {
            "episodes": len(rows),
            "candidate_goal_count_mean": float(np.mean([r["candidate_goal_count"] for r in rows])) if rows else 0.0,
            "goal_label_entropy": entropy,
            "majority_goal_label": majority,
            "top1_majority_baseline": float(counts.get(majority, 0) / max(total, 1)) if total else 0.0,
            "top3_majority_baseline": float(sum(counts.get(k, 0) for k in top3) / max(total, 1)) if total else 0.0,
            "horizons": sorted({r["horizon"] for r in rows}),
            "goal_ambiguity_score": float(entropy / np.log(max(len(counts), 2))) if counts else 0.0,
            "goal_prediction_meaningful": bool(len(counts) >= 2 and entropy > 0.2),
        }
    return {"stage": "7", "records": records, "datasets": dataset_summaries, "total_records": len(records)}


def write_outputs(payload: Dict) -> Dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "goalbench_records.json").write_text(json.dumps(payload["records"], indent=2), encoding="utf-8")
    summary = {k: v for k, v in payload.items() if k != "records"}
    (OUT_DIR / "goalbench_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "goalbench_summary_stage7.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = [{"dataset": k, **v} for k, v in payload["datasets"].items()]
    (REPORT_DIR / "goalbench_summary_stage7.md").write_text("# Stage 7 GoalBench Summary\n\n" + markdown_table(rows), encoding="utf-8")
    return payload


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"

