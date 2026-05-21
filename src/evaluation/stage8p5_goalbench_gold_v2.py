from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.scene.stage8p5_scene_gold_builder import load_stage8p5_scene_pack


OUT_DIR = Path("data/stage8p5_goalbench_gold_v2")
REPORT_DIR = Path("outputs/reports")


def available_episode_datasets(root: str | Path = "data/stage8p5_per_agent_episodes") -> List[str]:
    base = Path(root)
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []


def load_episodes(dataset: str) -> List[Dict]:
    rows = []
    for path in sorted((Path("data/stage8p5_per_agent_episodes") / dataset).glob("episode_*.npz")):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"].item()))
        rows.append(
            {
                "states": data["states"].astype(np.float32),
                "mask": data["agent_mask"].astype(bool),
                "labels": data["per_agent_goal_labels"].astype(int),
                "agent_ids": [str(x) for x in data["agent_ids"].tolist()],
                "meta": meta,
                "path": str(path),
            }
        )
    return rows


def build_goalbench_gold_v2(datasets: List[str] | None = None) -> Dict:
    datasets = datasets or available_episode_datasets()
    records = []
    for dataset in datasets:
        for ep in load_episodes(dataset):
            records.extend(records_from_episode(dataset, ep))
    return summarize(records)


def records_from_episode(dataset: str, ep: Dict) -> List[Dict]:
    meta = ep["meta"]
    pack = load_stage8p5_scene_pack(dataset, str(meta["scene_id"]))
    if not pack or not pack.get("goal_regions"):
        return []
    quality = pack.get("annotation_quality", "inferred_only")
    official = quality in {"gold", "silver"}
    goals = pack["goal_regions"]
    centers = np.asarray([g["center"] for g in goals], dtype=float)
    states = ep["states"]
    past = int(meta["past_horizon"])
    future_end = states[-1, :, 0:2]
    last_past = states[past - 1, :, 0:2]
    rows = []
    for idx, agent_id in enumerate(ep["agent_ids"]):
        if not ep["mask"][past - 1, idx] or not ep["mask"][-1, idx]:
            continue
        label = int(np.argmin(np.linalg.norm(centers - future_end[idx][None, :], axis=1)))
        distances = np.linalg.norm(centers - last_past[idx][None, :], axis=1)
        rows.append(
            {
                "scene_id": meta["scene_id"],
                "dataset": dataset,
                "episode_id": int(meta["episode_id"]),
                "agent_id": agent_id,
                "split": meta["split"],
                "candidate_goals": goals,
                "candidate_goal_count": len(goals),
                "annotation_quality": quality,
                "official_gold_silver": official,
                "goal_label_for_train_eval": label,
                "future_endpoint_label_only": True,
                "distance_to_goal_baseline_label": int(np.argmin(distances)),
                "distances_to_goals": distances.tolist(),
                "route_distance_available": bool(pack.get("route_corridors")),
                "ambiguity_score": ambiguity(distances),
                "hard_failure_label": bool(meta.get("hard_interaction") or meta.get("baseline_failure_proxy")),
                "horizon": int(meta["future_horizon"]),
                "test_endpoints_used_for_candidates": False,
            }
        )
    return rows


def ambiguity(distances: np.ndarray) -> float:
    if len(distances) < 2:
        return 0.0
    sd = np.sort(distances)
    return float(1.0 / (1.0 + max(sd[1] - sd[0], 0.0)))


def summarize(records: List[Dict]) -> Dict:
    official = [r for r in records if r["official_gold_silver"]]
    diag = [r for r in records if not r["official_gold_silver"]]
    return {
        "stage": "8.5",
        "records": records,
        "official_gold_silver_records": len(official),
        "diagnostic_inferred_records": len(diag),
        "official": split_summary(official),
        "diagnostic": split_summary(diag),
    }


def split_summary(rows: List[Dict]) -> Dict:
    if not rows:
        return empty_summary()
    labels = [r["goal_label_for_train_eval"] for r in rows]
    counts = Counter(labels)
    total = len(labels)
    top1 = counts.most_common(1)[0][1] / total
    top3 = sum(v for _, v in counts.most_common(3)) / total
    dist_top1 = np.mean([r["distance_to_goal_baseline_label"] == r["goal_label_for_train_eval"] for r in rows])
    # distance top3 is saturated if fewer than 4 goals; still reported.
    dist_top3 = []
    for r in rows:
        order = np.argsort(np.asarray(r["distances_to_goals"], dtype=float))[:3]
        dist_top3.append(r["goal_label_for_train_eval"] in order)
    probs = np.asarray([v / total for v in counts.values()], dtype=float)
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-9))).sum())
    by_dataset = defaultdict(int)
    for r in rows:
        by_dataset[r["dataset"]] += 1
    return {
        "records": len(rows),
        "candidate_goal_count_mean": float(np.mean([r["candidate_goal_count"] for r in rows])),
        "majority_top1": float(top1),
        "majority_top3": float(top3),
        "distance_baseline_top1": float(dist_top1),
        "distance_baseline_top3": float(np.mean(dist_top3)),
        "goal_entropy": entropy,
        "goal_ambiguity": float(np.mean([r["ambiguity_score"] for r in rows])),
        "whether_top3_saturated": bool(top3 >= 0.95),
        "whether_goal_prediction_meaningful": bool(len(counts) >= 2 and entropy > 0.2),
        "by_dataset": dict(by_dataset),
    }


def empty_summary() -> Dict:
    return {
        "records": 0,
        "candidate_goal_count_mean": 0.0,
        "majority_top1": 0.0,
        "majority_top3": 0.0,
        "distance_baseline_top1": 0.0,
        "distance_baseline_top3": 0.0,
        "goal_entropy": 0.0,
        "goal_ambiguity": 0.0,
        "whether_top3_saturated": False,
        "whether_goal_prediction_meaningful": False,
        "by_dataset": {},
    }


def write_goalbench_v2(payload: Dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "goalbench_gold_v2_records.json").write_text(json.dumps(payload["records"], indent=2), encoding="utf-8")
    summary = {k: v for k, v in payload.items() if k != "records"}
    (OUT_DIR / "goalbench_gold_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "stage8p5_goalbench_gold_v2_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = [{"split": "official", **payload["official"]}, {"split": "diagnostic", **payload["diagnostic"]}]
    (REPORT_DIR / "stage8p5_goalbench_gold_v2_report.md").write_text("# Stage 8.5 GoalBench-Gold v2\n\n" + markdown_table(rows), encoding="utf-8")


def markdown_table(rows: List[Dict]) -> str:
    keys = list(rows[0]) if rows else ["split"]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"
