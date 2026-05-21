from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.scene.stage10_scene_pack_builder import load_stage10_scene_pack
from src.stage10_common import REPORT_DIR, ensure_dir, is_official_annotation_quality, write_json, write_markdown_table


OUT_DIR = Path("data/stage10_goalbench_v3")


def build_goalbench_v3(root: str | Path = "data/stage10_multiagent_episodes") -> Dict:
    records = []
    for path in sorted(Path(root).glob("*/episode_*.npz")):
        records.extend(records_from_episode(path))
    payload = {"stage": "10", "records": records, "summary": summarize(records)}
    write_goalbench(payload)
    return payload


def records_from_episode(path: Path) -> List[Dict]:
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"].item()))
    dataset = meta["dataset_name"]
    pack = load_stage10_scene_pack(dataset, str(meta["scene_id"]))
    if not pack or not pack.get("goal_regions"):
        return []
    goals = pack["goal_regions"]
    centers = np.asarray([g.get("center", [0.0, 0.0]) for g in goals], dtype=float)
    states = data["states"].astype(np.float32)
    mask = data["agent_mask"].astype(bool)
    agent_ids = [str(x) for x in data["agent_ids"].tolist()]
    past = int(meta.get("past_horizon", 10))
    quality = pack.get("annotation_quality", "inferred_only")
    rows = []
    for idx, agent_id in enumerate(agent_ids):
        if not mask[past - 1, idx] or not mask[-1, idx]:
            continue
        last_past = states[past - 1, idx, 0:2]
        future_end = states[-1, idx, 0:2]
        goal_label = int(np.argmin(np.linalg.norm(centers - future_end[None, :], axis=1)))
        distances = np.linalg.norm(centers - last_past[None, :], axis=1)
        rows.append(
            {
                "scene_id": meta["scene_id"],
                "episode_id": meta["episode_id"],
                "dataset_name": dataset,
                "agent_id": agent_id,
                "candidate_goals": goals,
                "candidate_goal_count": len(goals),
                "goal_label": goal_label,
                "annotation_quality": quality,
                "official_record": is_official_annotation_quality(quality),
                "weaker_official_tier": quality == "silver_rule_confirmed",
                "horizon": int(meta.get("future_horizon", 0)),
                "hard_label": bool(meta.get("hard_label")),
                "baseline_failure_label": bool(meta.get("baseline_failure_label")),
                "majority_baseline": None,
                "distance_baseline": int(np.argmin(distances)),
                "route_distance_baseline": None,
                "ambiguity_score": ambiguity(distances),
                "test_endpoints_used_for_candidates": False,
                "future_endpoint_used_as_input": False,
            }
        )
    return rows


def ambiguity(distances: np.ndarray) -> float:
    if len(distances) < 2:
        return 0.0
    ordered = np.sort(distances)
    return float(1.0 / (1.0 + max(ordered[1] - ordered[0], 0.0)))


def summarize(records: List[Dict]) -> Dict:
    official = [r for r in records if r["official_record"]]
    diagnostic = [r for r in records if not r["official_record"]]
    return {
        "records_by_annotation_quality": dict(Counter(r["annotation_quality"] for r in records)),
        "official_records_count": len(official),
        "diagnostic_records_count": len(diagnostic),
        "official": split_summary(official),
        "diagnostic": split_summary(diagnostic),
    }


def split_summary(rows: List[Dict]) -> Dict:
    if not rows:
        return {"records": 0, "top1_majority": 0.0, "top3_majority": 0.0, "distance_baseline": 0.0, "route_baseline": None, "goal_entropy": 0.0, "goal_ambiguity": 0.0, "whether_top3_saturated": False, "whether_goal_prediction_meaningful": False}
    labels = [int(r["goal_label"]) for r in rows]
    counts = Counter(labels)
    total = len(labels)
    top1 = counts.most_common(1)[0][1] / total
    top3 = sum(v for _, v in counts.most_common(3)) / total
    probs = np.asarray([v / total for v in counts.values()], dtype=float)
    by_quality = defaultdict(int)
    for r in rows:
        by_quality[r["annotation_quality"]] += 1
    return {
        "records": total,
        "top1_majority": float(top1),
        "top3_majority": float(top3),
        "distance_baseline": float(np.mean([r["distance_baseline"] == r["goal_label"] for r in rows])),
        "route_baseline": None,
        "goal_entropy": float(-(probs * np.log(np.maximum(probs, 1e-9))).sum()),
        "goal_ambiguity": float(np.mean([r["ambiguity_score"] for r in rows])),
        "whether_top3_saturated": bool(top3 >= 0.95),
        "whether_goal_prediction_meaningful": bool(len(counts) >= 2 and top1 < 0.95),
        "by_quality": dict(by_quality),
    }


def write_goalbench(payload: Dict) -> None:
    ensure_dir(OUT_DIR)
    write_json(OUT_DIR / "goalbench_v3_records.json", payload["records"])
    write_json(OUT_DIR / "goalbench_v3_summary.json", payload["summary"])
    write_json(REPORT_DIR / "stage10_goalbench_v3_report.json", payload["summary"])
    rows = [{"split": "official", **payload["summary"]["official"]}, {"split": "diagnostic", **payload["summary"]["diagnostic"]}]
    write_markdown_table(REPORT_DIR / "stage10_goalbench_v3_report.md", "Stage 10 GoalBench v3", rows)


def main() -> None:
    payload = build_goalbench_v3()
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
