from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.stage10_common import REPORT_DIR, write_json, write_markdown_table


def select_annotation_scenes() -> Dict:
    annotations = json.loads(Path("outputs/reports/stage10_annotation_report.json").read_text(encoding="utf-8")) if Path("outputs/reports/stage10_annotation_report.json").exists() else {"annotations": []}
    episode_report = json.loads(Path("outputs/reports/stage8p5_per_agent_episode_report.json").read_text(encoding="utf-8")) if Path("outputs/reports/stage8p5_per_agent_episode_report.json").exists() else {"datasets": []}
    dataset_episode_counts = {r["dataset_name"]: r for r in episode_report.get("datasets", [])}
    rows: List[Dict] = []
    for ann in annotations.get("annotations", []):
        ds = ann["dataset_name"]
        epi = dataset_episode_counts.get(ds, {})
        score = 0.0
        score += 25 if ds in {"trajnet", "eth_ucy", "sdd", "opentraj"} else 0
        score += min(float(epi.get("mean_agents_per_episode", 0.0)) * 1.5, 25)
        score += min(float(epi.get("hard_interaction_episodes", 0)) / 5.0, 20)
        score += 15 if ann.get("goal_count", 0) >= 2 else 0
        score += 10 if ann.get("annotation_quality") == "silver_rule_confirmed" else 0
        score += 5 if ann.get("requires_human_review") else 0
        rows.append(
            {
                "dataset_name": ds,
                "scene_id": ann["scene_id"],
                "priority_score": round(score, 3),
                "recommended_batch": batch(score),
                "reason": "pedestrian/drone, multi-agent density, hard episodes, candidate goals, and needs human review",
            }
        )
    rows.sort(key=lambda r: r["priority_score"], reverse=True)
    payload = {"stage": "10", "scenes": rows, "batch_a": rows[:3], "batch_b": rows[:10], "batch_c": rows[:30]}
    write_json(REPORT_DIR / "stage10_annotation_priority_list.json", payload)
    write_markdown_table(REPORT_DIR / "stage10_annotation_priority_list.md", "Stage 10 Annotation Priority List", rows)
    return payload


def batch(score: float) -> str:
    if score >= 65:
        return "A"
    if score >= 45:
        return "B"
    return "C"


def main() -> None:
    payload = select_annotation_scenes()
    print(json.dumps({"scenes": len(payload["scenes"]), "batch_a": len(payload["batch_a"])}, indent=2))


if __name__ == "__main__":
    main()
