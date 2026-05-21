from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.stage10_common import REPORT_DIR, ensure_dir, write_json, write_markdown_table


OUT_DIR = Path("data/stage10_hard_failure")


def mine_stage10_hard_failure(root: str | Path = "data/stage10_multiagent_episodes") -> Dict:
    records: List[Dict] = []
    for path in sorted(Path(root).glob("*/episode_*.npz")):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"].item()))
        if meta.get("hard_label") or meta.get("baseline_failure_label"):
            records.append(
                {
                    "dataset_name": meta["dataset_name"],
                    "scene_id": meta["scene_id"],
                    "episode_id": meta["episode_id"],
                    "split": meta.get("split"),
                    "agent_count": meta.get("agent_count"),
                    "future_horizon": meta.get("future_horizon"),
                    "hard_label": bool(meta.get("hard_label")),
                    "baseline_failure_label": bool(meta.get("baseline_failure_label")),
                    "baseline_fde_proxy": meta.get("baseline_fde_proxy"),
                    "annotation_quality": meta.get("annotation_quality"),
                    "path": str(path),
                }
            )
    payload = summarize(records)
    ensure_dir(OUT_DIR)
    write_json(OUT_DIR / "hard_failure_records.json", records)
    write_json(OUT_DIR / "hard_failure_summary.json", payload)
    write_report(payload, records)
    return payload


def summarize(records: List[Dict]) -> Dict:
    return {
        "stage": "10",
        "total_records": len(records),
        "hard_episodes": sum(r["hard_label"] for r in records),
        "baseline_failure_episodes": sum(r["baseline_failure_label"] for r in records),
        "pedestrian_drone_hard_episodes": sum(r["hard_label"] for r in records if r["dataset_name"] in {"trajnet", "eth_ucy", "sdd", "opentraj"}),
        "verified_t50_records": sum(int(r.get("future_horizon", 0)) >= 50 for r in records),
        "verified_t100_records": sum(int(r.get("future_horizon", 0)) >= 100 for r in records),
        "official_training_records": sum(r["annotation_quality"] in {"gold_human", "silver_human_confirmed"} for r in records),
        "diagnostic_records": sum(r["annotation_quality"] not in {"gold_human", "silver_human_confirmed"} for r in records),
    }


def write_report(payload: Dict, records: List[Dict]) -> None:
    write_json(REPORT_DIR / "stage10_hard_failure_report.json", {"summary": payload, "records": records})
    rows = [
        {
            "dataset_name": r["dataset_name"],
            "scene_id": r["scene_id"],
            "episode_id": r["episode_id"],
            "hard": r["hard_label"],
            "failure": r["baseline_failure_label"],
            "baseline_fde": r["baseline_fde_proxy"],
            "quality": r["annotation_quality"],
        }
        for r in records[:100]
    ]
    extra = [f"{k}: {v}" for k, v in payload.items() if k != "stage"]
    write_markdown_table(REPORT_DIR / "stage10_hard_failure_report.md", "Stage 10 Hard/Failure Episode Report", rows, extra)


def main() -> None:
    payload = mine_stage10_hard_failure()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
