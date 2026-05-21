from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def mark_reviewed(path: str, reviewer_id: str, quality: str = "silver_human_confirmed") -> dict:
    if quality not in {"gold_human", "silver_human_confirmed"}:
        raise ValueError("Stage 12 human review can only promote to gold_human or silver_human_confirmed")
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["annotation_quality"] = quality
    data["reviewer_id"] = reviewer_id
    data["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    for key in ["goal_regions", "exit_regions", "entry_regions"]:
        for region in data.get(key, []):
            region["confirmed_by_human"] = True
            if quality == "gold_human":
                region["region_type"] = "true_goal_region"
            elif "goal" in str(region.get("region_type", "")):
                region["region_type"] = "silver_goal_region"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


__all__ = ["mark_reviewed"]
