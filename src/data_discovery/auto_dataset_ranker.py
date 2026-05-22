from __future__ import annotations

from typing import Dict, List


def rank_datasets(rows: List[Dict]) -> List[Dict]:
    def score(row: Dict) -> int:
        value = int(row.get("priority_score", 0) or 0)
        if row.get("trajectory_annotation_available") is True:
            value += 5
        if row.get("scene_image_available") is True:
            value += 5
        if row.get("t100_possible") is True:
            value += 5
        if row.get("requires_login") is True or row.get("requires_application") is True:
            value -= 5
        return value

    ranked = []
    for row in rows:
        item = dict(row)
        item["auto_rank_score"] = score(row)
        ranked.append(item)
    ranked.sort(key=lambda item: item["auto_rank_score"], reverse=True)
    return ranked

