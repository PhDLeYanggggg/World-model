from __future__ import annotations

from typing import Dict, List


def suggested_goal_regions_from_scene_pack(scene_pack: Dict) -> List[Dict]:
    suggestions = []
    for goal in scene_pack.get("candidate_goal_regions", []):
        suggestions.append(
            {
                "goal_id": goal["goal_id"],
                "region_type": "inferred_goal_region",
                "center": goal["center"],
                "radius": goal.get("radius", 1.0),
                "support_count": goal.get("support_count", 0),
                "support_fraction": goal.get("support_fraction", 0.0),
                "confirmed_by_human": False,
                "source": "train_endpoint_clustering",
            }
        )
    return suggestions

