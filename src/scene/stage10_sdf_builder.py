from __future__ import annotations

from typing import Dict, List


def analytic_sdf_summary(annotation: Dict) -> Dict:
    return {
        "walkable_sdf": "analytic_polygon_distance_placeholder",
        "obstacle_sdf": "not_available" if not annotation.get("obstacle_polygons") else "analytic_obstacle_polygon_distance",
        "goal_distance_fields": [
            {"goal_id": g.get("goal_id", f"goal_{i}"), "center": g.get("center"), "radius": g.get("radius")}
            for i, g in enumerate(annotation.get("goal_regions", []))
        ],
    }
