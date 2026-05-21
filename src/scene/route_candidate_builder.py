from __future__ import annotations

from typing import Dict, List


def route_waypoints_from_goals(boundary: Dict, goals: List[Dict]) -> List[Dict]:
    """Create simple route hypotheses from scene center to each candidate goal."""
    cx = 0.5 * (boundary["min_x"] + boundary["max_x"])
    cy = 0.5 * (boundary["min_y"] + boundary["max_y"])
    routes = []
    for goal in goals:
        gx, gy = goal["center"]
        routes.append(
            {
                "route_id": f"route_to_{goal['goal_id']}",
                "goal_id": goal["goal_id"],
                "route_type": "straight_line_inferred",
                "waypoints": [[float(cx), float(cy)], [float(gx), float(gy)]],
            }
        )
    return routes

