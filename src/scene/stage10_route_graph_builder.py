from __future__ import annotations

from typing import Dict


def build_stage10_route_graph(annotation: Dict) -> Dict:
    goals = annotation.get("goal_regions", [])
    return {
        "available": bool(annotation.get("route_corridors")),
        "node_count": len(goals),
        "edge_count": 0,
        "note": "Route graph is only explicit when route_corridors are human-confirmed.",
    }
