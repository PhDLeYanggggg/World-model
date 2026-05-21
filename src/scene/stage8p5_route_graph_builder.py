from __future__ import annotations

from typing import Dict


def build_route_graph(annotation: Dict) -> Dict:
    goals = annotation.get("goal_regions", [])
    corridors = annotation.get("route_corridors", [])
    return {
        "nodes": [{"id": g.get("goal_id"), "type": g.get("region_type"), "center": g.get("center")} for g in goals],
        "edges": corridors,
        "source": "annotation_route_corridors" if corridors else "goal_complete_graph_placeholder",
    }
