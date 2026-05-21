from __future__ import annotations

from typing import Dict, List


def build_route_graph_from_annotation(annotation: Dict) -> Dict:
    nodes = []
    edges = []
    for idx, goal in enumerate(annotation.get("goal_regions", [])):
        nodes.append({"node_id": f"goal_{idx}", "kind": goal.get("region_type", "goal"), "center": goal.get("center")})
    for route in annotation.get("route_waypoints", []):
        edges.append({"route_id": route.get("route_id", "route"), "waypoints": route.get("waypoints", [])})
    return {"nodes": nodes, "edges": edges, "source": "annotation_or_empty"}

