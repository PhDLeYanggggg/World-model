from __future__ import annotations

from typing import Dict


def scene_geometry_status(metadata: Dict) -> Dict:
    return {
        "has_scene_map": bool(metadata.get("has_scene_map", False) or metadata.get("whether_scene_geometry_available", False)),
        "has_obstacle_geometry": bool(metadata.get("has_obstacle_geometry", False)),
        "has_walkable_area": bool(metadata.get("has_walkable_area", False)),
        "source": "metadata_or_registry",
        "note": "Stage 5 quick path does not hallucinate geometry; unknown geometry remains unknown.",
    }
