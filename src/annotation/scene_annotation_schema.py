from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List


SCHEMA_VERSION = "stage8_scene_annotation_v1"


def make_annotation(
    dataset_name: str,
    scene_id: str,
    coordinate_unit: str,
    boundary_polygon: List[List[float]],
    goal_regions: List[Dict],
    image_path: str | None = None,
    annotation_quality: str = "inferred_only",
    annotator: str = "auto_stage8_endpoint_suggestion",
) -> Dict:
    return {
        "scene_id": scene_id,
        "dataset_name": dataset_name,
        "coordinate_system": "local_scene_origin",
        "coordinate_unit": coordinate_unit,
        "image_path": image_path,
        "homography": None,
        "scale_m_per_px": None,
        "annotation_quality": annotation_quality,
        "walkable_polygons": [{"polygon": boundary_polygon, "source": "inferred_observed_bbox"}],
        "obstacle_polygons": [],
        "boundary_polygon": boundary_polygon,
        "entry_regions": [],
        "exit_regions": [],
        "goal_regions": goal_regions,
        "route_waypoints": [],
        "notes": "Generated automatically from Stage 7 scene pack. Not gold unless manually edited.",
        "annotator": annotator,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": SCHEMA_VERSION,
    }

