from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List


def make_stage8p5_annotation(
    dataset_name: str,
    scene_id: str,
    coordinate_unit: str,
    boundary_polygon: List[List[float]],
    goal_regions: List[Dict],
    annotation_quality: str,
    walkable_polygons: List[List[List[float]]] | None = None,
    scene_image_path: str | None = None,
    homography=None,
    scale_m_per_px=None,
    notes: str = "",
) -> Dict:
    return {
        "scene_id": scene_id,
        "dataset_name": dataset_name,
        "coordinate_system": "image_or_dataset_bev",
        "coordinate_unit": coordinate_unit,
        "scene_image_path": scene_image_path,
        "homography": homography,
        "scale_m_per_px": scale_m_per_px,
        "annotation_quality": annotation_quality,
        "walkable_polygons": walkable_polygons or [boundary_polygon],
        "obstacle_polygons": [],
        "boundary_polygon": boundary_polygon,
        "entry_regions": [],
        "exit_regions": goal_regions,
        "goal_regions": goal_regions,
        "route_corridors": [],
        "annotator": "stage8p5_rule_confirmed" if annotation_quality == "silver" else "stage8p5_auto",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "stage8p5_v1",
        "notes": notes,
        "leakage_policy": {
            "candidate_goals_from_train_split_only": True,
            "test_endpoints_used_for_candidates": False,
            "future_endpoint_used_as_inference_input": False,
        },
    }
