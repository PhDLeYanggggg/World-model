from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

from src.stage10_common import stage10_quality_from_previous


def make_stage10_annotation(previous: Dict, human_quality: str | None = None, annotator_id: str = "not_reviewed") -> Dict:
    quality = human_quality or stage10_quality_from_previous(previous.get("annotation_quality", "inferred_only"))
    return {
        "scene_id": previous["scene_id"],
        "dataset_name": previous["dataset_name"],
        "scene_image_path": previous.get("scene_image_path"),
        "coordinate_system": previous.get("coordinate_system", "image_or_dataset_bev"),
        "coordinate_unit": previous.get("coordinate_unit", "unknown"),
        "homography": previous.get("homography"),
        "scale_m_per_px": previous.get("scale_m_per_px"),
        "annotation_quality": quality,
        "annotator_id": annotator_id,
        "reviewer_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_at": None,
        "version": "stage10_v1",
        "walkable_polygons": previous.get("walkable_polygons", []),
        "obstacle_polygons": previous.get("obstacle_polygons", []),
        "boundary_polygon": previous.get("boundary_polygon", []),
        "entry_regions": previous.get("entry_regions", []),
        "exit_regions": previous.get("exit_regions", []),
        "goal_regions": normalize_goal_regions(previous.get("goal_regions", []), quality),
        "route_corridors": previous.get("route_corridors", []),
        "no_go_zones": previous.get("no_go_zones", []),
        "notes": previous.get("notes", "") + " Stage 10 keeps rule-confirmed labels separate from human-confirmed labels.",
        "leakage_policy": {
            "candidate_goals_from_train_split_only": True,
            "test_endpoints_used_for_candidates": False,
            "future_endpoint_used_as_inference_input": False,
            "central_velocity_used": False,
        },
        "requires_human_review": quality == "silver_rule_confirmed",
    }


def normalize_goal_regions(goals, quality: str):
    out = []
    for idx, goal in enumerate(goals):
        row = dict(goal)
        row.setdefault("goal_id", f"stage10_goal_{idx}")
        if quality == "gold_human":
            row["region_type"] = "true_goal_region"
            row["confirmed_by_human"] = True
        elif quality == "silver_human_confirmed":
            row["region_type"] = "silver_goal_region"
            row["confirmed_by_human"] = True
        elif quality == "silver_rule_confirmed":
            row["region_type"] = "silver_rule_goal_region"
            row["confirmed_by_human"] = False
            row["confirmed_by_rule"] = True
        else:
            row["region_type"] = "inferred_goal_region"
            row["confirmed_by_human"] = False
        row["future_endpoint_label_only"] = False
        out.append(row)
    return out
