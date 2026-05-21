from __future__ import annotations

from typing import Dict, List


VALID_QUALITIES = {"gold", "silver", "inferred_only"}


def validate_annotation(annotation: Dict) -> List[str]:
    errors: List[str] = []
    for key in ["scene_id", "dataset_name", "coordinate_unit", "annotation_quality", "boundary_polygon", "goal_regions"]:
        if key not in annotation:
            errors.append(f"missing:{key}")
    if annotation.get("annotation_quality") not in VALID_QUALITIES:
        errors.append("annotation_quality_must_be_gold_silver_or_inferred_only")
    if annotation.get("annotation_quality") == "gold":
        if not annotation.get("walkable_polygons") or not annotation.get("goal_regions"):
            errors.append("gold_requires_walkable_and_goal_regions")
    if annotation.get("coordinate_unit") == "pixel" and annotation.get("scale_m_per_px"):
        errors.append("pixel_annotation_with_scale_m_per_px_must_be_marked_weak_metric_or_metric")
    for goal in annotation.get("goal_regions", []):
        if goal.get("region_type") == "true_goal_region" and annotation.get("annotation_quality") == "inferred_only":
            errors.append("inferred_only_annotation_cannot_contain_true_goal_region")
    return errors

