from __future__ import annotations

from typing import Dict, List


def validate_stage8p5_annotation(annotation: Dict) -> List[str]:
    errors = []
    for key in ["scene_id", "dataset_name", "coordinate_unit", "annotation_quality", "boundary_polygon", "goal_regions"]:
        if key not in annotation:
            errors.append(f"missing {key}")
    if annotation.get("annotation_quality") not in {"gold", "silver", "inferred_only"}:
        errors.append("annotation_quality must be gold/silver/inferred_only")
    if len(annotation.get("boundary_polygon", [])) < 3:
        errors.append("boundary_polygon must have at least 3 points")
    if annotation.get("annotation_quality") in {"gold", "silver"} and not annotation.get("goal_regions"):
        errors.append("gold/silver annotations require goal regions")
    if not annotation.get("leakage_policy", {}).get("candidate_goals_from_train_split_only", False):
        errors.append("candidate goals must be train split only")
    if annotation.get("leakage_policy", {}).get("test_endpoints_used_for_candidates", True):
        errors.append("test endpoints used for candidate goals")
    return errors
