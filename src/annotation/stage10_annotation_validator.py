from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.stage10_common import REPORT_DIR, is_human_annotation_quality, write_json, write_markdown_table


def validate_stage10_annotations(root: str | Path = "data/stage10_annotations") -> Dict:
    rows: List[Dict] = []
    for path in sorted(Path(root).glob("*/*/scene_annotation.json")):
        ann = json.loads(path.read_text(encoding="utf-8"))
        errors = []
        if not ann.get("boundary_polygon"):
            errors.append("missing_boundary_polygon")
        if not ann.get("walkable_polygons"):
            errors.append("missing_walkable_polygons")
        if not ann.get("goal_regions"):
            errors.append("missing_goal_regions")
        if ann.get("leakage_policy", {}).get("test_endpoints_used_for_candidates"):
            errors.append("test_endpoints_used_for_candidates")
        if ann.get("annotation_quality") in {"gold_human", "silver_human_confirmed"} and not ann.get("reviewed_at"):
            errors.append("human_quality_without_review_timestamp")
        rows.append(
            {
                "dataset_name": ann.get("dataset_name"),
                "scene_id": ann.get("scene_id"),
                "annotation_quality": ann.get("annotation_quality"),
                "human_confirmed": is_human_annotation_quality(ann.get("annotation_quality")),
                "valid": not errors,
                "errors": errors,
                "path": str(path),
            }
        )
    payload = {"stage": "10", "annotations": rows, "valid_count": sum(r["valid"] for r in rows)}
    write_validation_report(payload)
    return payload


def write_validation_report(payload: Dict) -> None:
    write_json(REPORT_DIR / "stage10_annotation_validation.json", payload)
    write_markdown_table(
        REPORT_DIR / "stage10_annotation_validation.md",
        "Stage 10 Annotation Validation",
        [{k: r.get(k) for k in ["dataset_name", "scene_id", "annotation_quality", "human_confirmed", "valid", "errors"]} for r in payload["annotations"]],
    )


def main() -> None:
    payload = validate_stage10_annotations()
    print(json.dumps({"annotations": len(payload["annotations"]), "valid": payload["valid_count"]}, indent=2))


if __name__ == "__main__":
    main()
