from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.stage10_common import REPORT_DIR, is_human_annotation_quality, write_json, write_markdown_table


def validate_stage10_annotations(root: str | Path = "data/stage10_annotations") -> Dict:
    rows: List[Dict] = []
    report_rows: List[Dict] = []
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
        report_rows.append(
            {
                "dataset_name": ann.get("dataset_name"),
                "scene_id": ann.get("scene_id"),
                "annotation_quality": ann.get("annotation_quality"),
                "goal_count": len(ann.get("goal_regions", [])),
                "requires_human_review": ann.get("requires_human_review", not is_human_annotation_quality(ann.get("annotation_quality"))),
                "task_path": str(Path("data/stage10_annotation_tasks") / str(ann.get("dataset_name")) / str(ann.get("scene_id")) / "annotation_task.json"),
                "annotation_path": str(path),
                "test_endpoints_used": bool(ann.get("leakage_policy", {}).get("test_endpoints_used_for_candidates")),
            }
        )
    payload = {"stage": "10", "annotations": rows, "valid_count": sum(r["valid"] for r in rows)}
    write_validation_report(payload)
    write_current_annotation_report({"stage": "10", "annotations": report_rows})
    return payload


def write_validation_report(payload: Dict) -> None:
    write_json(REPORT_DIR / "stage10_annotation_validation.json", payload)
    write_markdown_table(
        REPORT_DIR / "stage10_annotation_validation.md",
        "Stage 10 Annotation Validation",
        [{k: r.get(k) for k in ["dataset_name", "scene_id", "annotation_quality", "human_confirmed", "valid", "errors"]} for r in payload["annotations"]],
    )


def write_current_annotation_report(payload: Dict) -> None:
    write_json(REPORT_DIR / "stage10_annotation_report.json", payload)
    rows = payload["annotations"]
    table_rows = [
        {
            "dataset_name": r["dataset_name"],
            "scene_id": r["scene_id"],
            "annotation_quality": r["annotation_quality"],
            "goal_count": r["goal_count"],
            "requires_human_review": r["requires_human_review"],
        }
        for r in rows
    ]
    extra = [
        f"gold_human scenes: {sum(r['annotation_quality'] == 'gold_human' for r in rows)}",
        f"silver_human_confirmed scenes: {sum(r['annotation_quality'] == 'silver_human_confirmed' for r in rows)}",
        f"silver_rule_confirmed scenes: {sum(r['annotation_quality'] == 'silver_rule_confirmed' for r in rows)}",
        f"inferred_only scenes: {sum(r['annotation_quality'] == 'inferred_only' for r in rows)}",
        "Annotation report is refreshed from current scene_annotation.json files during validation.",
    ]
    write_markdown_table(REPORT_DIR / "stage10_annotation_report.md", "Stage 10 Annotation Task Report", table_rows, extra)


def main() -> None:
    payload = validate_stage10_annotations()
    print(json.dumps({"annotations": len(payload["annotations"]), "valid": payload["valid_count"]}, indent=2))


if __name__ == "__main__":
    main()
