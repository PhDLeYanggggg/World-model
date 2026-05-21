from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src.stage10_common import ensure_dir


def build_annotation_task(annotation: Dict, previous_annotation_path: str | Path, preview_path: str | None = None) -> Dict:
    return {
        "task_type": "stage10_human_scene_goal_review",
        "dataset_name": annotation["dataset_name"],
        "scene_id": annotation["scene_id"],
        "annotation_path": str(previous_annotation_path),
        "suggested_annotation_quality": annotation["annotation_quality"],
        "target_human_actions": [
            "confirm_or_edit_walkable_polygons",
            "confirm_or_edit_boundary_polygon",
            "confirm_or_edit_entry_exit_goal_regions",
            "mark_obstacles_or_no_go_zones_if_visible",
            "set annotation_quality to gold_human or silver_human_confirmed only after review",
        ],
        "preview_image": preview_path or "",
        "trajectory_heatmap": preview_path or "",
        "endpoint_heatmap_from_train_split_only": True,
        "test_endpoints_used": False,
        "future_endpoint_as_input": False,
        "notes": "This is an annotation task, not a completed human annotation.",
    }


def write_annotation_task(task: Dict, out_root: str | Path = "data/stage10_annotation_tasks") -> Path:
    out_dir = ensure_dir(Path(out_root) / task["dataset_name"] / task["scene_id"])
    path = out_dir / "annotation_task.json"
    path.write_text(json.dumps(task, indent=2), encoding="utf-8")
    return path
