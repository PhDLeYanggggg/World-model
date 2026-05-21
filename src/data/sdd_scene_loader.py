from __future__ import annotations

from pathlib import Path
from typing import Dict


def inspect_sdd_scene_root(root: str | Path | None = None) -> Dict:
    path = Path(root) if root else Path("data/stage7_raw/sdd")
    exists = path.exists()
    return {
        "dataset_name": "stanford_drone_dataset",
        "actual_downloaded_or_user_path_verified": exists,
        "license": "non-commercial research license; manual acceptance required",
        "coordinate_unit": "pixel_or_image_coordinate_until_homography",
        "homography_available": False,
        "scene_image_available": exists,
        "walkable_area_available": False,
        "obstacle_geometry_available": False,
        "exit_goal_region_available": False,
        "notes": "Stage 7 does not auto-download SDD because license/manual terms must be handled by the user.",
    }

