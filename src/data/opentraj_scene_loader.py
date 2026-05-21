from __future__ import annotations

from pathlib import Path
from typing import Dict


def inspect_opentraj_root(root: str | Path | None = None) -> Dict:
    path = Path(root) if root else Path("data/stage7_raw/opentraj")
    return {
        "dataset_name": "opentraj_supported_pedestrian",
        "actual_downloaded_or_user_path_verified": path.exists(),
        "license": "varies_by_source_dataset",
        "coordinate_unit": "varies",
        "homography_available": False,
        "scene_image_available": False,
        "walkable_area_available": False,
        "obstacle_geometry_available": False,
        "exit_goal_region_available": False,
        "notes": "Placeholder inspector only; source-specific conversion is required before official benchmark use.",
    }

