from __future__ import annotations

from pathlib import Path
from typing import Dict


def inspect_full_trajnet_scene_root(root: str | Path | None = None) -> Dict:
    path = Path(root) if root else Path("data/stage5b_raw/trajnetplusplusdataset")
    return {
        "dataset_name": "full_trajnetplusplus",
        "actual_downloaded_or_user_path_verified": path.exists(),
        "license": "dataset-specific; TrajNet++ repo includes references to original data terms",
        "coordinate_unit": "dataset_coordinate",
        "homography_available": False,
        "scene_image_available": path.exists(),
        "walkable_area_available": False,
        "obstacle_geometry_available": False,
        "exit_goal_region_available": False,
        "notes": "Local TrajNet++ source exists, but current quick conversion only supports t+10 pedestrian-like episodes.",
    }

