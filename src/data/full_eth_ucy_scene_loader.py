from __future__ import annotations

from pathlib import Path
from typing import Dict


def inspect_full_eth_ucy_scene_root(root: str | Path | None = None) -> Dict:
    path = Path(root) if root else Path("data/stage5b_raw/trajnetplusplusdataset")
    return {
        "dataset_name": "full_eth_ucy",
        "actual_downloaded_or_user_path_verified": path.exists(),
        "license": "ETH/UCY academic pedestrian trajectory data; verify original terms before redistribution",
        "coordinate_unit": "dataset_coordinate_or_meter_depending_source",
        "homography_available": False,
        "scene_image_available": path.exists(),
        "walkable_area_available": False,
        "obstacle_geometry_available": False,
        "exit_goal_region_available": False,
        "notes": "Current converted ETH/UCY fallback is t+10 only; no verified pedestrian t+50/t+100 locally.",
    }

