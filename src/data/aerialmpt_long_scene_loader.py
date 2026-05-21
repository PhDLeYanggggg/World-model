from __future__ import annotations

from pathlib import Path
from typing import Dict


def inspect_aerialmpt_long_root(root: str | Path | None = None) -> Dict:
    path = Path(root) if root else Path("data/aerialmpt/extracted")
    return {
        "dataset_name": "aerialmpt_long",
        "actual_downloaded_or_user_path_verified": path.exists(),
        "license": "local extracted sample; verify source license before publication",
        "coordinate_unit": "pixel",
        "homography_available": False,
        "scene_image_available": path.exists(),
        "walkable_area_available": False,
        "obstacle_geometry_available": False,
        "exit_goal_region_available": False,
        "notes": "Existing AerialMPT bauma sample is short; t+100 remains qualitative-only.",
    }

