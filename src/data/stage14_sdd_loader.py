from __future__ import annotations

from pathlib import Path
from typing import Dict


def verify_local(root: str | Path) -> Dict[str, object]:
    path = Path(root)
    return {
        "dataset_name": "sdd",
        "local_path": str(path),
        "exists": path.exists(),
        "license": "Stanford Drone Dataset non-commercial; user must accept terms",
        "loader_status": "path_verified" if path.exists() else "missing_path",
        "metric_status": "pixel_or_unknown_until_homography",
    }

