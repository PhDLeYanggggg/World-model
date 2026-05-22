from __future__ import annotations

from pathlib import Path
from typing import Dict


def verify_local(root: str | Path) -> Dict[str, object]:
    path = Path(root)
    return {
        "dataset_name": "opentraj",
        "local_path": str(path),
        "exists": path.exists(),
        "license": "mixed; obey each underlying dataset license",
        "loader_status": "path_verified" if path.exists() else "missing_path",
    }

