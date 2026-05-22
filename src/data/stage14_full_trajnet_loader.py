from __future__ import annotations

from pathlib import Path
from typing import Dict


def verify_local(root: str | Path) -> Dict[str, object]:
    path = Path(root)
    return {
        "dataset_name": "trajnet_full",
        "local_path": str(path),
        "exists": path.exists(),
        "loader_status": "path_verified" if path.exists() else "missing_path",
    }

