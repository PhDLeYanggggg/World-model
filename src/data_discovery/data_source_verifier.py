from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_EXPECTED_PATHS = {
    "sdd": ["annotations", "videos"],
    "opentraj": ["datasets"],
    "trajnet": ["data"],
    "eth_ucy": ["seq_eth", "seq_hotel"],
}


def verify_local_source(root: str | Path, expected_children: Iterable[str] | None = None) -> Dict:
    root = Path(root)
    children = list(expected_children or [])
    return {
        "root": str(root),
        "exists": root.exists(),
        "is_dir": root.is_dir(),
        "missing_children": [child for child in children if not (root / child).exists()] if root.exists() else children,
    }


def expected_path_structure() -> Dict[str, List[str]]:
    return DEFAULT_EXPECTED_PATHS

