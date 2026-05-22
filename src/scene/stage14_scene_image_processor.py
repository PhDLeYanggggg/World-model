from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from PIL import Image


def image_info(path: str | Path) -> Dict[str, object]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "size": None}
    with Image.open(p) as img:
        size: Tuple[int, int] = img.size
    return {"path": str(p), "exists": True, "size": list(size)}

