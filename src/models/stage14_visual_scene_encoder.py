from __future__ import annotations

from pathlib import Path
from typing import Dict


def encode_scene_image_reference(path: str | Path | None) -> Dict[str, object]:
    if not path:
        return {"available": False}
    p = Path(path)
    return {"available": p.exists(), "path": str(p)}

