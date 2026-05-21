from __future__ import annotations

import shutil
from pathlib import Path

from src.stage10_common import ensure_dir


def export_annotation_package(src_root: str | Path = "data/stage10_annotations", dst_root: str | Path = "outputs/world_model_stage10_results/annotations") -> int:
    src = Path(src_root)
    dst = ensure_dir(dst_root)
    count = 0
    for path in sorted(src.glob("*/*/scene_annotation.json")):
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        count += 1
    return count
