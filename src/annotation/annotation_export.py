from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def write_annotation(annotation: Dict, root: str | Path = "data/stage8_annotations") -> Path:
    path = Path(root) / annotation["dataset_name"] / annotation["scene_id"] / "scene_annotation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
    return path


def load_annotation(dataset_name: str, scene_id: str, root: str | Path = "data/stage8_annotations") -> Dict | None:
    path = Path(root) / dataset_name / scene_id / "scene_annotation.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

