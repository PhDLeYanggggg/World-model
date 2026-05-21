from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


ROOT = Path("data/stage8p5_annotations")


def write_stage8p5_annotation(annotation: Dict) -> Path:
    out = ROOT / annotation["dataset_name"] / annotation["scene_id"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scene_annotation.json"
    path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
    return path


def read_stage8p5_annotation(dataset: str, scene_id: str) -> Dict | None:
    p = ROOT / dataset / scene_id / "scene_annotation.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
