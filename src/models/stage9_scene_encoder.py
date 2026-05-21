from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_scene_pack(dataset: str, scene_id: str) -> dict:
    p = Path("data/stage8p5_scene_gold_packs") / dataset / scene_id / "scene_gold_pack.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def encode_scene(pack: dict, position: np.ndarray) -> np.ndarray:
    quality = pack.get("annotation_quality", "not_available")
    boundary = np.asarray(pack.get("boundary_polygon", []), dtype=float)
    if len(boundary) >= 3:
        xmin, ymin = boundary.min(axis=0)
        xmax, ymax = boundary.max(axis=0)
        dist = min(position[0] - xmin, xmax - position[0], position[1] - ymin, ymax - position[1])
        inside = float(xmin <= position[0] <= xmax and ymin <= position[1] <= ymax)
        scale = max(float(xmax - xmin), float(ymax - ymin), 1.0)
    else:
        dist, inside, scale = 0.0, 0.0, 1.0
    return np.asarray(
        [
            1.0 if quality == "gold" else 0.0,
            1.0 if quality == "silver" else 0.0,
            1.0 if quality == "inferred_only" else 0.0,
            inside,
            float(np.clip(dist / scale, -1.0, 1.0)),
            min(len(pack.get("obstacle_polygons", [])) / 5.0, 1.0),
        ],
        dtype=float,
    )
