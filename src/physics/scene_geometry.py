from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str = "obstacle"


@dataclass
class SceneSpec:
    name: str
    width: float
    height: float
    obstacles: List[Rect]
    exits: Dict[str, Tuple[float, float]]
    spawn_regions: List[Dict]
    event_hint: str
    has_real_boundary: bool = True
    has_real_obstacles: bool = True
    has_real_exits: bool = True


def make_scene_templates() -> List[SceneSpec]:
    return [
        SceneSpec(
            "open_space",
            42.0,
            26.0,
            [Rect(19.0, 11.0, 23.0, 15.0, "kiosk")],
            {"west": (1.0, 13.0), "east": (41.0, 13.0), "north": (21.0, 1.0), "south": (21.0, 25.0)},
            [
                {"rect": (1.0, 8.0, 7.0, 18.0), "goal": "east", "weight": 0.55},
                {"rect": (35.0, 8.0, 41.0, 18.0), "goal": "west", "weight": 0.25},
                {"rect": (16.0, 20.0, 26.0, 25.0), "goal": "north", "weight": 0.20},
            ],
            "smooth_passage",
        ),
        SceneSpec(
            "narrow_corridor_jam",
            42.0,
            26.0,
            [Rect(18.5, 0.0, 22.0, 9.7, "wall"), Rect(18.5, 16.3, 22.0, 26.0, "wall"), Rect(8.0, 3.0, 12.0, 7.5, "booth")],
            {"west": (1.0, 13.0), "east": (41.0, 13.0), "north": (21.0, 1.0), "south": (21.0, 25.0)},
            [
                {"rect": (1.0, 8.0, 7.0, 18.0), "goal": "east", "weight": 0.70},
                {"rect": (35.0, 8.0, 41.0, 18.0), "goal": "west", "weight": 0.30},
            ],
            "corridor_jam",
        ),
        SceneSpec(
            "obstacle_detour",
            42.0,
            26.0,
            [Rect(16.5, 6.0, 25.5, 20.0, "large_obstacle"), Rect(6.0, 0.0, 9.0, 7.0, "wall"), Rect(6.0, 19.0, 9.0, 26.0, "wall")],
            {"west": (1.0, 13.0), "east": (41.0, 13.0), "north": (21.0, 1.0), "south": (21.0, 25.0)},
            [
                {"rect": (1.0, 7.0, 6.0, 19.0), "goal": "east", "weight": 0.80},
                {"rect": (34.0, 7.0, 41.0, 19.0), "goal": "west", "weight": 0.20},
            ],
            "obstacle_detour",
        ),
        SceneSpec(
            "crossing_split_merge",
            42.0,
            26.0,
            [Rect(19.0, 11.0, 23.0, 15.0, "kiosk"), Rect(7.0, 7.0, 11.0, 11.0, "booth"), Rect(31.0, 15.0, 35.0, 19.0, "booth")],
            {"west": (1.0, 13.0), "east": (41.0, 13.0), "north": (21.0, 1.0), "south": (21.0, 25.0)},
            [
                {"rect": (1.0, 9.0, 7.0, 17.0), "goal": "east", "weight": 0.35},
                {"rect": (35.0, 9.0, 41.0, 17.0), "goal": "west", "weight": 0.25},
                {"rect": (16.0, 20.0, 26.0, 25.0), "goal": "north", "weight": 0.25},
                {"rect": (16.0, 1.0, 26.0, 6.0), "goal": "south", "weight": 0.15},
            ],
            "group_split",
        ),
    ]


def scene_to_dict(scene: SceneSpec) -> Dict:
    return {
        "name": scene.name,
        "width": scene.width,
        "height": scene.height,
        "obstacles": [rect.__dict__ for rect in scene.obstacles],
        "exits": scene.exits,
        "spawn_regions": scene.spawn_regions,
        "event_hint": scene.event_hint,
    }


def scene_from_dict(data: Dict) -> SceneSpec:
    return SceneSpec(
        name=data["name"],
        width=float(data["width"]),
        height=float(data["height"]),
        obstacles=[Rect(**rect) for rect in data["obstacles"]],
        exits={key: tuple(value) for key, value in data["exits"].items()},
        spawn_regions=data["spawn_regions"],
        event_hint=data["event_hint"],
    )


def point_in_rect(point: tuple[float, float], rect: Rect, pad: float = 0.0) -> bool:
    return rect.x1 - pad <= point[0] <= rect.x2 + pad and rect.y1 - pad <= point[1] <= rect.y2 + pad


def point_in_any_rect(point: tuple[float, float], rects: list[Rect], pad: float = 0.0) -> bool:
    return any(point_in_rect(point, rect, pad) for rect in rects)


def nearest_point_rect(point: tuple[float, float], rect: Rect) -> tuple[float, float]:
    return float(np.clip(point[0], rect.x1, rect.x2)), float(np.clip(point[1], rect.y1, rect.y2))


def push_out_rect(point: tuple[float, float], rect: Rect, pad: float) -> tuple[float, float]:
    x, y = point
    candidates = [
        (abs(x - (rect.x1 - pad)), (rect.x1 - pad, y)),
        (abs(x - (rect.x2 + pad)), (rect.x2 + pad, y)),
        (abs(y - (rect.y1 - pad)), (x, rect.y1 - pad)),
        (abs(y - (rect.y2 + pad)), (x, rect.y2 + pad)),
    ]
    return min(candidates, key=lambda item: item[0])[1]


def nearest_obstacle_vector(point: tuple[float, float], scene: SceneSpec) -> tuple[np.ndarray, float, bool]:
    best_vec = np.array([1.0, 0.0], dtype=np.float32)
    best_dist = 99.0
    inside_any = False
    for rect in scene.obstacles:
        nearest = nearest_point_rect(point, rect)
        vec = np.array([point[0] - nearest[0], point[1] - nearest[1]], dtype=np.float32)
        dist = float(np.linalg.norm(vec))
        inside = point_in_rect(point, rect)
        if inside:
            inside_any = True
            center = np.array([(rect.x1 + rect.x2) * 0.5, (rect.y1 + rect.y2) * 0.5], dtype=np.float32)
            vec = np.array(point, dtype=np.float32) - center
            dist = max(1e-5, float(np.linalg.norm(vec)))
        if dist < best_dist:
            best_dist = dist
            best_vec = vec / max(1e-5, dist)
    return best_vec.astype(np.float32), best_dist, inside_any
