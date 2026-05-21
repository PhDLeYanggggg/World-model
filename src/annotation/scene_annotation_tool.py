from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.annotation.annotation_export import write_annotation
from src.annotation.auto_goal_suggestions import suggested_goal_regions_from_scene_pack
from src.annotation.scene_annotation_schema import make_annotation


def prepare_annotations_from_scene_packs(scene_pack_root: str | Path = "data/scene_packs") -> List[Dict]:
    annotations = []
    for pack_path in Path(scene_pack_root).glob("*/*/scene_pack.json"):
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        goals = suggested_goal_regions_from_scene_pack(pack)
        annotation = make_annotation(
            dataset_name=pack["dataset_name"],
            scene_id=pack["scene_id"],
            coordinate_unit=pack["coordinate_unit"],
            boundary_polygon=pack["boundary_polygon"],
            goal_regions=goals,
            image_path=pack.get("scene_image"),
            annotation_quality="inferred_only",
        )
        write_annotation(annotation)
        annotations.append(annotation)
    return annotations


def prepare_annotations_from_multiagent_episodes(root: str | Path = "data/stage8_multiagent_episodes") -> List[Dict]:
    """Create inferred-only annotations for stage8 scenes that lack packs.

    Goal suggestions use train-split future endpoints only. If no train split is
    available, the scene is skipped rather than using test endpoints.
    """

    annotations = []
    for dataset_dir in sorted(Path(root).glob("*")):
        if not dataset_dir.is_dir():
            continue
        grouped: Dict[str, List[Dict]] = {}
        for path in sorted(dataset_dir.glob("episode_*.npz")):
            data = np.load(path, allow_pickle=True)
            meta = json.loads(str(data["meta"].item()))
            if meta.get("split") != "train":
                continue
            grouped.setdefault(str(meta.get("scene_id", dataset_dir.name)), []).append({"states": data["states"].astype(float), "meta": meta})
        for scene_id, episodes in grouped.items():
            ann_path = Path("data/stage8_annotations") / dataset_dir.name / scene_id / "scene_annotation.json"
            if ann_path.exists():
                continue
            all_pos = []
            endpoints = []
            coordinate_unit = "unknown"
            for ep in episodes:
                states = ep["states"]
                meta = ep["meta"]
                coordinate_unit = str(meta.get("coordinate_unit", coordinate_unit))
                all_pos.append(states[:, :, 0:2].reshape(-1, 2))
                endpoints.append(states[-1, 0, 0:2])
            if not endpoints:
                continue
            positions = np.concatenate(all_pos, axis=0)
            positions = positions[np.linalg.norm(positions, axis=1) > 0]
            if len(positions) == 0:
                continue
            pad = max(float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1])), 1.0) * 0.05
            xmin, ymin = positions.min(axis=0) - pad
            xmax, ymax = positions.max(axis=0) + pad
            goals = goals_from_endpoints(np.asarray(endpoints, dtype=float))
            annotation = make_annotation(
                dataset_name=dataset_dir.name,
                scene_id=scene_id,
                coordinate_unit=coordinate_unit,
                boundary_polygon=[[float(xmin), float(ymin)], [float(xmax), float(ymin)], [float(xmax), float(ymax)], [float(xmin), float(ymax)]],
                goal_regions=goals,
                image_path=None,
                annotation_quality="inferred_only",
            )
            write_annotation(annotation)
            annotations.append(annotation)
    return annotations


def goals_from_endpoints(endpoints: np.ndarray, max_goals: int = 5) -> List[Dict]:
    if len(endpoints) == 0:
        return []
    centers = [endpoints[0]]
    for point in endpoints[1:]:
        d = np.linalg.norm(np.asarray(centers) - point[None, :], axis=1)
        if float(d.min()) > 5.0 and len(centers) < max_goals:
            centers.append(point)
    assignments = []
    for point in endpoints:
        d = np.linalg.norm(np.asarray(centers) - point[None, :], axis=1)
        assignments.append(int(np.argmin(d)))
    goals = []
    for idx, center in enumerate(centers):
        support = sum(a == idx for a in assignments)
        goals.append(
            {
                "goal_id": f"inferred_goal_{idx}",
                "region_type": "inferred_goal_region",
                "center": [float(center[0]), float(center[1])],
                "radius": 2.0,
                "support_count": int(support),
                "support_fraction": float(support / max(len(assignments), 1)),
                "confirmed_by_human": False,
                "source": "train_split_endpoint_clustering",
            }
        )
    return goals
