from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.evaluation.baseline_benchmark_stage5b import available_datasets, load_dataset_episodes
from src.scene.goal_region_builder import endpoint_clusters
from src.scene.route_candidate_builder import route_waypoints_from_goals
from src.scene.walkable_area_builder import boundary_summary, rectangular_boundary


OUT_DIR = Path("data/scene_packs")
REPORT_DIR = Path("outputs/reports")


def collect_scene_points(dataset: str) -> Dict[str, Dict[str, List[np.ndarray]]]:
    scenes: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: {"all_points": [], "train_endpoints": [], "all_endpoints": []})
    for ep in load_dataset_episodes(dataset, split="all"):
        states = ep["states"]
        meta = ep["meta"]
        scene_id = str(meta.get("scene_id", dataset))
        past = int(meta.get("past_horizon", 10))
        scenes[scene_id]["all_points"].append(states[:, :, 0:2].reshape(-1, 2))
        scenes[scene_id]["all_endpoints"].append(states[-1, 0, 0:2])
        if meta.get("split") == "train":
            scenes[scene_id]["train_endpoints"].append(states[-1, 0, 0:2])
        # Decision points are allowed as causal context and useful for bbox scale.
        scenes[scene_id].setdefault("decision_points", []).append(states[past - 1, 0, 0:2])
    return scenes


def build_scene_pack(dataset: str, scene_id: str, bucket: Dict[str, List[np.ndarray]]) -> Dict:
    points = np.concatenate(bucket["all_points"], axis=0) if bucket["all_points"] else np.zeros((0, 2))
    train_endpoints = np.asarray(bucket["train_endpoints"] or bucket["all_endpoints"], dtype=float).reshape(-1, 2)
    polygon = rectangular_boundary(points)
    boundary = boundary_summary(polygon)
    goals = endpoint_clusters(train_endpoints, max_goals=6)
    coordinate_unit = infer_coordinate_unit(dataset)
    scene_image_available = dataset in {"trajnet", "eth_ucy"} and Path("data/stage5b_raw/trajnetplusplusdataset").exists()
    has_homography = False
    metric = coordinate_unit == "meter"
    annotation_quality = {
        "walkable_area": "inferred_bbox_not_manual",
        "obstacles": "not_available",
        "goals": "inferred_scene_goal_from_training_endpoints",
        "homography": "not_available",
        "metric_evaluation": "available" if metric else "not_available",
    }
    return {
        "scene_id": scene_id,
        "dataset_name": dataset,
        "coordinate_unit": coordinate_unit,
        "scene_image": None,
        "scene_image_available": scene_image_available,
        "homography": None,
        "has_homography": has_homography,
        "walkable_area": {"type": "polygon", "source": "inferred_bbox_from_observed_trajectories", "polygon": polygon},
        "obstacle_polygons": [],
        "boundary_polygon": polygon,
        "boundary_summary": boundary,
        "candidate_goal_regions": goals,
        "candidate_route_waypoints": route_waypoints_from_goals(boundary, goals),
        "distance_to_goal_fields": "computed_on_demand_from_candidate_goal_regions",
        "walkability_sdf": "analytic_rectangular_boundary_distance",
        "obstacle_sdf": "not_available",
        "density_prior_map": "not_available",
        "annotation_quality": annotation_quality,
        "goal_label_policy": {
            "inference_input": "past_trajectory_plus_scene_candidate_goal_dictionary_only",
            "training_label": "future_endpoint_cluster_label",
            "warning": "future endpoint is never an inference feature",
        },
    }


def infer_coordinate_unit(dataset: str) -> str:
    episodes = load_dataset_episodes(dataset, split="all")
    if not episodes:
        return "unknown"
    unit = episodes[0]["meta"].get("coordinate_unit", "unknown")
    if unit == "meter":
        return "meter"
    return str(unit)


def build_scene_packs(datasets: List[str] | None = None) -> Dict:
    datasets = datasets or available_datasets()
    packs = []
    for dataset in datasets:
        scene_buckets = collect_scene_points(dataset)
        for scene_id, bucket in scene_buckets.items():
            pack = build_scene_pack(dataset, scene_id, bucket)
            scene_dir = OUT_DIR / dataset / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / "scene_pack.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
            packs.append(pack)
    return {"stage": "7", "scene_packs": packs, "total_scene_packs": len(packs)}


def write_report(payload: Dict) -> Dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for pack in payload["scene_packs"]:
        rows.append(
            {
                "dataset": pack["dataset_name"],
                "scene": pack["scene_id"],
                "unit": pack["coordinate_unit"],
                "homography": pack["has_homography"],
                "goals": len(pack["candidate_goal_regions"]),
                "walkable_area": pack["annotation_quality"]["walkable_area"],
                "goal_source": pack["annotation_quality"]["goals"],
                "metric_eval": pack["annotation_quality"]["metric_evaluation"],
            }
        )
    (REPORT_DIR / "stage7_scene_pack_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORT_DIR / "stage7_scene_pack_report.md").write_text("# Stage 7 Scene Pack Report\n\n" + markdown_table(rows), encoding="utf-8")
    return payload


def load_scene_pack(dataset: str, scene_id: str) -> Dict | None:
    path = OUT_DIR / dataset / scene_id / "scene_pack.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"

