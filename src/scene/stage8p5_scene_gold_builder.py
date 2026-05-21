from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.scene.stage8p5_route_graph_builder import build_route_graph
from src.scene.stage8p5_sdf_builder import goal_distance_fields


OUT_DIR = Path("data/stage8p5_scene_gold_packs")
REPORT_DIR = Path("outputs/reports")


def build_stage8p5_scene_gold_packs(annotation_root: str | Path = "data/stage8p5_annotations") -> Dict:
    packs = []
    for path in sorted(Path(annotation_root).glob("*/*/scene_annotation.json")):
        ann = json.loads(path.read_text(encoding="utf-8"))
        pack = pack_from_annotation(ann)
        out_dir = OUT_DIR / ann["dataset_name"] / ann["scene_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "scene_gold_pack.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
        packs.append(pack)
    return summarize(packs)


def pack_from_annotation(ann: Dict) -> Dict:
    metric = "metric" if ann.get("coordinate_unit") == "meter" else ("weak_metric" if ann.get("scale_m_per_px") else "non_metric")
    return {
        "scene_id": ann["scene_id"],
        "dataset_name": ann["dataset_name"],
        "annotation_quality": ann.get("annotation_quality", "inferred_only"),
        "coordinate_unit": ann.get("coordinate_unit", "unknown"),
        "metric_status": metric,
        "homography": ann.get("homography"),
        "scale_m_per_px": ann.get("scale_m_per_px"),
        "walkable_mask_or_polygon": ann.get("walkable_polygons", []),
        "obstacle_polygons": ann.get("obstacle_polygons", []),
        "boundary_polygon": ann.get("boundary_polygon", []),
        "entry_regions": ann.get("entry_regions", []),
        "exit_regions": ann.get("exit_regions", []),
        "goal_regions": ann.get("goal_regions", []),
        "route_corridors": ann.get("route_corridors", []),
        "walkable_sdf": "analytic_boundary_distance",
        "obstacle_sdf": "not_available" if not ann.get("obstacle_polygons") else "polygon_distance",
        "goal_distance_fields": goal_distance_fields(ann.get("goal_regions", [])),
        "route_distance_fields": "route_graph_shortest_path_if_corridors_available",
        "route_graph": build_route_graph(ann),
        "annotation_source": ann.get("annotator", "unknown"),
        "leakage_policy": ann.get("leakage_policy", {}),
    }


def summarize(packs: List[Dict]) -> Dict:
    return {
        "stage": "8.5",
        "scene_packs": packs,
        "number_of_gold_scenes": sum(p["annotation_quality"] == "gold" for p in packs),
        "number_of_silver_scenes": sum(p["annotation_quality"] == "silver" for p in packs),
        "number_of_inferred_only_scenes": sum(p["annotation_quality"] == "inferred_only" for p in packs),
        "scenes_with_homography": sum(p["homography"] is not None for p in packs),
        "scenes_with_metric_scale": sum(p["metric_status"] in {"metric", "weak_metric"} for p in packs),
        "scenes_with_walkable": sum(bool(p["walkable_mask_or_polygon"]) for p in packs),
        "scenes_with_goals": sum(bool(p["goal_regions"]) for p in packs),
        "scenes_with_obstacles": sum(bool(p["obstacle_polygons"]) for p in packs),
        "scenes_eligible_for_official_goalbench": sum(p["annotation_quality"] in {"gold", "silver"} for p in packs),
    }


def write_scene_gold_pack_report(payload: Dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage8p5_scene_gold_pack_report.json").write_text(json.dumps({k: v for k, v in payload.items() if k != "scene_packs"}, indent=2), encoding="utf-8")
    rows = [
        {
            "dataset": p["dataset_name"],
            "scene": p["scene_id"],
            "quality": p["annotation_quality"],
            "metric_status": p["metric_status"],
            "goals": len(p["goal_regions"]),
            "obstacles": len(p["obstacle_polygons"]),
        }
        for p in payload["scene_packs"]
    ]
    lines = ["# Stage 8.5 Scene-Gold Pack Report", "", "| dataset | scene | quality | metric_status | goals | obstacles |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append("| " + " | ".join(str(v) for v in r.values()) + " |")
    for key in [
        "number_of_gold_scenes",
        "number_of_silver_scenes",
        "number_of_inferred_only_scenes",
        "scenes_with_homography",
        "scenes_with_metric_scale",
        "scenes_with_walkable",
        "scenes_with_goals",
        "scenes_with_obstacles",
        "scenes_eligible_for_official_goalbench",
    ]:
        lines.append(f"{key}: {payload[key]}")
    (REPORT_DIR / "stage8p5_scene_gold_pack_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_stage8p5_scene_pack(dataset: str, scene_id: str) -> Dict | None:
    p = OUT_DIR / dataset / scene_id / "scene_gold_pack.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
