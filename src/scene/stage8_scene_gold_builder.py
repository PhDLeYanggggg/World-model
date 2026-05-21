from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.annotation.annotation_export import load_annotation
from src.scene.stage8_route_graph import build_route_graph_from_annotation
from src.scene.walkable_area_builder import boundary_summary


OUT_DIR = Path("data/scene_gold_packs")
REPORT_DIR = Path("outputs/reports")


def build_scene_gold_packs(annotation_root: str | Path = "data/stage8_annotations") -> Dict:
    packs = []
    for ann_path in Path(annotation_root).glob("*/*/scene_annotation.json"):
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        pack = pack_from_annotation(ann)
        out_dir = OUT_DIR / ann["dataset_name"] / ann["scene_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "scene_gold_pack.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
        packs.append(pack)
    return summarize(packs)


def pack_from_annotation(ann: Dict) -> Dict:
    boundary = boundary_summary(ann["boundary_polygon"])
    goals = []
    for goal in ann.get("goal_regions", []):
        goals.append(
            {
                "goal_id": goal.get("goal_id"),
                "goal_type": goal.get("region_type", "inferred_goal_region"),
                "center": goal.get("center"),
                "radius": goal.get("radius", 1.0),
                "confirmed_by_human": goal.get("confirmed_by_human", False),
            }
        )
    quality = ann.get("annotation_quality", "inferred_only")
    return {
        "scene_id": ann["scene_id"],
        "dataset_name": ann["dataset_name"],
        "coordinate_unit": ann["coordinate_unit"],
        "image": ann.get("image_path"),
        "homography": ann.get("homography"),
        "scale_m_per_px": ann.get("scale_m_per_px"),
        "walkable_polygons": ann.get("walkable_polygons", []),
        "obstacle_polygons": ann.get("obstacle_polygons", []),
        "boundary_polygon": ann.get("boundary_polygon", []),
        "boundary_summary": boundary,
        "entry_regions": ann.get("entry_regions", []),
        "exit_regions": ann.get("exit_regions", []),
        "goal_regions": goals,
        "route_graph": build_route_graph_from_annotation(ann),
        "walkable_sdf": "analytic_boundary_distance",
        "obstacle_sdf": "not_available" if not ann.get("obstacle_polygons") else "polygon_distance",
        "goal_distance_fields": "computed_on_demand",
        "annotation_quality": quality,
        "whether_metric": ann["coordinate_unit"] == "meter" or ann.get("scale_m_per_px") is not None,
        "whether_manually_annotated": quality in {"gold", "silver"},
        "eligible_for_official_stage8_benchmark": quality in {"gold", "silver"} or bool(goals),
    }


def summarize(packs: List[Dict]) -> Dict:
    summary = {
        "stage": "8",
        "scene_packs": packs,
        "total_scene_packs": len(packs),
        "gold_scenes": sum(p["annotation_quality"] == "gold" for p in packs),
        "silver_scenes": sum(p["annotation_quality"] == "silver" for p in packs),
        "inferred_only_scenes": sum(p["annotation_quality"] == "inferred_only" for p in packs),
        "scenes_with_homography": sum(p["homography"] is not None for p in packs),
        "scenes_with_metric_scale": sum(p["scale_m_per_px"] is not None or p["coordinate_unit"] == "meter" for p in packs),
        "scenes_with_walkable_polygon": sum(bool(p["walkable_polygons"]) for p in packs),
        "scenes_with_exit_goal_regions": sum(bool(p["goal_regions"] or p["exit_regions"]) for p in packs),
        "official_stage8_eligible_scenes": sum(p["eligible_for_official_stage8_benchmark"] for p in packs),
    }
    return summary


def write_report(payload: Dict) -> Dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage8_scene_gold_report.json").write_text(json.dumps({k: v for k, v in payload.items() if k != "scene_packs"}, indent=2), encoding="utf-8")
    rows = [
        {
            "dataset": p["dataset_name"],
            "scene": p["scene_id"],
            "quality": p["annotation_quality"],
            "goals": len(p["goal_regions"]),
            "metric": p["whether_metric"],
            "manual": p["whether_manually_annotated"],
            "eligible": p["eligible_for_official_stage8_benchmark"],
        }
        for p in payload["scene_packs"]
    ]
    lines = ["# Stage 8 Scene-Gold Pack Report", "", "| dataset | scene | quality | goals | metric | manual | eligible |", "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['dataset']} | {r['scene']} | {r['quality']} | {r['goals']} | {r['metric']} | {r['manual']} | {r['eligible']} |" for r in rows]
    lines += [
        "",
        f"gold scenes: {payload['gold_scenes']}",
        f"silver scenes: {payload['silver_scenes']}",
        f"inferred-only scenes: {payload['inferred_only_scenes']}",
        f"scenes with homography: {payload['scenes_with_homography']}",
        f"scenes with metric scale: {payload['scenes_with_metric_scale']}",
    ]
    (REPORT_DIR / "stage8_scene_gold_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def load_scene_gold_pack(dataset_name: str, scene_id: str) -> Dict | None:
    p = OUT_DIR / dataset_name / scene_id / "scene_gold_pack.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

