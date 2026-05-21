from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.scene.stage10_route_graph_builder import build_stage10_route_graph
from src.scene.stage10_sdf_builder import analytic_sdf_summary
from src.stage10_common import REPORT_DIR, ensure_dir, is_human_annotation_quality, is_official_annotation_quality, write_json, write_markdown_table


OUT_DIR = Path("data/stage10_scene_packs")


def build_stage10_scene_packs(annotation_root: str | Path = "data/stage10_annotations") -> Dict:
    packs = []
    for path in sorted(Path(annotation_root).glob("*/*/scene_annotation.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        pack = pack_from_annotation(annotation)
        out_dir = ensure_dir(OUT_DIR / annotation["dataset_name"] / annotation["scene_id"])
        (out_dir / "scene_pack.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
        packs.append(pack)
    payload = summarize(packs)
    write_scene_pack_report(payload)
    return payload


def pack_from_annotation(annotation: Dict) -> Dict:
    sdf = analytic_sdf_summary(annotation)
    metric = "metric" if annotation.get("coordinate_unit") == "meter" else ("weak_metric" if annotation.get("scale_m_per_px") else "non_metric_or_dataset_coordinate")
    return {
        "scene_id": annotation["scene_id"],
        "dataset_name": annotation["dataset_name"],
        "annotation_quality": annotation.get("annotation_quality", "inferred_only"),
        "coordinate_unit": annotation.get("coordinate_unit", "unknown"),
        "metric_status": metric,
        "homography": annotation.get("homography"),
        "scale_m_per_px": annotation.get("scale_m_per_px"),
        "walkable_mask_or_polygon": annotation.get("walkable_polygons", []),
        "obstacle_polygons": annotation.get("obstacle_polygons", []),
        "boundary_polygon": annotation.get("boundary_polygon", []),
        "entry_regions": annotation.get("entry_regions", []),
        "exit_regions": annotation.get("exit_regions", []),
        "goal_regions": annotation.get("goal_regions", []),
        "route_corridors": annotation.get("route_corridors", []),
        "walkable_sdf": sdf["walkable_sdf"],
        "obstacle_sdf": sdf["obstacle_sdf"],
        "goal_distance_fields": sdf["goal_distance_fields"],
        "route_distance_fields": "available_if_route_graph_available",
        "route_graph": build_stage10_route_graph(annotation),
        "annotation_source": annotation.get("annotator_id") or annotation.get("annotator") or "unknown",
        "annotation_version": annotation.get("version", "stage10_v1"),
        "human_confirmed": is_human_annotation_quality(annotation.get("annotation_quality")),
        "leakage_policy": annotation.get("leakage_policy", {}),
    }


def summarize(packs: List[Dict]) -> Dict:
    return {
        "stage": "10",
        "scene_packs": packs,
        "gold_human_scenes": sum(p["annotation_quality"] == "gold_human" for p in packs),
        "silver_human_confirmed_scenes": sum(p["annotation_quality"] == "silver_human_confirmed" for p in packs),
        "silver_rule_confirmed_scenes": sum(p["annotation_quality"] == "silver_rule_confirmed" for p in packs),
        "inferred_only_scenes": sum(p["annotation_quality"] == "inferred_only" for p in packs),
        "scenes_with_homography": sum(p["homography"] is not None for p in packs),
        "scenes_with_metric_scale": sum(p["metric_status"] in {"metric", "weak_metric"} for p in packs),
        "scenes_with_walkable": sum(bool(p["walkable_mask_or_polygon"]) for p in packs),
        "scenes_with_goals": sum(bool(p["goal_regions"]) for p in packs),
        "scenes_with_obstacles": sum(bool(p["obstacle_polygons"]) for p in packs),
        "scenes_eligible_for_official_goalbench": sum(is_official_annotation_quality(p["annotation_quality"]) for p in packs),
        "scenes_eligible_for_stage11_training": sum(p["annotation_quality"] in {"gold_human", "silver_human_confirmed"} for p in packs),
    }


def write_scene_pack_report(payload: Dict) -> None:
    summary = {k: v for k, v in payload.items() if k != "scene_packs"}
    write_json(REPORT_DIR / "stage10_scene_pack_report.json", summary)
    rows = [
        {
            "dataset_name": p["dataset_name"],
            "scene_id": p["scene_id"],
            "annotation_quality": p["annotation_quality"],
            "metric_status": p["metric_status"],
            "goals": len(p["goal_regions"]),
            "obstacles": len(p["obstacle_polygons"]),
            "human_confirmed": p["human_confirmed"],
        }
        for p in payload["scene_packs"]
    ]
    extra = [f"{k}: {v}" for k, v in summary.items() if k != "stage"]
    write_markdown_table(REPORT_DIR / "stage10_scene_pack_report.md", "Stage 10 Scene Pack Report", rows, extra)


def load_stage10_scene_pack(dataset: str, scene_id: str) -> Dict | None:
    path = OUT_DIR / dataset / scene_id / "scene_pack.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    payload = build_stage10_scene_packs()
    print(json.dumps({k: v for k, v in payload.items() if k != "scene_packs"}, indent=2))


if __name__ == "__main__":
    main()
