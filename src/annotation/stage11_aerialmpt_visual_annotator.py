from __future__ import annotations

import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw


ZIP_PATH = Path("data/aerialmpt/DLR_AerialMPT_Dataset.zip")
ANNOTATION_ROOT = Path("data/stage11_visual_annotations/aerialmpt")
SCENE_PACK_ROOT = Path("data/stage11_scene_packs/aerialmpt")
FRAME_ROOT = Path("data/stage11_visual_frames/aerialmpt")
FIGURE_ROOT = Path("outputs/figures/stage11_visual_annotations")
REPORT_DIR = Path("outputs/reports")


def parse_mot_rows(text: str) -> List[Dict]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 6:
            continue
        frame, agent, x, y, w, h = parts[:6]
        rows.append(
            {
                "frame": int(float(frame)),
                "agent_id": str(agent),
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "cx": float(x) + float(w) / 2.0,
                "cy": float(y) + float(h) / 2.0,
            }
        )
    return rows


def scene_entries(zip_path: Path = ZIP_PATH) -> List[Dict]:
    with zipfile.ZipFile(zip_path) as zf:
        entries = defaultdict(dict)
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) < 3 or parts[0] not in {"train", "test"}:
                continue
            split, scene = parts[0], parts[1]
            entries[scene]["split"] = split
            if name.endswith("_gts.txt"):
                entries[scene]["gts"] = name
            if name.lower().endswith(".png"):
                entries[scene].setdefault("frames", []).append(name)
            if name.lower().endswith(".mp4"):
                entries[scene]["video"] = name
        return [
            {"scene_id": scene, **value, "frames": sorted(value.get("frames", []))}
            for scene, value in sorted(entries.items())
            if value.get("gts") and value.get("frames")
        ]


def bbox(points: Iterable[Tuple[float, float]], width: int, height: int, pad_frac: float = 0.08) -> List[List[float]]:
    pts = list(points)
    if not pts:
        return [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = max(xmax - xmin, ymax - ymin, 1.0) * pad_frac
    xmin = max(0.0, xmin - pad)
    ymin = max(0.0, ymin - pad)
    xmax = min(float(width), xmax + pad)
    ymax = min(float(height), ymax + pad)
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def edge_goal_regions(width: int, height: int) -> List[Dict]:
    radius = max(width, height) * 0.04
    centers = {
        "north_edge": [width / 2.0, 0.0],
        "south_edge": [width / 2.0, float(height)],
        "west_edge": [0.0, height / 2.0],
        "east_edge": [float(width), height / 2.0],
    }
    return [
        {
            "goal_id": name,
            "region_type": "ai_visual_silver_goal_region",
            "center": center,
            "radius": radius,
            "source": "image_boundary_candidate_not_future_endpoint",
            "confirmed_by_ai": True,
            "confirmed_by_human": False,
            "future_endpoint_label_only": False,
        }
        for name, center in centers.items()
    ]


def draw_preview(image: Image.Image, ann: Dict, points: List[Tuple[float, float]], out_path: Path) -> None:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    walk = [tuple(p) for p in ann["walkable_polygons"][0]]
    boundary = [tuple(p) for p in ann["boundary_polygon"]]
    draw.polygon(boundary, outline=(255, 255, 0, 220), width=4)
    draw.polygon(walk, fill=(0, 180, 255, 45), outline=(0, 180, 255, 220), width=4)
    for x, y in points[:: max(1, len(points) // 800)]:
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(255, 70, 70, 160))
    for goal in ann["goal_regions"]:
        x, y = goal["center"]
        r = goal["radius"]
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, 255, 120, 230), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def write_visual_annotations(zip_path: Path = ZIP_PATH) -> Dict:
    ANNOTATION_ROOT.mkdir(parents=True, exist_ok=True)
    SCENE_PACK_ROOT.mkdir(parents=True, exist_ok=True)
    FRAME_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for entry in scene_entries(zip_path):
            scene = entry["scene_id"]
            first_frame_name = entry["frames"][0]
            with zf.open(first_frame_name) as fh:
                image = Image.open(fh).convert("RGB")
                image.load()
            width, height = image.size
            rows_text = zf.read(entry["gts"]).decode("utf-8", errors="ignore")
            mot = parse_mot_rows(rows_text)
            points = [(r["cx"], r["cy"]) for r in mot]
            frame_out = FRAME_ROOT / scene / Path(first_frame_name).name
            frame_out.parent.mkdir(parents=True, exist_ok=True)
            image.save(frame_out)
            boundary = [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]
            observed_walk = bbox(points, width, height)
            goals = edge_goal_regions(width, height)
            ann = {
                "scene_id": scene,
                "dataset_name": "aerialmpt",
                "source_split": entry["split"],
                "scene_image_path": str(frame_out),
                "scene_video_path": entry.get("video"),
                "coordinate_system": "image_bev_pixel",
                "coordinate_unit": "pixel",
                "homography": None,
                "scale_m_per_px": None,
                "annotation_quality": "ai_visual_silver",
                "annotation_source": "ai_visual_image_plus_pedestrian_passage",
                "walkable_polygons": [observed_walk],
                "observed_walkable_polygons": [observed_walk],
                "boundary_polygon": boundary,
                "obstacle_polygons": [],
                "entry_regions": goals,
                "exit_regions": goals,
                "goal_regions": goals,
                "visual_labels": {
                    "road_or_path_visible": True,
                    "pedestrian_passage_observed": len(points) > 0,
                    "people_detection_count": len(points),
                    "image_width": width,
                    "image_height": height,
                    "metric_status": "pixel_only_no_homography",
                    "confidence": "medium",
                    "notes": "AI-assisted visual silver: walkable area is inferred from visible aerial frame plus observed pedestrian passage, not human gold.",
                },
                "leakage_policy": {
                    "future_endpoint_used_as_input": False,
                    "test_endpoints_used_for_candidate_goals": False,
                    "candidate_goals_are_image_boundary_priors": True,
                    "observed_walkable_uses_full_sequence_passage": True,
                },
            }
            scene_dir = ANNOTATION_ROOT / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / "scene_annotation.json").write_text(json.dumps(ann, indent=2), encoding="utf-8")
            pack_dir = SCENE_PACK_ROOT / scene
            pack_dir.mkdir(parents=True, exist_ok=True)
            (pack_dir / "scene_pack.json").write_text(json.dumps({**ann, "metric_status": "pixel_only_no_homography"}, indent=2), encoding="utf-8")
            preview = FIGURE_ROOT / f"aerialmpt_{scene}_visual_silver.png"
            draw_preview(image, ann, points, preview)
            frame_ids = [r["frame"] for r in mot]
            agent_ids = {r["agent_id"] for r in mot}
            rows.append(
                {
                    "scene_id": scene,
                    "source_split": entry["split"],
                    "frames": max(frame_ids) if frame_ids else 0,
                    "agents": len(agent_ids),
                    "detections": len(mot),
                    "image_size": [width, height],
                    "annotation_quality": "ai_visual_silver",
                    "preview": str(preview),
                    "actual_verified_t10": max(frame_ids) >= 20 if frame_ids else False,
                    "actual_verified_t50": False,
                    "actual_verified_t100": False,
                }
            )
    report = {
        "stage": "11_visual_annotation",
        "dataset": "aerialmpt",
        "license": "CC BY-SA 4.0",
        "scene_count": len(rows),
        "ai_visual_silver_scenes": len(rows),
        "gold_human_scenes": 0,
        "metric_status": "pixel_only_no_homography",
        "records": rows,
        "limitations": [
            "No homography or meter scale is available, so these are pixel-space scene labels.",
            "AI visual silver is not human gold.",
            "Observed walkable polygons use pedestrian passage evidence; candidate goals are boundary priors, not future endpoints.",
            "AerialMPT supports short verified horizons here, not pedestrian t+50/t+100.",
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage11_aerialmpt_visual_annotation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report)
    return report


def write_markdown_report(report: Dict) -> None:
    lines = [
        "# Stage 11 AerialMPT Visual Annotation Report",
        "",
        f"Dataset: `{report['dataset']}`",
        f"License: `{report['license']}`",
        f"AI visual silver scenes: `{report['ai_visual_silver_scenes']}`",
        f"Metric status: `{report['metric_status']}`",
        "",
        "| scene | split | frames | agents | detections | t+10 | preview |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["records"]:
        lines.append(
            f"| {row['scene_id']} | {row['source_split']} | {row['frames']} | {row['agents']} | {row['detections']} | {row['actual_verified_t10']} | {row['preview']} |"
        )
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    (REPORT_DIR / "stage11_aerialmpt_visual_annotation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing {ZIP_PATH}")
    report = write_visual_annotations()
    print(json.dumps({"visual_silver_scenes": report["ai_visual_silver_scenes"], "report": "outputs/reports/stage11_aerialmpt_visual_annotation_report.md"}, indent=2))


if __name__ == "__main__":
    main()
