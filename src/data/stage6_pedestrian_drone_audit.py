from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


REPORT_DIR = Path("outputs/reports")
REGISTRY = Path("outputs/world_model_stage5_data_results/data_registry/dataset_registry_stage5.json")


SOURCE_KEYS = [
    ("Stanford Drone Dataset", "sdd", "drone"),
    ("OpenTraj-supported pedestrian datasets", "opentraj", "pedestrian"),
    ("full TrajNet++", "trajnet", "pedestrian"),
    ("full ETH/UCY", "eth_ucy", "pedestrian"),
    ("UCY original crowd", "ucy", "pedestrian"),
    ("AerialMPT longer sequences", "aerialmpt", "drone"),
]


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def registry_rows() -> List[Dict]:
    rows = load_json(REGISTRY, [])
    return rows if isinstance(rows, list) else rows.get("datasets", [])


def find_registry(display_name: str) -> Dict:
    needle = display_name.lower().replace("full ", "").replace("original ", "")
    for row in registry_rows():
        hay = " ".join(str(row.get(k, "")) for k in ["dataset_name", "notes", "official_url"]).lower()
        if needle in hay or any(token in hay for token in needle.split("/") if len(token) > 2):
            return row
    return {}


def episode_stats(key: str) -> Dict:
    if key == "aerialmpt":
        root = Path("data/aerialmpt")
        return {
            "actual_downloaded_or_user_path_verified": root.exists(),
            "coordinate_unit": "pixel_or_unknown",
            "max_raw_horizon": 12 if root.exists() else 0,
            "horizon_counts": {"10": int(root.exists()), "25": 0, "50": 0, "100": 0},
            "dt": None,
        }
    root = Path("data/stage5b_episodes") / key
    if not root.exists():
        return {"actual_downloaded_or_user_path_verified": False, "coordinate_unit": "unknown", "max_raw_horizon": 0, "horizon_counts": {"10": 0, "25": 0, "50": 0, "100": 0}, "dt": None}
    horizons = []
    dts = []
    units = set()
    for path in root.glob("episode_*.npz"):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"].item()))
        future = int(meta.get("future_horizon", data["states"].shape[0] - int(meta.get("past_horizon", 10))))
        horizons.append(future)
        dts.append(float(meta.get("dt_s", 0.0)))
        units.add(str(meta.get("coordinate_unit", "unknown")))
    counts = {str(h): sum(1 for v in horizons if v >= h) for h in [10, 25, 50, 100]}
    return {
        "actual_downloaded_or_user_path_verified": bool(horizons),
        "coordinate_unit": ",".join(sorted(units)) if units else "unknown",
        "max_raw_horizon": int(max(horizons)) if horizons else 0,
        "horizon_counts": counts,
        "dt": round(float(np.median(dts)), 6) if dts else None,
    }


def row_for(display_name: str, key: str, domain: str) -> Dict:
    reg = find_registry(display_name)
    stats = episode_stats(key)
    unit = stats["coordinate_unit"] or reg.get("coordinate_unit", "unknown")
    metric = unit == "meter" or bool(reg.get("has_metric_coordinates", False) and unit not in {"dataset_coordinate", "pixel_or_unknown", "unknown"})
    dt = stats["dt"]
    effective = {str(h): (round(h * dt, 6) if isinstance(dt, (int, float)) and dt else "unknown") for h in [10, 25, 50, 100]}
    return {
        "dataset_name": display_name,
        "dataset_key": key,
        "domain": domain,
        "actual_downloaded_or_user_path_verified": stats["actual_downloaded_or_user_path_verified"],
        "license": reg.get("license", "unknown_or_not_registered"),
        "coordinate_unit": unit,
        "pixel_or_metric": "metric" if metric else ("pixel/dataset_coordinate" if unit != "unknown" else "unknown"),
        "homography_available": bool(reg.get("has_homography", False)),
        "scene_image_available": bool(reg.get("has_images", False) or reg.get("has_raw_video", False)),
        "max_raw_horizon": stats["max_raw_horizon"],
        "max_verified_t10": stats["horizon_counts"]["10"],
        "max_verified_t25": stats["horizon_counts"]["25"],
        "max_verified_t50": stats["horizon_counts"]["50"],
        "max_verified_t100": stats["horizon_counts"]["100"],
        "effective_seconds": effective,
        "actual_verified_t50_source": bool(stats["horizon_counts"]["50"]),
        "actual_verified_t100_source": bool(stats["horizon_counts"]["100"]),
        "eligible_for_pedestrian_drone_long_horizon_gate": bool(stats["horizon_counts"]["50"]) and domain in {"pedestrian", "drone"},
        "download_status": reg.get("download_status", "not_registered_or_not_downloaded"),
        "notes": "No pedestrian/drone long-horizon world model claim is allowed." if not stats["horizon_counts"]["50"] else "verified local long horizon exists",
    }


def run_audit() -> List[Dict]:
    return [row_for(*item) for item in SOURCE_KEYS]


def markdown_table(rows: List[Dict]) -> str:
    keys = [
        "dataset_name",
        "actual_downloaded_or_user_path_verified",
        "license",
        "coordinate_unit",
        "pixel_or_metric",
        "homography_available",
        "max_raw_horizon",
        "max_verified_t50",
        "max_verified_t100",
        "eligible_for_pedestrian_drone_long_horizon_gate",
        "notes",
    ]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(rows: Iterable[Dict]) -> List[Dict]:
    rows = list(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage6_pedestrian_drone_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    text = "\n".join(
        [
            "# Stage 6 Pedestrian / Drone Long-Horizon Audit",
            "",
            "This audit counts only actual local converted/user-path verified data. Registry-estimated t+100 does not count.",
            "",
            markdown_table(rows),
            "No pedestrian/drone long-horizon world model claim is allowed unless at least one actual verified t+50/t+100 source appears here.",
        ]
    )
    (REPORT_DIR / "stage6_pedestrian_drone_audit.md").write_text(text, encoding="utf-8")
    return rows

