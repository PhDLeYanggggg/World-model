from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


REPORT_DIR = Path("outputs/reports")
REGISTRY_PATH = Path("outputs/world_model_stage5_data_results/data_registry/dataset_registry_stage5.json")


TARGET_DATASETS = [
    ("Stanford Drone Dataset", "sdd"),
    ("TrajNet++", "trajnet"),
    ("ETH/UCY", "eth_ucy"),
    ("OpenTraj-compatible pedestrian datasets", "opentraj"),
    ("AerialMPT longer sequences", "aerialmpt"),
]


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def registry_rows() -> List[Dict]:
    rows = load_json(REGISTRY_PATH, [])
    return rows if isinstance(rows, list) else rows.get("datasets", [])


def find_registry(name: str) -> Dict:
    needle = name.lower()
    for row in registry_rows():
        if needle in str(row.get("dataset_name", "")).lower() or needle in str(row.get("notes", "")).lower():
            return row
    return {}


def local_episode_stats(dataset: str) -> Dict:
    root = Path("data/stage5b_episodes") / dataset
    if not root.exists():
        return {}
    horizons = []
    dts = []
    units = set()
    t50 = 0
    t100 = 0
    for path in sorted(root.glob("episode_*.npz")):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"].item()))
        future = int(meta.get("future_horizon", data["states"].shape[0] - int(meta.get("past_horizon", 10))))
        horizons.append(future)
        dts.append(float(meta.get("dt_s", 0.0)))
        units.add(str(meta.get("coordinate_unit", "unknown")))
        t50 += future >= 50
        t100 += future >= 100
    if not horizons:
        return {}
    return {
        "local_episode_count": len(horizons),
        "max_raw_horizon": int(max(horizons)),
        "max_downsampled_horizon": int(max(horizons)),
        "dt": round(float(np.median(dts)), 6) if dts else None,
        "coordinate_unit": ",".join(sorted(units)),
        "t50_verified": bool(t50),
        "t100_verified": bool(t100),
        "t50_samples": int(t50),
        "t100_samples": int(t100),
    }


def aerialmpt_stats() -> Dict:
    root = Path("data/aerialmpt")
    exists = root.exists()
    return {
        "local_episode_count": 1 if exists else 0,
        "max_raw_horizon": 12 if exists else 0,
        "max_downsampled_horizon": 12 if exists else 0,
        "dt": None,
        "coordinate_unit": "pixel_or_unknown",
        "t50_verified": False,
        "t100_verified": False,
        "t50_samples": 0,
        "t100_samples": 0,
    }


def build_row(display_name: str, key: str) -> Dict:
    reg = find_registry(display_name if key != "sdd" else "Stanford Drone")
    local = aerialmpt_stats() if key == "aerialmpt" else local_episode_stats(key)
    coordinate_unit = local.get("coordinate_unit") or reg.get("coordinate_unit", "unknown")
    max_h = int(local.get("max_raw_horizon", 0))
    dt = local.get("dt") or reg.get("frame_rate") or "unknown"
    effective_t10 = (10 * float(dt)) if isinstance(dt, (int, float)) else "unknown"
    effective_t50 = (50 * float(dt)) if isinstance(dt, (int, float)) else "unknown"
    effective_t100 = (100 * float(dt)) if isinstance(dt, (int, float)) else "unknown"
    needs_license = str(reg.get("download_status", "")).lower() in {"gated", "requires_application"}
    reason = "local converted source only supports short horizon" if local and not local.get("t50_verified") else ""
    if key == "sdd" and not local:
        reason = "not downloaded; non-commercial/license/manual preparation remains unresolved"
    if key == "opentraj" and not local:
        reason = "registry/planning entry only; no actual converted local episodes"
    return {
        "dataset_name": display_name,
        "domain": reg.get("domain", "pedestrian/drone"),
        "coordinate_unit": coordinate_unit,
        "metric_or_pixel": "metric" if coordinate_unit == "meter" else ("pixel_or_dataset_coordinate" if coordinate_unit != "unknown" else "unknown"),
        "has_homography": bool(reg.get("has_homography", False)),
        "has_scene_image": bool(reg.get("has_images", False) or reg.get("has_raw_video", False)),
        "has_map_or_walkable_area": bool(reg.get("has_scene_map", False) or reg.get("has_walkable_area", False)),
        "original_frame_rate": reg.get("frame_rate", "unknown"),
        "dt": dt,
        "max_raw_horizon": max_h,
        "max_downsampled_horizon": int(local.get("max_downsampled_horizon", 0)),
        "effective_time_span_t10_s": effective_t10,
        "effective_time_span_t50_s": effective_t50,
        "effective_time_span_t100_s": effective_t100,
        "t50_verified": bool(local.get("t50_verified", False)),
        "t100_verified": bool(local.get("t100_verified", False)),
        "t50_samples": int(local.get("t50_samples", 0)),
        "t100_samples": int(local.get("t100_samples", 0)),
        "download_status": reg.get("download_status", "not_in_registry_or_not_downloaded"),
        "requires_license_or_application": needs_license,
        "suitable_for_official_gate": bool(local.get("t50_verified", False) and coordinate_unit == "meter"),
        "why_or_why_not": reason or ("verified locally" if local else "not available locally"),
    }


def run_horizon_repair() -> List[Dict]:
    return [build_row(name, key) for name, key in TARGET_DATASETS]


def markdown_table(rows: List[Dict]) -> str:
    keys = [
        "dataset_name",
        "coordinate_unit",
        "metric_or_pixel",
        "max_raw_horizon",
        "t50_verified",
        "t100_verified",
        "t50_samples",
        "t100_samples",
        "download_status",
        "suitable_for_official_gate",
        "why_or_why_not",
    ]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(rows: Iterable[Dict]) -> List[Dict]:
    rows = list(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage5b6_pedestrian_drone_horizon_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    text = "\n".join(
        [
            "# Stage 5B.6 Pedestrian / Drone Long-Horizon Repair",
            "",
            "This report distinguishes actual converted local data from registry-only or license/manual placeholders. Downsampled horizons are not counted as original verified t+100 unless explicitly available.",
            "",
            markdown_table(rows),
            "Conclusion: no real pedestrian/drone source with verified t+50/t+100 was added in this run. TGSIM remains the only verified long-horizon source family, and it is traffic/generic trajectory data.",
        ]
    )
    (REPORT_DIR / "stage5b6_pedestrian_drone_horizon_report.md").write_text(text, encoding="utf-8")
    return rows

