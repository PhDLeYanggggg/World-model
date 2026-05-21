from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.stage10_common import PEDESTRIAN_DATASETS, REPORT_DIR, available_world_state_sources, write_json, write_markdown_table


DATASET_CANDIDATES = {
    "sdd": {
        "display": "Stanford Drone Dataset",
        "official_source": "https://cvgl.stanford.edu/projects/uav_data/",
        "license": "non-commercial; user must accept dataset terms",
        "paths": ["data/stage10_raw/sdd", "data/sdd", "data/StanfordDroneDataset"],
    },
    "opentraj": {
        "display": "OpenTraj-compatible pedestrian datasets",
        "official_source": "https://github.com/crowdbotp/OpenTraj",
        "license": "dataset-specific; verify each source before use",
        "paths": ["data/stage10_raw/opentraj", "data/opentraj"],
    },
    "trajnet": {
        "display": "TrajNet++ bundled Stanford subset",
        "official_source": "https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge",
        "license": "dataset-specific; bundled original-data fallback",
        "paths": ["data/stage8p5_world_state/trajnet", "data/stage5b_world_state/trajnet"],
    },
    "eth_ucy": {
        "display": "ETH/UCY bundled fallback",
        "official_source": "https://icu.ee.ethz.ch/research/datsets.html",
        "license": "research dataset; verify original terms",
        "paths": ["data/stage8p5_world_state/eth_ucy", "data/stage5b_world_state/eth_ucy"],
    },
    "aerialmpt_long": {
        "display": "AerialMPT longer local sequences",
        "official_source": "local AerialMPT extraction if available",
        "license": "dataset-specific; local archive only",
        "paths": ["data/aerialmpt/extracted", "data/aerialmpt"],
    },
}


def audit_stage10_sources() -> List[Dict]:
    rows = []
    world_sources = set(available_world_state_sources())
    for name, info in DATASET_CANDIDATES.items():
        local_paths = [Path(p) for p in info["paths"]]
        local_existing = [str(p) for p in local_paths if p.exists()]
        world_csv = Path("data/stage8p5_world_state") / name / "world_state.csv"
        if world_csv.exists():
            rows.append(audit_world_state_source(name, info, world_csv, local_existing))
        elif name in world_sources:
            rows.append(audit_world_state_source(name, info, Path("data/stage8p5_world_state") / name / "world_state.csv", local_existing))
        else:
            rows.append(unavailable_row(name, info, local_existing))
    return rows


def audit_world_state_source(name: str, info: Dict, csv_path: Path, local_existing: List[str]) -> Dict:
    df = pd.read_csv(csv_path)
    track_lengths = df.groupby(["scene_id", "agent_id"])["frame_id"].nunique().to_numpy(dtype=int) if len(df) else np.asarray([], dtype=int)
    dt = pd.to_numeric(df.get("dt_s", pd.Series(dtype=float)), errors="coerce")
    coord = str(df["coordinate_unit"].iloc[0]) if "coordinate_unit" in df and len(df) else "unknown"
    metric = metric_status(coord, name)
    row = {
        "dataset_name": name,
        "official_source": info["official_source"],
        "license": info["license"],
        "download_status": "local_converted",
        "local_path_status": "verified_existing_world_state",
        "local_paths_found": local_existing,
        "loader_status": "loaded_stage8p5_world_state",
        "coordinate_unit": coord,
        "metric_or_pixel": metric,
        "homography_available": False,
        "scale_available": False,
        "scene_image_available": False,
        "annotation_available": bool(list((Path("data/stage8p5_annotations") / name).glob("*/scene_annotation.json"))),
        "agent_types": sorted(str(x) for x in df.get("agent_type", pd.Series(["unknown"])).dropna().astype(str).unique().tolist()),
        "fps_or_dt": float(np.nanmedian(dt.to_numpy(dtype=float))) if len(dt.dropna()) else None,
        "track_count": int(len(track_lengths)),
        "scene_count": int(df["scene_id"].nunique()) if "scene_id" in df else 0,
        "max_track_length": int(track_lengths.max()) if len(track_lengths) else 0,
        "mean_track_length": float(track_lengths.mean()) if len(track_lengths) else 0.0,
        "samples_t10": sample_count(track_lengths, 10),
        "samples_t25": sample_count(track_lengths, 25),
        "samples_t50": sample_count(track_lengths, 50),
        "samples_t100": sample_count(track_lengths, 100),
        "actual_verified_t50": bool(sample_count(track_lengths, 50) > 0),
        "actual_verified_t100": bool(sample_count(track_lengths, 100) > 0),
        "effective_seconds_t50": effective_seconds(dt, 50),
        "effective_seconds_t100": effective_seconds(dt, 100),
        "eligible_for_stage10": bool(name in PEDESTRIAN_DATASETS and sample_count(track_lengths, 10) > 0),
        "failure_reason_if_not_eligible": "",
    }
    if not row["eligible_for_stage10"]:
        row["failure_reason_if_not_eligible"] = "No local converted pedestrian/drone world_state with t+10 samples."
    if metric == "pixel-space":
        row["failure_reason_if_not_eligible"] = row["failure_reason_if_not_eligible"] or "Pixel-space only; metric world-model claims forbidden until homography/scale is added."
    return row


def unavailable_row(name: str, info: Dict, local_existing: List[str]) -> Dict:
    reason = "No local converted world_state found. Provide a local path or run a legal download/prepare step."
    return {
        "dataset_name": name,
        "official_source": info["official_source"],
        "license": info["license"],
        "download_status": "not_downloaded_or_not_prepared",
        "local_path_status": "found_unconverted_path" if local_existing else "missing",
        "local_paths_found": local_existing,
        "loader_status": "not_loaded",
        "coordinate_unit": "unknown",
        "metric_or_pixel": "unknown",
        "homography_available": False,
        "scale_available": False,
        "scene_image_available": False,
        "annotation_available": False,
        "agent_types": [],
        "fps_or_dt": None,
        "track_count": 0,
        "scene_count": 0,
        "max_track_length": 0,
        "mean_track_length": 0.0,
        "samples_t10": 0,
        "samples_t25": 0,
        "samples_t50": 0,
        "samples_t100": 0,
        "actual_verified_t50": False,
        "actual_verified_t100": False,
        "effective_seconds_t50": None,
        "effective_seconds_t100": None,
        "eligible_for_stage10": False,
        "failure_reason_if_not_eligible": reason,
    }


def metric_status(coordinate_unit: str, dataset: str) -> str:
    unit = str(coordinate_unit).lower()
    if unit in {"meter", "meters", "m"}:
        return "metric"
    if "pixel" in unit:
        return "pixel-space"
    if dataset == "trajnet":
        return "dataset-coordinate"
    return "non-metric-or-unknown"


def sample_count(track_lengths: np.ndarray, horizon: int, past: int = 10) -> int:
    if len(track_lengths) == 0:
        return 0
    return int(np.sum(track_lengths >= past + horizon))


def effective_seconds(dt: pd.Series, horizon: int):
    clean = pd.to_numeric(dt, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(np.nanmedian(clean.to_numpy(dtype=float)) * horizon)


def write_stage10_data_audit(rows: List[Dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "stage10_data_audit.json", rows)
    keys = [
        "dataset_name",
        "download_status",
        "local_path_status",
        "loader_status",
        "metric_or_pixel",
        "scene_count",
        "track_count",
        "max_track_length",
        "samples_t10",
        "samples_t25",
        "samples_t50",
        "samples_t100",
        "actual_verified_t50",
        "actual_verified_t100",
        "eligible_for_stage10",
        "failure_reason_if_not_eligible",
    ]
    write_markdown_table(REPORT_DIR / "stage10_data_audit.md", "Stage 10 Pedestrian/Drone Data Audit", [{k: r.get(k) for k in keys} for r in rows])


def main() -> None:
    rows = audit_stage10_sources()
    write_stage10_data_audit(rows)
    print(json.dumps({"datasets": len(rows), "eligible": sum(r["eligible_for_stage10"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
