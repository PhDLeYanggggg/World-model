from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.sdd_loader import load_sdd_trajectories


REPORT_DIR = Path("outputs/reports")
WORLD_OUT = Path("data/stage8p5_world_state")


def copy_existing_world_state(dataset: str) -> Dict:
    src = Path("data/stage5b_world_state") / dataset / "world_state.csv"
    meta_src = Path("data/stage5b_world_state") / dataset / "metadata.json"
    out_dir = WORLD_OUT / dataset
    if not src.exists():
        return {"dataset_name": dataset, "download_status": "not_available", "failure_reason_if_not_eligible": "stage5b world_state missing"}
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_dir / "world_state.csv")
    if meta_src.exists():
        shutil.copy2(meta_src, out_dir / "metadata.json")
    return summarize_world_state(dataset, out_dir / "world_state.csv", source="existing_stage5b_conversion")


def convert_sdd_if_available(root: str | None, quick: bool) -> Dict:
    if not root:
        return {
            "dataset_name": "sdd",
            "license": "Stanford Drone Dataset non-commercial research license; manual agreement required",
            "download_status": "requires_manual_download",
            "local_path_status": "missing",
            "coordinate_unit": "pixel",
            "metric_or_pixel": "pixel",
            "homography_available": False,
            "scene_image_available": False,
            "annotation_available": False,
            "agent_types": ["pedestrian", "biker", "cart", "car", "bus", "skater", "unknown"],
            "fps_or_dt": "unknown_without_dataset_metadata",
            "track_count": 0,
            "scene_count": 0,
            "max_track_length": 0,
            "samples_t10": 0,
            "samples_t25": 0,
            "samples_t50": 0,
            "samples_t100": 0,
            "actual_verified_t50": False,
            "actual_verified_t100": False,
            "effective_seconds_t50": "unknown",
            "effective_seconds_t100": "unknown",
            "eligible_for_stage8p5_gate": False,
            "failure_reason_if_not_eligible": "no local SDD path supplied",
        }
    try:
        table, meta = load_sdd_trajectories(root, quick=quick)
    except Exception as exc:  # noqa: BLE001
        return {
            "dataset_name": "sdd",
            "license": "Stanford Drone Dataset non-commercial research license; manual agreement required",
            "download_status": "local_path_failed",
            "local_path_status": "exists_but_unparsed",
            "coordinate_unit": "pixel",
            "metric_or_pixel": "pixel",
            "homography_available": False,
            "scene_image_available": Path(root).exists(),
            "annotation_available": False,
            "agent_types": [],
            "fps_or_dt": "unknown",
            "track_count": 0,
            "scene_count": 0,
            "max_track_length": 0,
            "samples_t10": 0,
            "samples_t25": 0,
            "samples_t50": 0,
            "samples_t100": 0,
            "actual_verified_t50": False,
            "actual_verified_t100": False,
            "effective_seconds_t50": "unknown",
            "effective_seconds_t100": "unknown",
            "eligible_for_stage8p5_gate": False,
            "failure_reason_if_not_eligible": str(exc),
        }
    out_dir = WORLD_OUT / "sdd"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "world_state.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summarize_world_state("sdd", out_dir / "world_state.csv", source="local_sdd_conversion")


def summarize_world_state(dataset: str, csv_path: Path, source: str) -> Dict:
    df = pd.read_csv(csv_path)
    unit = str(df["coordinate_unit"].iloc[0]) if "coordinate_unit" in df and len(df) else "unknown"
    track_lengths = df.groupby(["scene_id", "agent_id"])["frame_id"].nunique() if len(df) else pd.Series(dtype=float)
    samples = {h: int((track_lengths >= h + 10).sum()) for h in [10, 25, 50, 100]}
    dt = float(np.nanmedian(df["dt_s"].to_numpy(dtype=float))) if "dt_s" in df and len(df) else None
    agent_types = sorted(str(x) for x in df.get("agent_type", pd.Series(["unknown"])).dropna().unique().tolist()) if len(df) else []
    is_ped = dataset in {"trajnet", "eth_ucy", "sdd", "opentraj", "aerialmpt_long"}
    return {
        "dataset_name": dataset,
        "license": license_for(dataset),
        "download_status": "actual_loaded_and_converted",
        "local_path_status": "converted_world_state_exists",
        "coordinate_unit": unit,
        "metric_or_pixel": "metric" if unit == "meter" else ("pixel" if unit == "pixel" else "dataset_coordinate"),
        "homography_available": False,
        "scene_image_available": dataset == "sdd",
        "annotation_available": dataset == "sdd",
        "agent_types": agent_types,
        "fps_or_dt": dt if dt is not None else "unknown",
        "track_count": int(track_lengths.shape[0]),
        "scene_count": int(df["scene_id"].nunique()) if len(df) else 0,
        "max_track_length": int(track_lengths.max()) if len(track_lengths) else 0,
        "samples_t10": samples[10],
        "samples_t25": samples[25],
        "samples_t50": samples[50],
        "samples_t100": samples[100],
        "actual_verified_t50": bool(samples[50] > 0),
        "actual_verified_t100": bool(samples[100] > 0),
        "effective_seconds_t50": "unknown_without_verified_frame_rate" if dt is None else dt * 50,
        "effective_seconds_t100": "unknown_without_verified_frame_rate" if dt is None else dt * 100,
        "eligible_for_stage8p5_gate": bool(is_ped and len(df) and samples[10] > 0),
        "failure_reason_if_not_eligible": "" if is_ped and len(df) and samples[10] > 0 else "not a loaded pedestrian/drone source with usable tracks",
        "source": source,
    }


def license_for(dataset: str) -> str:
    if dataset == "sdd":
        return "SDD non-commercial research license; manual agreement required"
    if dataset == "eth_ucy":
        return "ETH/UCY academic terms; verify before redistribution"
    if dataset == "trajnet":
        return "TrajNet++ / source dataset terms; verify before redistribution"
    return "dataset-specific; verify original terms"


def write_outputs(records: List[Dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage8p5_data_audit.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    keys = [
        "dataset_name",
        "download_status",
        "coordinate_unit",
        "track_count",
        "scene_count",
        "samples_t10",
        "samples_t25",
        "samples_t50",
        "samples_t100",
        "actual_verified_t50",
        "actual_verified_t100",
        "eligible_for_stage8p5_gate",
        "failure_reason_if_not_eligible",
    ]
    lines = ["# Stage 8.5 Data Audit", "", "| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in records:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    lines.append("\nTraffic datasets are intentionally excluded from the pedestrian/drone Stage 8.5 gate.")
    (REPORT_DIR / "stage8p5_data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdd-root")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    WORLD_OUT.mkdir(parents=True, exist_ok=True)
    records = [copy_existing_world_state("trajnet"), copy_existing_world_state("eth_ucy"), convert_sdd_if_available(args.sdd_root, quick=args.quick)]
    write_outputs(records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
