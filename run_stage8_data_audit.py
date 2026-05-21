#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPORT_DIR = Path("outputs/reports")


def stage5b_dataset_summary(dataset: str) -> dict:
    root = Path("data/stage5b_episodes") / dataset
    if not root.exists():
        return {}
    tracks = set()
    scenes = set()
    max_len = 0
    samples = {10: 0, 25: 0, 50: 0, 100: 0}
    dt_values = []
    units = set()
    for p in root.glob("episode_*.npz"):
        data = np.load(p, allow_pickle=True)
        meta = json.loads(str(data["meta"].item()))
        states = data["states"]
        future = states.shape[0] - int(meta.get("past_horizon", 10))
        max_len = max(max_len, states.shape[0])
        tracks.add(str(meta.get("primary_agent_id", meta.get("episode_id"))))
        scenes.add(str(meta.get("scene_id", dataset)))
        dt_values.append(float(meta.get("dt_s", 0.0) or 0.0))
        units.add(str(meta.get("coordinate_unit", "unknown")))
        for h in samples:
            samples[h] += int(future >= h)
    return {
        "track_count": len(tracks),
        "scene_count": len(scenes),
        "max_track_length": max_len,
        "samples_t10": samples[10],
        "samples_t25": samples[25],
        "samples_t50": samples[50],
        "samples_t100": samples[100],
        "fps_or_dt": float(np.median(dt_values)) if dt_values else None,
        "coordinate_unit": sorted(units)[0] if units else "unknown",
    }


def run_audit(args) -> list[dict]:
    converted = {
        "full_trajnetplusplus": ("trajnet", Path(args.trajnet_root)),
        "full_eth_ucy": ("eth_ucy", Path(args.eth_ucy_root)),
        "aerialmpt_long": (None, Path(args.aerialmpt_root)),
        "stanford_drone_dataset": (None, Path(args.sdd_root) if args.sdd_root else None),
        "opentraj_supported_pedestrian": (None, Path(args.opentraj_root) if args.opentraj_root else None),
    }
    records = []
    for name, (stage5b_name, local_path) in converted.items():
        summary = stage5b_dataset_summary(stage5b_name) if stage5b_name else {}
        exists = bool(local_path and local_path.exists())
        sdd = name == "stanford_drone_dataset"
        unit = summary.get("coordinate_unit") or ("pixel" if sdd or name == "aerialmpt_long" else "unknown")
        t50 = bool(summary.get("samples_t50", 0) > 0)
        t100 = bool(summary.get("samples_t100", 0) > 0)
        record = {
            "dataset_name": name,
            "license": "SDD non-commercial research license; manual acceptance required" if sdd else "dataset-specific; verify original terms",
            "download_status": "local_or_user_path_verified" if exists else ("requires_manual_download" if sdd else "not_available_or_placeholder"),
            "local_path_status": "exists" if exists else "missing",
            "coordinate_unit": unit,
            "metric_or_pixel": "metric" if unit == "meter" else ("pixel" if unit == "pixel" else "dataset_coordinate"),
            "homography_available": False,
            "scene_image_available": exists,
            "annotation_available": False,
            "agent_types": ["pedestrian"] if "tgsim" not in name else ["traffic"],
            "fps_or_dt": summary.get("fps_or_dt"),
            "track_count": summary.get("track_count", 0),
            "scene_count": summary.get("scene_count", 0),
            "max_track_length": summary.get("max_track_length", 0),
            "samples_t10": summary.get("samples_t10", 0),
            "samples_t25": summary.get("samples_t25", 0),
            "samples_t50": summary.get("samples_t50", 0),
            "samples_t100": summary.get("samples_t100", 0),
            "actual_verified_t50": t50,
            "actual_verified_t100": t100,
            "effective_seconds_t50": (summary.get("fps_or_dt") or 0.0) * 50,
            "effective_seconds_t100": (summary.get("fps_or_dt") or 0.0) * 100,
            "whether_eligible_for_stage8_gate": bool(exists and (t50 or t100) and name not in {"aerialmpt_long"}),
            "notes": "No pedestrian/drone long-horizon claim unless actual_verified_t50 or actual_verified_t100 is true.",
        }
        records.append(record)
    return records


def write_outputs(records: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage8_data_audit.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# Stage 8 Data Audit", "", "| dataset | download | local | unit | image | homography | tracks | scenes | t10 | t25 | t50 | t100 | eligible | license |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        lines.append(
            f"| {r['dataset_name']} | {r['download_status']} | {r['local_path_status']} | {r['coordinate_unit']} | "
            f"{r['scene_image_available']} | {r['homography_available']} | {r['track_count']} | {r['scene_count']} | "
            f"{r['samples_t10']} | {r['samples_t25']} | {r['samples_t50']} | {r['samples_t100']} | "
            f"{r['whether_eligible_for_stage8_gate']} | {r['license']} |"
        )
    lines.append("\nIf SDD is not provided locally, Stage 8 cannot claim pedestrian/drone verified t+50/t+100.")
    (REPORT_DIR / "stage8_data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdd-root")
    parser.add_argument("--opentraj-root")
    parser.add_argument("--trajnet-root", default="data/stage5b_raw/trajnetplusplusdataset")
    parser.add_argument("--eth-ucy-root", default="data/stage5b_raw/trajnetplusplusdataset")
    parser.add_argument("--aerialmpt-root", default="data/aerialmpt/extracted")
    args = parser.parse_args()
    records = run_audit(args)
    write_outputs(records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

