#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from src.data.aerialmpt_long_scene_loader import inspect_aerialmpt_long_root
from src.data.full_eth_ucy_scene_loader import inspect_full_eth_ucy_scene_root
from src.data.full_trajnet_scene_loader import inspect_full_trajnet_scene_root
from src.data.opentraj_scene_loader import inspect_opentraj_root
from src.data.sdd_scene_loader import inspect_sdd_scene_root


REPORT_DIR = Path("outputs/reports")


def converted_dataset_horizons() -> dict:
    out = {}
    for dataset_dir in Path("data/stage5b_episodes").glob("*"):
        if not dataset_dir.is_dir():
            continue
        summary_path = dataset_dir / "episode_summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        horizons = set()
        dt_values = []
        for ep_path in dataset_dir.glob("episode_*.npz"):
            import numpy as np

            data = np.load(ep_path, allow_pickle=True)
            meta = json.loads(str(data["meta"].item()))
            future = int(data["states"].shape[0] - int(meta.get("past_horizon", 10)))
            dt_values.append(float(meta.get("dt_s", 0.0) or 0.0))
            for h in [1, 10, 25, 50, 100]:
                if future >= h:
                    horizons.add(h)
        payload["official_horizons"] = sorted(horizons)
        payload["dt_s"] = sorted(dt_values)[len(dt_values) // 2] if dt_values else 0.0
        out[dataset_dir.name] = payload
    return out


def run_audit() -> list[dict]:
    converted = converted_dataset_horizons()
    raw = [
        inspect_sdd_scene_root(),
        inspect_opentraj_root(),
        inspect_full_trajnet_scene_root(),
        inspect_full_eth_ucy_scene_root(),
        inspect_aerialmpt_long_root(),
    ]
    records = []
    for row in raw:
        name = row["dataset_name"]
        related = {}
        if "trajnet" in name:
            related = converted.get("trajnet", {})
        elif "eth_ucy" in name:
            related = converted.get("eth_ucy", {})
        record = {
            **row,
            "pixel_or_metric": "metric" if row["coordinate_unit"] == "meter" else "pixel_or_dataset_coordinate",
            "max_raw_horizon": related.get("official_horizons", [1, 10])[-1] if related.get("official_horizons") else 0,
            "max_verified_t10": bool(related.get("episodes", 0) and 10 in related.get("official_horizons", [])),
            "max_verified_t25": bool(25 in related.get("official_horizons", [])),
            "max_verified_t50": bool(50 in related.get("official_horizons", [])),
            "max_verified_t100": bool(100 in related.get("official_horizons", [])),
            "effective_seconds_t10": 10 * float(related.get("dt_s", 0.0) or 0.0),
            "effective_seconds_t25": 25 * float(related.get("dt_s", 0.0) or 0.0),
            "effective_seconds_t50": 50 * float(related.get("dt_s", 0.0) or 0.0),
            "effective_seconds_t100": 100 * float(related.get("dt_s", 0.0) or 0.0),
            "suitable_for_scene_grounded_pedestrian_model": bool(row["actual_downloaded_or_user_path_verified"] and row["scene_image_available"]),
            "suitable_for_metric_evaluation": bool(row["homography_available"] or row["coordinate_unit"] == "meter"),
            "suitable_only_for_pixel_space_evaluation": bool(row["actual_downloaded_or_user_path_verified"] and not (row["homography_available"] or row["coordinate_unit"] == "meter")),
            "eligible_for_pedestrian_drone_long_horizon_gate": bool(50 in related.get("official_horizons", []) or 100 in related.get("official_horizons", [])),
        }
        records.append(record)
    return records


def write_outputs(records: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage7_scene_data_audit.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# Stage 7 Scene Data Audit", "", "| dataset | local/path | unit | scene_image | homography | t10 | t25 | t50 | t100 | metric_eval | notes |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        lines.append(
            f"| {r['dataset_name']} | {r['actual_downloaded_or_user_path_verified']} | {r['coordinate_unit']} | {r['scene_image_available']} | "
            f"{r['homography_available']} | {r['max_verified_t10']} | {r['max_verified_t25']} | {r['max_verified_t50']} | {r['max_verified_t100']} | "
            f"{r['suitable_for_metric_evaluation']} | {r['notes']} |"
        )
    lines += ["", "No pedestrian/drone t+50/t+100 claim is allowed unless the corresponding verified flags are true."]
    (REPORT_DIR / "stage7_scene_data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    payload = run_audit()
    write_outputs(payload)
    print(json.dumps(payload, indent=2))
