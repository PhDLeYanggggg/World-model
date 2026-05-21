from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data.real_trajectory_loader import load_real_trajectory_table
from src.data_unification.normalization import add_kinematic_norms, local_scene_origin
from src.data_unification.world_state_schema import WORLD_STATE_REQUIRED_COLUMNS


def convert_real_dataset(dataset: str, data_path: str, output_root: str | Path = "data/stage5_world_state", quick: bool = True) -> Tuple[pd.DataFrame, Dict]:
    raw, meta = load_real_trajectory_table(dataset, data_path, quick=quick, max_rows=250_000 if quick else None)
    table = pd.DataFrame()
    table["episode_id"] = raw["episode_id"]
    table["dataset_name"] = meta.get("dataset_name", dataset)
    table["scene_id"] = raw["scene_id"]
    table["frame_id"] = raw["frame_id"]
    table["time_s"] = raw["time"] if "time" in raw.columns else raw["frame_id"].astype(float)
    table["dt_s"] = raw["dt"] if "dt" in raw.columns else raw.groupby("agent_id")["frame_id"].diff().fillna(1.0)
    table["agent_id"] = raw["agent_id"]
    table["agent_type"] = raw["agent_type"]
    table["x"] = raw["x"].astype(float)
    table["y"] = raw["y"].astype(float)
    table["z"] = 0.0
    table["vx"] = raw["causal_vx"] if "causal_vx" in raw.columns else raw["vx"]
    table["vy"] = raw["causal_vy"] if "causal_vy" in raw.columns else raw["vy"]
    table["vz"] = 0.0
    table["ax"] = raw["causal_ax"] if "causal_ax" in raw.columns else raw["ax"]
    table["ay"] = raw["causal_ay"] if "causal_ay" in raw.columns else raw["ay"]
    table["az"] = 0.0
    table["heading"] = np.arctan2(table["vy"], table["vx"])
    table["heading_rate"] = table.groupby("agent_id")["heading"].diff() / table["dt_s"].replace(0, np.nan)
    table["body_radius"] = 0.35
    table["body_length"] = np.nan
    table["body_width"] = np.nan
    table["valid"] = raw["valid"]
    table["observed_mask"] = True
    table["coordinate_unit"] = meta.get("coordinate_unit", "meter")
    table["coordinate_frame"] = "source_world"
    table["source_velocity_type"] = "causal_fd"
    table["source_position_type"] = "source_position"
    table = add_kinematic_norms(local_scene_origin(table))
    table = table[WORLD_STATE_REQUIRED_COLUMNS]
    out_dir = Path(output_root) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "world_state.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return table, meta


def write_conversion_report(dataset: str, table: pd.DataFrame, meta: Dict, output_dir: str | Path = "outputs/reports/stage5_data") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": dataset,
        "rows": int(len(table)),
        "agents": int(table["agent_id"].nunique()) if len(table) else 0,
        "scenes": int(table["scene_id"].nunique()) if len(table) else 0,
        "velocity_type": "causal_fd",
        "source_meta": meta,
    }
    (out / f"conversion_report_{dataset}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
