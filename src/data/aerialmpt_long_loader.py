from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_aerialmpt_long_trajectories(root: str | Path, quick: bool = True, max_rows: int | None = None):
    path = Path(root)
    csvs = list(path.rglob("*.csv")) if path.exists() else []
    if not csvs:
        raise FileNotFoundError(f"No AerialMPT long CSV files found under {path}; current bauma3 remains short-horizon only.")
    table = pd.read_csv(csvs[0])
    if max_rows:
        table = table.head(max_rows)
    required = {"frame_id", "agent_id", "x", "y"}
    if not required.issubset(table.columns):
        raise ValueError(f"AerialMPT long loader needs columns {required}; got {list(table.columns)}")
    table = table.copy()
    table["scene_id"] = table.get("scene_id", csvs[0].stem)
    table["episode_id"] = table.get("episode_id", csvs[0].stem)
    table["agent_type"] = table.get("agent_type", "pedestrian")
    table["valid"] = table.get("valid", True)
    table = table.sort_values(["scene_id", "agent_id", "frame_id"])
    table["time"] = table.get("time", table["frame_id"].astype(float))
    table["dt"] = table.groupby(["scene_id", "agent_id"])["time"].diff().fillna(table.groupby(["scene_id", "agent_id"])["time"].diff().median()).fillna(1.0)
    table["causal_vx"] = table.groupby(["scene_id", "agent_id"])["x"].diff().fillna(0.0) / table["dt"].replace(0, 1.0)
    table["causal_vy"] = table.groupby(["scene_id", "agent_id"])["y"].diff().fillna(0.0) / table["dt"].replace(0, 1.0)
    table["causal_ax"] = table.groupby(["scene_id", "agent_id"])["causal_vx"].diff().fillna(0.0) / table["dt"].replace(0, 1.0)
    table["causal_ay"] = table.groupby(["scene_id", "agent_id"])["causal_vy"].diff().fillna(0.0) / table["dt"].replace(0, 1.0)
    meta = {"dataset_name": "AerialMPT long candidate", "coordinate_unit": "pixel", "whether_metric_coordinates": False}
    return table, meta
