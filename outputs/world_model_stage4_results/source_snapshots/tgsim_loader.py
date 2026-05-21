from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data.tgsim_adapter import TGSIM_FOGGY_BOTTOM_CSV_URL, infer_tgsim_column_map, iter_tgsim_csv, normalize_tgsim_chunk


def load_tgsim_trajectories(data_path: str | Path, quick: bool = False, max_rows: int | None = None) -> Tuple[pd.DataFrame, Dict]:
    path = resolve_tgsim_path(data_path)
    if max_rows is None:
        max_rows = 250_000 if quick else None
    chunks = []
    column_map = None
    for chunk in iter_tgsim_csv(path, chunksize=100_000 if quick else 250_000, max_rows=max_rows):
        column_map = column_map or infer_tgsim_column_map(chunk.columns)
        chunks.append(normalize_tgsim_chunk(chunk, column_map))
    if not chunks:
        raise ValueError(f"TGSIM path produced no rows: {data_path}")
    raw = pd.concat(chunks, ignore_index=True)
    table = pd.DataFrame(
        {
            "episode_id": "tgsim_foggy_bottom",
            "scene_id": "tgsim_foggy_bottom",
            "frame_id": dense_frame_ids(raw["time"]),
            "agent_id": raw["agent_id"].astype(str),
            "x": raw["x_m"].astype(float),
            "y": raw["y_m"].astype(float),
            "vx": raw["vx_mps"].astype(float),
            "vy": raw["vy_mps"].astype(float),
            "ax": raw["ax_mps2"].astype(float),
            "ay": raw["ay_mps2"].astype(float),
            "heading": np.arctan2(raw["vy_mps"].astype(float), raw["vx_mps"].astype(float)),
            "agent_type": raw["agent_type"].astype(str),
            "valid": True,
        }
    )
    table = fill_acceleration_components(table)
    meta = {
        "dataset_name": "TGSIM Foggy Bottom",
        "source_path": str(data_path),
        "coordinate_unit": "meter",
        "whether_metric_coordinates": True,
        "whether_scene_geometry_available": False,
        "scene_geometry_note": "TGSIM publishes region polygons, but this loader currently consumes trajectory CSV only unless polygons are provided separately.",
        "columns_inferred": column_map.__dict__ if column_map else {},
    }
    return table, meta


def resolve_tgsim_path(data_path: str | Path) -> str:
    if str(data_path).lower() in {"public", "url", "remote"}:
        return TGSIM_FOGGY_BOTTOM_CSV_URL
    path = Path(data_path)
    if path.is_dir():
        csvs = sorted(path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in TGSIM directory: {path}")
        return str(csvs[0])
    return str(data_path)


def dense_frame_ids(time: pd.Series) -> np.ndarray:
    unique = {value: idx for idx, value in enumerate(sorted(time.dropna().unique()))}
    return time.map(unique).astype(int).to_numpy()


def fill_acceleration_components(table: pd.DataFrame) -> pd.DataFrame:
    out = table.sort_values(["agent_id", "frame_id"]).copy()
    dt = out.groupby("agent_id")["frame_id"].diff().replace(0, np.nan)
    derived_ax = out.groupby("agent_id")["vx"].diff() / dt
    derived_ay = out.groupby("agent_id")["vy"].diff() / dt
    out["ax"] = out["ax"].fillna(derived_ax)
    out["ay"] = out["ay"].fillna(derived_ay)
    out[["ax", "ay"]] = out[["ax", "ay"]].fillna(0.0)
    return out
