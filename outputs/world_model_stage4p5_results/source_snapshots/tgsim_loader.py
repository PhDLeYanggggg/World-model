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
    variants = compute_velocity_variants(raw)
    table = pd.DataFrame(
        {
            "episode_id": "tgsim_foggy_bottom",
            "scene_id": "tgsim_foggy_bottom",
            "frame_id": dense_frame_ids(raw["time"]),
            "agent_id": raw["agent_id"].astype(str),
            "time": raw["time"].astype(float),
            "dt": variants["dt"].astype(float),
            "x": raw["x_m"].astype(float),
            "y": raw["y_m"].astype(float),
            "vx": variants["causal_vx"].astype(float),
            "vy": variants["causal_vy"].astype(float),
            "ax": variants["causal_ax"].astype(float),
            "ay": variants["causal_ay"].astype(float),
            "native_vx": raw["vx_mps"].astype(float),
            "native_vy": raw["vy_mps"].astype(float),
            "native_ax": raw["ax_mps2"].astype(float),
            "native_ay": raw["ay_mps2"].astype(float),
            "causal_vx": variants["causal_vx"].astype(float),
            "causal_vy": variants["causal_vy"].astype(float),
            "causal_ax": variants["causal_ax"].astype(float),
            "causal_ay": variants["causal_ay"].astype(float),
            "central_vx": variants["central_vx"].astype(float),
            "central_vy": variants["central_vy"].astype(float),
            "central_ax": variants["central_ax"].astype(float),
            "central_ay": variants["central_ay"].astype(float),
            "heading": np.arctan2(variants["causal_vy"].astype(float), variants["causal_vx"].astype(float)),
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
        "default_velocity_source": "causal_fd",
        "velocity_columns": ["native_vx", "native_vy", "causal_vx", "causal_vy", "central_vx", "central_vy"],
        "dt_note": "dt is inferred from consecutive native TGSIM time values per agent; dense frame_id is only an index.",
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


def compute_velocity_variants(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.sort_values(["agent_id", "time"]).copy()
    group = out.groupby("agent_id", sort=False)
    dt = group["time"].diff().replace(0, np.nan)
    dx = group["x_m"].diff()
    dy = group["y_m"].diff()
    causal_vx = dx / dt
    causal_vy = dy / dt

    prev_t = group["time"].shift(1)
    next_t = group["time"].shift(-1)
    prev_x = group["x_m"].shift(1)
    next_x = group["x_m"].shift(-1)
    prev_y = group["y_m"].shift(1)
    next_y = group["y_m"].shift(-1)
    central_dt = (next_t - prev_t).replace(0, np.nan)
    central_vx = (next_x - prev_x) / central_dt
    central_vy = (next_y - prev_y) / central_dt

    native_vx = out["vx_mps"]
    native_vy = out["vy_mps"]
    causal_vx = causal_vx.fillna(native_vx).fillna(0.0)
    causal_vy = causal_vy.fillna(native_vy).fillna(0.0)
    central_vx = central_vx.fillna(causal_vx).fillna(0.0)
    central_vy = central_vy.fillna(causal_vy).fillna(0.0)

    causal_ax = causal_vx.groupby(out["agent_id"]).diff() / dt
    causal_ay = causal_vy.groupby(out["agent_id"]).diff() / dt
    central_ax = central_vx.groupby(out["agent_id"]).diff() / dt
    central_ay = central_vy.groupby(out["agent_id"]).diff() / dt

    variants = pd.DataFrame(index=out.index)
    variants["dt"] = dt.fillna(dt.groupby(out["agent_id"]).transform("median")).fillna(dt.median()).fillna(1.0)
    variants["causal_vx"] = causal_vx.astype(float)
    variants["causal_vy"] = causal_vy.astype(float)
    variants["causal_ax"] = causal_ax.fillna(out["ax_mps2"]).fillna(0.0).astype(float)
    variants["causal_ay"] = causal_ay.fillna(out["ay_mps2"]).fillna(0.0).astype(float)
    variants["central_vx"] = central_vx.astype(float)
    variants["central_vy"] = central_vy.astype(float)
    variants["central_ax"] = central_ax.fillna(variants["causal_ax"]).fillna(0.0).astype(float)
    variants["central_ay"] = central_ay.fillna(variants["causal_ay"]).fillna(0.0).astype(float)
    return variants.loc[raw.index]


def fill_acceleration_components(table: pd.DataFrame) -> pd.DataFrame:
    out = table.sort_values(["agent_id", "frame_id"]).copy()
    dt = out.groupby("agent_id")["time"].diff().replace(0, np.nan) if "time" in out.columns else out.groupby("agent_id")["frame_id"].diff().replace(0, np.nan)
    derived_ax = out.groupby("agent_id")["vx"].diff() / dt
    derived_ay = out.groupby("agent_id")["vy"].diff() / dt
    out["ax"] = out["ax"].fillna(derived_ax)
    out["ay"] = out["ay"].fillna(derived_ay)
    out[["ax", "ay"]] = out[["ax", "ay"]].fillna(0.0)
    return out
