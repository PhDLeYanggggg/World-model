from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import numpy as np
import pandas as pd


TGSIM_FOGGY_BOTTOM_CSV_URL = "https://data.transportation.gov/api/views/brzy-6zfh/rows.csv?accessType=DOWNLOAD"
TGSIM_PIXEL_TO_METER = 0.0186613838586


@dataclass
class TGSIMColumnMap:
    agent_id: str
    time: str
    x: str
    y: str
    speed: str | None = None
    acceleration: str | None = None
    vx: str | None = None
    vy: str | None = None
    width: str | None = None
    length: str | None = None
    agent_type: str | None = None


ALIASES = {
    "agent_id": ["agent_id", "track_id", "id", "object_id", "objectid", "road_user_id", "vehicle_id"],
    "time": ["time", "timestamp", "frame", "frame_id", "global_time", "time_s", "time_sec"],
    "x": ["x", "x_m", "global_x", "center_x", "xcenter", "location_x", "pos_x", "x_position"],
    "y": ["y", "y_m", "global_y", "center_y", "ycenter", "location_y", "pos_y", "y_position"],
    "speed": ["speed", "speed_mps", "v", "velocity", "instantaneous_speed"],
    "acceleration": ["acceleration", "accel", "acceleration_mps2", "a", "instantaneous_acceleration"],
    "vx": ["vx", "v_x", "velocity_x", "x_velocity"],
    "vy": ["vy", "v_y", "velocity_y", "y_velocity"],
    "width": ["width", "width_m", "object_width"],
    "length": ["length", "length_m", "object_length"],
    "agent_type": ["type", "agent_type", "road_user_type", "object_type", "class"],
}


def iter_tgsim_csv(path_or_url: str | Path, chunksize: int = 250_000, max_rows: int | None = None) -> Iterator[pd.DataFrame]:
    read = 0
    for chunk in pd.read_csv(path_or_url, chunksize=chunksize):
        if max_rows is not None:
            remaining = max_rows - read
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        read += len(chunk)
        yield chunk


def infer_tgsim_column_map(columns: Iterable[str]) -> TGSIMColumnMap:
    original = list(columns)
    normalized = {normalize_name(column): column for column in original}

    def pick(key: str, required: bool = False) -> str | None:
        for alias in ALIASES[key]:
            if normalize_name(alias) in normalized:
                return normalized[normalize_name(alias)]
        if required:
            raise ValueError(f"Could not infer required TGSIM column `{key}` from columns: {original[:50]}")
        return None

    return TGSIMColumnMap(
        agent_id=pick("agent_id", required=True),
        time=pick("time", required=True),
        x=pick("x", required=True),
        y=pick("y", required=True),
        speed=pick("speed"),
        acceleration=pick("acceleration"),
        vx=pick("vx"),
        vy=pick("vy"),
        width=pick("width"),
        length=pick("length"),
        agent_type=pick("agent_type"),
    )


def normalize_tgsim_chunk(chunk: pd.DataFrame, column_map: TGSIMColumnMap | None = None, pixels_to_meters: bool = False) -> pd.DataFrame:
    cmap = column_map or infer_tgsim_column_map(chunk.columns)
    out = pd.DataFrame()
    out["agent_id"] = chunk[cmap.agent_id].astype(str)
    out["time"] = pd.to_numeric(chunk[cmap.time], errors="coerce")
    x = pd.to_numeric(chunk[cmap.x], errors="coerce")
    y = pd.to_numeric(chunk[cmap.y], errors="coerce")
    scale = TGSIM_PIXEL_TO_METER if pixels_to_meters else 1.0
    out["x_m"] = x * scale
    out["y_m"] = y * scale
    out["speed_mps"] = pd.to_numeric(chunk[cmap.speed], errors="coerce") if cmap.speed else np.nan
    out["acceleration_mps2"] = pd.to_numeric(chunk[cmap.acceleration], errors="coerce") if cmap.acceleration else np.nan
    out["vx_mps"] = pd.to_numeric(chunk[cmap.vx], errors="coerce") if cmap.vx else np.nan
    out["vy_mps"] = pd.to_numeric(chunk[cmap.vy], errors="coerce") if cmap.vy else np.nan
    out["width_m"] = pd.to_numeric(chunk[cmap.width], errors="coerce") * scale if cmap.width else np.nan
    out["length_m"] = pd.to_numeric(chunk[cmap.length], errors="coerce") * scale if cmap.length else np.nan
    out["agent_type"] = chunk[cmap.agent_type].astype(str) if cmap.agent_type else "unknown"
    out = out.dropna(subset=["time", "x_m", "y_m"])
    return fill_missing_kinematics(out)


def fill_missing_kinematics(data: pd.DataFrame) -> pd.DataFrame:
    out = data.sort_values(["agent_id", "time"]).copy()
    dt = out.groupby("agent_id")["time"].diff()
    dx = out.groupby("agent_id")["x_m"].diff()
    dy = out.groupby("agent_id")["y_m"].diff()
    derived_vx = dx / dt.replace(0, np.nan)
    derived_vy = dy / dt.replace(0, np.nan)
    out["vx_mps"] = out["vx_mps"].fillna(derived_vx)
    out["vy_mps"] = out["vy_mps"].fillna(derived_vy)
    derived_speed = np.sqrt(out["vx_mps"] ** 2 + out["vy_mps"] ** 2)
    out["speed_mps"] = out["speed_mps"].fillna(derived_speed)
    dv = out.groupby("agent_id")["speed_mps"].diff()
    out["acceleration_mps2"] = out["acceleration_mps2"].fillna(dv / dt.replace(0, np.nan))
    out["body_radius_m"] = np.nanmax(
        np.vstack(
            [
                np.full(len(out), 0.30),
                np.nan_to_num(out["width_m"].to_numpy(dtype=float), nan=0.0) * 0.5,
            ]
        ),
        axis=0,
    )
    return out.fillna({"vx_mps": 0.0, "vy_mps": 0.0, "speed_mps": 0.0, "acceleration_mps2": 0.0})


def summarize_tgsim_file(path_or_url: str | Path, max_rows: int = 50_000) -> Dict:
    chunks = []
    column_map = None
    for chunk in iter_tgsim_csv(path_or_url, chunksize=min(max_rows, 50_000), max_rows=max_rows):
        column_map = column_map or infer_tgsim_column_map(chunk.columns)
        chunks.append(normalize_tgsim_chunk(chunk, column_map))
    if not chunks:
        return {"rows": 0, "error": "no rows"}
    data = pd.concat(chunks, ignore_index=True)
    return {
        "rows_sampled": int(len(data)),
        "agents": int(data["agent_id"].nunique()),
        "time_min": float(data["time"].min()),
        "time_max": float(data["time"].max()),
        "agent_types": sorted(str(x) for x in data["agent_type"].dropna().unique())[:20],
        "x_range_m": [float(data["x_m"].min()), float(data["x_m"].max())],
        "y_range_m": [float(data["y_m"].min()), float(data["y_m"].max())],
        "columns_inferred": column_map.__dict__ if column_map else {},
    }


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())
