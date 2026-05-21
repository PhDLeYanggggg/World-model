from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_trajnet_trajectories(data_path: str | Path, quick: bool = False, max_rows: int | None = None) -> Tuple[pd.DataFrame, Dict]:
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"TrajNet++ path does not exist: {root}")
    files = sorted(root.rglob("*.ndjson")) + sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No TrajNet++ ndjson/jsonl/csv files found under: {root}")
    rows: List[Dict] = []
    for file in files:
        if file.suffix.lower() == ".csv":
            rows.extend(read_generic_csv(file, scene_id=file.stem))
        else:
            rows.extend(read_trajnet_ndjson(file))
        if max_rows and len(rows) >= max_rows:
            rows = rows[:max_rows]
            break
        if quick and len(rows) >= 150_000:
            rows = rows[:150_000]
            break
    table = finalize_table(pd.DataFrame(rows), dataset_scene_prefix="trajnet")
    meta = {
        "dataset_name": "TrajNet++",
        "source_path": str(root),
        "coordinate_unit": "dataset_coordinate",
        "whether_metric_coordinates": False,
        "whether_scene_geometry_available": False,
        "scene_geometry_note": "TrajNet++ is trajectory-centric; scene geometry is usually unavailable.",
    }
    return table, meta


def read_trajnet_ndjson(path: Path) -> List[Dict]:
    rows = []
    scene_id = path.stem
    current_scene = scene_id
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if "scene" in item:
                scene = item["scene"]
                current_scene = str(scene.get("id", scene_id))
            if "track" in item:
                track = item["track"]
                rows.append(
                    {
                        "scene_id": current_scene,
                        "frame_id": int(track.get("f", track.get("frame", 0))),
                        "agent_id": str(track.get("p", track.get("pedestrian", track.get("agent_id", "0")))),
                        "x": float(track.get("x")),
                        "y": float(track.get("y")),
                        "agent_type": "pedestrian",
                    }
                )
    return rows


def read_generic_csv(path: Path, scene_id: str) -> List[Dict]:
    df = pd.read_csv(path)
    cols = {normalize_name(col): col for col in df.columns}
    frame = cols.get("frame") or cols.get("frameid") or cols.get("f")
    agent = cols.get("agentid") or cols.get("pedestrianid") or cols.get("trackid") or cols.get("p")
    x = cols.get("x") or cols.get("posx")
    y = cols.get("y") or cols.get("posy")
    if not all([frame, agent, x, y]):
        raise ValueError(f"Cannot infer TrajNet CSV columns from {path}: {list(df.columns)}")
    return [
        {"scene_id": scene_id, "frame_id": int(row[frame]), "agent_id": str(row[agent]), "x": float(row[x]), "y": float(row[y]), "agent_type": "pedestrian"}
        for _, row in df.iterrows()
    ]


def finalize_table(df: pd.DataFrame, dataset_scene_prefix: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No trajectory rows parsed.")
    out = df.sort_values(["scene_id", "agent_id", "frame_id"]).copy()
    out["episode_id"] = out["scene_id"].astype(str)
    for col in ["vx", "vy"]:
        out[col] = np.nan
    dt = out.groupby(["scene_id", "agent_id"])["frame_id"].diff().replace(0, np.nan)
    out["vx"] = out.groupby(["scene_id", "agent_id"])["x"].diff() / dt
    out["vy"] = out.groupby(["scene_id", "agent_id"])["y"].diff() / dt
    out["ax"] = out.groupby(["scene_id", "agent_id"])["vx"].diff() / dt
    out["ay"] = out.groupby(["scene_id", "agent_id"])["vy"].diff() / dt
    out[["vx", "vy", "ax", "ay"]] = out[["vx", "vy", "ax", "ay"]].fillna(0.0)
    out["heading"] = np.arctan2(out["vy"], out["vx"])
    out["valid"] = True
    out["scene_id"] = dataset_scene_prefix + "_" + out["scene_id"].astype(str)
    out["episode_id"] = out["scene_id"]
    return out


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())
