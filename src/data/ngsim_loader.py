from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.data.trajnet_loader import finalize_table


def load_ngsim_trajectories(data_path: str | Path, quick: bool = False, max_rows: int | None = None) -> Tuple[pd.DataFrame, Dict]:
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"NGSIM path does not exist: {root}")
    files = sorted(root.rglob("*.csv")) + sorted(root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No NGSIM csv/txt files found under: {root}")
    frames = []
    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception:
            df = pd.read_csv(file, sep=r"\s+", engine="python")
        cols = {normalize_name(c): c for c in df.columns}
        frame = cols.get("frameid") or cols.get("frame") or cols.get("globaltime")
        agent = cols.get("vehicleid") or cols.get("vehicle") or cols.get("id")
        x = cols.get("localx") or cols.get("globalx") or cols.get("x")
        y = cols.get("localy") or cols.get("globaly") or cols.get("y")
        if not all([frame, agent, x, y]):
            continue
        frames.append(pd.DataFrame({"scene_id": file.stem, "frame_id": df[frame].astype(int), "agent_id": df[agent].astype(str), "x": df[x].astype(float), "y": df[y].astype(float), "agent_type": "vehicle"}))
        if quick and sum(len(item) for item in frames) >= 150_000:
            break
    if not frames:
        raise ValueError("NGSIM loader could not infer columns.")
    raw = pd.concat(frames, ignore_index=True)
    if max_rows:
        raw = raw.head(max_rows)
    table = finalize_table(raw, "ngsim")
    return table, {"dataset_name": "NGSIM", "coordinate_unit": "feet_or_meter_source_dependent", "whether_metric_coordinates": False, "whether_scene_geometry_available": False, "default_velocity_source": "causal_fd"}


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())
