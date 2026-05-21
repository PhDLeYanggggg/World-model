from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.data.trajnet_loader import finalize_table


def load_opendd_trajectories(data_path: str | Path, quick: bool = False, max_rows: int | None = None) -> Tuple[pd.DataFrame, Dict]:
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"OpenDD path does not exist: {root}")
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No OpenDD CSV files found under: {root}")
    frames = []
    for file in files:
        df = pd.read_csv(file)
        cols = {normalize_name(c): c for c in df.columns}
        frame = cols.get("frame") or cols.get("frameid") or cols.get("timestampms") or cols.get("time")
        agent = cols.get("trackid") or cols.get("agentid") or cols.get("id")
        x = cols.get("x") or cols.get("utmutmxx") or cols.get("posx") or cols.get("xcenter")
        y = cols.get("y") or cols.get("utmy") or cols.get("posy") or cols.get("ycenter")
        if not all([frame, agent, x, y]):
            continue
        label = cols.get("class") or cols.get("agenttype") or cols.get("type")
        frames.append(pd.DataFrame({"scene_id": file.stem, "frame_id": df[frame].astype(int), "agent_id": df[agent].astype(str), "x": df[x].astype(float), "y": df[y].astype(float), "agent_type": df[label].astype(str) if label else "unknown"}))
        if quick and sum(len(item) for item in frames) >= 150_000:
            break
    if not frames:
        raise ValueError("OpenDD loader could not infer columns from CSV files.")
    raw = pd.concat(frames, ignore_index=True)
    if max_rows:
        raw = raw.head(max_rows)
    table = finalize_table(raw, "opendd")
    return table, {"dataset_name": "OpenDD", "coordinate_unit": "meter", "whether_metric_coordinates": True, "whether_scene_geometry_available": True, "default_velocity_source": "causal_fd"}


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())
