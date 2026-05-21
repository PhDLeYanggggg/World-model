from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data.trajnet_loader import finalize_table


def load_eth_ucy_trajectories(data_path: str | Path, quick: bool = False, max_rows: int | None = None) -> Tuple[pd.DataFrame, Dict]:
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"ETH/UCY path does not exist: {root}")
    files = sorted(root.rglob("*.txt")) + sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No ETH/UCY txt/csv files found under: {root}")
    frames = []
    for file in files:
        df = read_eth_ucy_file(file)
        df["scene_id"] = file.stem
        frames.append(df)
        if quick and sum(len(item) for item in frames) >= 150_000:
            break
    raw = pd.concat(frames, ignore_index=True)
    if max_rows:
        raw = raw.head(max_rows)
    table = finalize_table(raw, dataset_scene_prefix="eth_ucy")
    meta = {
        "dataset_name": "ETH/UCY",
        "source_path": str(root),
        "coordinate_unit": "dataset_coordinate",
        "whether_metric_coordinates": False,
        "whether_scene_geometry_available": False,
        "scene_geometry_note": "Scene geometry/homography is not loaded by this generic ETH/UCY adapter.",
    }
    return table, meta


def read_eth_ucy_file(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        cols = {normalize_name(col): col for col in df.columns}
        frame = cols.get("frame") or cols.get("frameid")
        agent = cols.get("agentid") or cols.get("pedestrianid") or cols.get("trackid")
        x = cols.get("x") or cols.get("posx")
        y = cols.get("y") or cols.get("posy")
        if all([frame, agent, x, y]):
            return pd.DataFrame({"frame_id": df[frame].astype(int), "agent_id": df[agent].astype(str), "x": df[x].astype(float), "y": df[y].astype(float), "agent_type": "pedestrian"})
    except Exception:
        pass
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 4:
        raise ValueError(f"ETH/UCY file must have at least 4 numeric columns: {path}")
    return pd.DataFrame({"frame_id": arr[:, 0].astype(int), "agent_id": arr[:, 1].astype(int).astype(str), "x": arr[:, 2].astype(float), "y": arr[:, 3].astype(float), "agent_type": "pedestrian"})


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())
