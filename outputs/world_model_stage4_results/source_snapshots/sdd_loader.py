from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data.trajnet_loader import finalize_table


def load_sdd_trajectories(data_path: str | Path, quick: bool = False, max_rows: int | None = None) -> Tuple[pd.DataFrame, Dict]:
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"SDD path does not exist: {root}")
    files = sorted(root.rglob("annotations.txt")) + sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No SDD annotations.txt/csv files found under: {root}")
    frames = []
    for file in files:
        scene = "_".join(file.parts[-4:-1]) if file.name == "annotations.txt" and len(file.parts) >= 4 else file.stem
        frames.append(read_sdd_file(file, scene))
        if quick and sum(len(item) for item in frames) >= 150_000:
            break
    raw = pd.concat(frames, ignore_index=True)
    if max_rows:
        raw = raw.head(max_rows)
    table = finalize_table(raw, dataset_scene_prefix="sdd")
    meta = {
        "dataset_name": "Stanford Drone Dataset",
        "source_path": str(root),
        "coordinate_unit": "pixel",
        "whether_metric_coordinates": False,
        "whether_scene_geometry_available": False,
        "scene_geometry_note": "SDD is image-space here; homography/scale must be provided before metric physical claims.",
    }
    return table, meta


def read_sdd_file(path: Path, scene_id: str) -> pd.DataFrame:
    if path.name == "annotations.txt":
        cols = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
        df = pd.read_csv(path, sep=r"\s+", names=cols, engine="python")
        x = (df["xmin"] + df["xmax"]) * 0.5
        y = (df["ymin"] + df["ymax"]) * 0.5
        return pd.DataFrame({"scene_id": scene_id, "frame_id": df["frame"].astype(int), "agent_id": df["track_id"].astype(str), "x": x.astype(float), "y": y.astype(float), "agent_type": df["label"].astype(str)})
    df = pd.read_csv(path)
    cols = {normalize_name(col): col for col in df.columns}
    frame = cols.get("frame") or cols.get("frameid")
    agent = cols.get("trackid") or cols.get("agentid")
    if "xmin" in cols and "xmax" in cols:
        x = (df[cols["xmin"]] + df[cols["xmax"]]) * 0.5
        y = (df[cols["ymin"]] + df[cols["ymax"]]) * 0.5
    else:
        x = df[cols.get("x")]
        y = df[cols.get("y")]
    label = df[cols["label"]].astype(str) if "label" in cols else "unknown"
    return pd.DataFrame({"scene_id": scene_id, "frame_id": df[frame].astype(int), "agent_id": df[agent].astype(str), "x": x.astype(float), "y": y.astype(float), "agent_type": label})


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())
