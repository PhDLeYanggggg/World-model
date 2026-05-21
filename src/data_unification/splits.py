from __future__ import annotations

import pandas as pd


def assign_scene_or_time_splits(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    scenes = sorted(out["scene_id"].unique())
    if len(scenes) >= 3:
        train = set(scenes[: max(1, int(0.6 * len(scenes)))])
        val = set(scenes[max(1, int(0.6 * len(scenes))) : max(2, int(0.8 * len(scenes)))])
        out["split"] = out["scene_id"].map(lambda scene: "train" if scene in train else ("val" if scene in val else "test"))
        return out
    frames = sorted(out["frame_id"].unique())
    n = len(frames)
    train_cut = frames[int(0.6 * n)] if n else 0
    val_cut = frames[int(0.8 * n)] if n else 0
    out["split"] = out["frame_id"].map(lambda frame: "train" if frame < train_cut else ("val" if frame < val_cut else "test"))
    return out
