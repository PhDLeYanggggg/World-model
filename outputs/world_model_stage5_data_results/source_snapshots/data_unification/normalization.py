from __future__ import annotations

import pandas as pd


def local_scene_origin(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for _, idx in out.groupby("scene_id").groups.items():
        x0 = out.loc[idx, "x"].min()
        y0 = out.loc[idx, "y"].min()
        out.loc[idx, "x"] = out.loc[idx, "x"] - x0
        out.loc[idx, "y"] = out.loc[idx, "y"] - y0
    out["coordinate_frame"] = "local_scene_origin"
    return out


def add_kinematic_norms(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["speed"] = (out["vx"] ** 2 + out["vy"] ** 2 + out["vz"] ** 2) ** 0.5
    out["acceleration_norm"] = (out["ax"] ** 2 + out["ay"] ** 2 + out["az"] ** 2) ** 0.5
    return out
