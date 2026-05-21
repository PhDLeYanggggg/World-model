from __future__ import annotations

from pathlib import Path

from src.data.ngsim_loader import load_ngsim_trajectories


def load_opentraj_trajectories(root: str | Path, quick: bool = True, max_rows: int | None = None):
    table, meta = load_ngsim_trajectories(root, quick=quick, max_rows=max_rows)
    meta["dataset_name"] = "OpenTraj-compatible generic trajectories"
    return table, meta
