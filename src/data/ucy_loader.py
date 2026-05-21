from __future__ import annotations

from pathlib import Path

from src.data.full_eth_ucy_loader import load_full_eth_ucy_trajectories


def load_ucy_trajectories(root: str | Path, quick: bool = True, max_rows: int | None = None):
    table, meta = load_full_eth_ucy_trajectories(root, quick=quick, max_rows=max_rows)
    meta["dataset_name"] = "UCY crowd trajectories"
    return table, meta
