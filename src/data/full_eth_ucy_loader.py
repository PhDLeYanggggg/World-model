from __future__ import annotations

from pathlib import Path

from src.data.trajnet_loader import load_trajnet_trajectories


def load_full_eth_ucy_trajectories(root: str | Path, quick: bool = True, max_rows: int | None = None):
    table, meta = load_trajnet_trajectories(root, quick=quick, max_rows=max_rows)
    meta["dataset_name"] = "ETH/UCY full or bundled fallback"
    table["episode_id"] = "eth_ucy_full"
    return table, meta
