from __future__ import annotations

from pathlib import Path

from src.data.trajnet_loader import load_trajnet_trajectories


def load_full_trajnet_trajectories(root: str | Path, quick: bool = True, max_rows: int | None = None):
    return load_trajnet_trajectories(root, quick=quick, max_rows=max_rows)
