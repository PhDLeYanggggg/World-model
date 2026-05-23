from __future__ import annotations

from pathlib import Path

from src.stage20_pipeline import _verify_path


def verify_ego_video_path(dataset_id: str, path: str):
    return _verify_path(dataset_id, Path(path))


__all__ = ["verify_ego_video_path"]

