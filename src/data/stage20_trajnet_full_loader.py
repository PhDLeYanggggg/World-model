from __future__ import annotations

from pathlib import Path

from src.stage20_pipeline import _verify_path


def verify_trajnet_full_path(path: str):
    return _verify_path("trajnet_full", Path(path))


__all__ = ["verify_trajnet_full_path"]

