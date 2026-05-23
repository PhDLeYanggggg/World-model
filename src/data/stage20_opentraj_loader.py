from __future__ import annotations

from pathlib import Path

from src.stage20_pipeline import _verify_path


def verify_opentraj_path(path: str):
    return _verify_path("opentraj", Path(path))


__all__ = ["verify_opentraj_path"]

