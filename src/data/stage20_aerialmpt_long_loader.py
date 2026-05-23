from __future__ import annotations

from pathlib import Path

from src.stage20_pipeline import _verify_path


def verify_aerialmpt_long_path(path: str):
    return _verify_path("aerialmpt_long", Path(path))


__all__ = ["verify_aerialmpt_long_path"]

