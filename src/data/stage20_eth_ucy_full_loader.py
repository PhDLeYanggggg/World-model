from __future__ import annotations

from pathlib import Path

from src.stage20_pipeline import _verify_path


def verify_eth_ucy_full_path(path: str):
    return _verify_path("eth_ucy_full", Path(path))


__all__ = ["verify_eth_ucy_full_path"]

