from __future__ import annotations

from pathlib import Path

from src.stage20_pipeline import _verify_path, build_stage20_sources


def list_stage20_sources():
    return build_stage20_sources()


def verify_stage20_path(dataset_id: str, path: str):
    return _verify_path(dataset_id, Path(path))


__all__ = ["list_stage20_sources", "verify_stage20_path"]

