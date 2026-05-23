from __future__ import annotations

from src.stage20_pipeline import build_stage20_sources


def resolve_official_sources():
    return build_stage20_sources()


__all__ = ["resolve_official_sources"]

