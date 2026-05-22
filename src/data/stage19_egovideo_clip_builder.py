from __future__ import annotations

from src.stage19_pipeline import verify_egocentric_data


def build_ego_clips(quick: bool = True):
    return verify_egocentric_data(quick=quick)

