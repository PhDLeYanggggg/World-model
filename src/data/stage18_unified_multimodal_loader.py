from __future__ import annotations

from src.stage18_pipeline import collect_multimodal_data


def load_stage18_multimodal_sources(quick: bool = True):
    return collect_multimodal_data(quick=quick)

