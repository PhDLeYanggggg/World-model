from __future__ import annotations

from src.stage18_pipeline import collect_multimodal_data


def verify_full_trajnet_paths():
    return collect_multimodal_data(quick=True)

