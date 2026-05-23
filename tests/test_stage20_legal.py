from __future__ import annotations

from src.stage20_pipeline import build_stage20_sources


def test_stage20_no_gated_dataset_is_auto_downloaded():
    for source in build_stage20_sources():
        gated = (
            source["requires_login"]
            or source["requires_application"]
            or source["requires_manual_terms_acceptance"]
        )
        if gated:
            assert not source["auto_download_allowed"]


def test_stage20_ego_video_not_official_topdown_benchmark():
    for source in build_stage20_sources():
        if source["category"] == "human_egocentric_video_pretraining":
            assert source["usable_for_JEPA_pretraining"]
            assert not source["usable_for_official_eval"]


def test_stage20_traffic_is_diagnostic_only():
    for source in build_stage20_sources():
        if source["category"] == "traffic / driving diagnostic only":
            assert source["usable_for_diagnostic_only"]
            assert not source["usable_for_official_eval"]

