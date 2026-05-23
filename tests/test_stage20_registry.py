from __future__ import annotations

from src.stage20_pipeline import build_stage20_sources, compute_priority_score


def test_stage20_registry_has_required_source_scale_and_categories():
    sources = build_stage20_sources()
    assert len(sources) >= 20
    categories = {source["category"] for source in sources}
    assert "real_topdown_pedestrian_drone_official_benchmark" in categories
    assert "human_egocentric_video_pretraining" in categories
    assert "simulation_and_synthetic" in categories
    assert "traffic / driving diagnostic only" in categories
    assert sum(1 for source in sources if source["official_source_found"]) >= 10


def test_stage20_priority_score_is_bounded():
    for source in build_stage20_sources():
        score = compute_priority_score(source)
        assert 0 <= score <= 100

