from __future__ import annotations

from src.stage26_pipeline import _feature_names, _goal_features, _select_cost_aware


def test_stage26_feature_names_do_not_include_forbidden_inputs():
    names = " ".join(_feature_names()).lower()
    assert "future" not in names
    assert "future_endpoint" not in names
    assert "test_endpoint" not in names
    assert "central" not in names
    assert "oracle" not in names


def test_stage26_goal_features_handles_missing_scene_pack():
    row = {"scene_id": "missing", "video_id": "missing"}
    out = _goal_features(row, __import__("numpy").array([0.0, 0.0]), __import__("numpy").array([1.0, 0.0]), {})
    assert out["feature_values"][0] == 0.0
    assert out["feature_values"][1] >= 1e6


def test_stage26_select_cost_aware_is_importable():
    assert callable(_select_cost_aware)
