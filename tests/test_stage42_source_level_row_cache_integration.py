import numpy as np

from src import stage42_source_level_row_cache_integration as iv


def test_alignment_accepts_geometry_even_when_text_ids_differ():
    labels = {
        "domain": np.asarray(["TrajNet", "UCY", "UCY"]),
        "source_file": np.asarray(["t", "source_a", "source_b"]),
        "scene_id": np.asarray(["t", "scene_a", "scene_b"]),
        "horizon": np.asarray([50, 50, 100]),
        "current_xy": np.asarray([[9.0, 9.0], [0.0, 1.0], [2.0, 3.0]]),
        "future_xy": np.asarray([[9.0, 9.0], [4.0, 5.0], [6.0, 7.0]]),
        "waypoint_xy": np.zeros((3, 4, 2), dtype=float),
        "waypoint_valid": np.ones((3, 4), dtype=bool),
        "normalizer": np.asarray([1.0, 2.0, 3.0]),
    }
    vlabels = {
        "domain": np.asarray(["UCY", "UCY"]),
        "source_file": np.asarray(["other_a", "other_b"]),
        "scene_id": np.asarray(["other_scene_a", "other_scene_b"]),
        "horizon": np.asarray([50, 100]),
        "current_xy": labels["current_xy"][1:],
        "future_xy": labels["future_xy"][1:],
        "waypoint_xy": labels["waypoint_xy"][1:],
        "waypoint_valid": labels["waypoint_valid"][1:],
        "normalizer": labels["normalizer"][1:],
    }
    checks = iv._alignment_checks(labels, vlabels)
    assert checks["strict_geometry_alignment_pass"] is True
    assert checks["source_file_text_match"] is False
    assert checks["scene_id_text_match"] is False


def test_bootstrap_reports_easy_degradation_as_nonnegative():
    selected = np.asarray([1.0, 1.1, 0.8, 0.9, 1.0] * 20)
    floor = np.ones_like(selected)
    mask = np.ones(len(selected), dtype=bool)
    row = iv._bootstrap(selected, floor, mask, easy=True, seed=1)
    assert row["bootstrap_n"] == iv.BOOTSTRAP_N
    assert row["rows"] == len(selected)
    assert row["mean"] >= 0.0


def test_stage42_iv_run_builds_cache_and_keeps_claim_boundary():
    result = iv.run_stage42_source_level_row_cache_integration()
    gate = result["stage42_iv_gate"]
    assert gate["gates"]["source_level_cache_built"]
    assert gate["gates"]["ucy_geometry_alignment_pass"]
    assert result["source_level_test_domains"]["TrajNet"] == 37918
    assert result["source_level_test_domains"]["UCY"] == 9540
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
