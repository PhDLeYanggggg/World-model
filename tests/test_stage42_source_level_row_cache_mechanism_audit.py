import numpy as np

from src import stage42_source_level_row_cache_mechanism_audit as iw


def test_waypoint_shape_summary_detects_non_linear_waypoints():
    cache = {
        "waypoint_xy": np.asarray(
            [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [4.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            ],
            dtype=float,
        ),
        "waypoint_valid": np.ones((2, 4), dtype=bool),
        "current_xy": np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float),
        "future_xy": np.asarray([[4.0, 0.0], [4.0, 0.0]], dtype=float),
    }
    out = iw._waypoint_shape_summary(cache)
    assert out["full_waypoint_rate"] == 1.0
    assert out["mean_raw_residual_from_linear_bridge"] > 0.0
    assert out["mean_turn_angle_radians"] > 0.0


def test_switch_summary_requires_fallback_exact_floor():
    cache = {
        "selected_ade_seed_mean": np.asarray([0.8, 1.0, 0.9, 1.0], dtype=float),
        "floor_ade": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=float),
        "switch_seed_mean": np.asarray([1.0, 0.0, 1.0, 0.0], dtype=float),
    }
    labels = {
        "domain": np.asarray(["A", "A", "B", "B"]),
        "horizon": np.asarray([50, 50, 100, 100]),
        "hard": np.asarray([True, False, True, False]),
        "failure": np.asarray([False, False, False, False]),
        "easy": np.asarray([False, True, False, True]),
    }
    out = iw._switch_summary(cache, labels)
    assert out["switch_rows"] == 2
    assert out["fallback_exact_floor_rate"] == 1.0
    assert out["mean_gain_switched_rows"] > 0.0


def test_stage42_iw_run_builds_mechanism_audit():
    result = iw.run_stage42_source_level_row_cache_mechanism_audit()
    gate = result["stage42_iw_gate"]
    assert gate["gates"]["stage42iv_cache_loaded"]
    assert gate["gates"]["safe_switch_has_positive_mean_gain"]
    assert gate["gates"]["fallback_rows_match_floor"]
    assert gate["gates"]["waypoint_sequence_labels_available"]
    assert gate["gates"]["full_waypoint_completeness_reported"]
    assert result["rows"] == 47458
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
