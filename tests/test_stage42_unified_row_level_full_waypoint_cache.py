import numpy as np

from src import stage42_unified_row_level_full_waypoint_cache as x


def test_ucy_alignment_requires_ordered_waypoint_match():
    labels = {
        "domain": np.asarray(["UCY", "UCY", "ETH_UCY"]),
        "source_file": np.asarray(["a", "b", "c"]),
        "scene_id": np.asarray(["s1", "s2", "s3"]),
        "horizon": np.asarray([50, 100, 50]),
        "current_xy": np.asarray([[0.0, 0.0], [1.0, 1.0], [9.0, 9.0]]),
        "future_xy": np.asarray([[2.0, 0.0], [2.0, 1.0], [9.0, 9.0]]),
        "waypoint_xy": np.zeros((3, 4, 2)),
        "waypoint_valid": np.ones((3, 4), dtype=bool),
        "normalizer": np.asarray([1.0, 2.0, 3.0]),
    }
    vlabels = {k: np.asarray(v) for k, v in labels.items()}
    vlabels["domain"] = np.asarray(["UCY", "UCY", "OTHER"])
    result = x._assert_ucy_alignment(labels, vlabels)
    assert result["global_ucy_rows"] == 2
    assert result["stage42v_ucy_rows"] == 2
    assert result["waypoint_xy_match"] is True


def test_summary_tracks_t50_and_easy_from_merged_metrics():
    rows = [
        {
            "pair_idx": 0,
            "merged_test_metrics": {
                "ade": {"all_improvement": 0.1, "t50_improvement": 0.2, "t100_improvement": 0.3, "hard_failure_improvement": 0.4, "easy_degradation": 0.01},
                "fde": {"t50_improvement": 0.5},
                "switch_rate": 0.2,
            },
        },
        {
            "pair_idx": 1,
            "merged_test_metrics": {
                "ade": {"all_improvement": 0.3, "t50_improvement": 0.4, "t100_improvement": 0.5, "hard_failure_improvement": 0.6, "easy_degradation": 0.02},
                "fde": {"t50_improvement": 0.7},
                "switch_rate": 0.4,
            },
        },
    ]
    summary = x._summary(rows)
    assert summary["ade_all"]["mean"] == 0.2
    assert summary["ade_t50"]["mean"] == 0.30000000000000004
    assert summary["ade_easy_degradation"]["mean"] == 0.015
    assert summary["fde_t50"]["mean"] == 0.6


def test_stage42_x_run_records_no_overclaim():
    result = x.run_stage42_unified_row_level_full_waypoint_cache()
    gate = result["stage42_x_gate"]
    assert gate["gates"]["row_level_cache_built"]
    assert gate["gates"]["ucy_alignment_pass"]
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
