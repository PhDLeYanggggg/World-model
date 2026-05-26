from __future__ import annotations

import numpy as np

from src import stage42_common_validation_composer_safety as cp


def test_bootstrap_reports_positive_interval_for_uniform_gain() -> None:
    selected = np.full(100, 9.0)
    ref = np.full(100, 10.0)
    mask = np.ones(100, dtype=bool)
    row = cp._bootstrap(selected, ref, mask, seed=1)
    assert row["bootstrap_n"] == cp.BOOTSTRAP_N
    assert row["low"] > 0.09
    assert row["high"] < 0.11


def test_delta_stats_tracks_collision_and_smoothness() -> None:
    a = {
        "near_collision_rate_002": 0.1,
        "near_collision_rate_005": 0.2,
        "p05_min_group_distance": 0.3,
        "mean_min_group_distance": 0.4,
        "smoothness": {"jagged_rate": 0.01, "mean_max_normalized_step": 0.5},
    }
    b = {
        "near_collision_rate_002": 0.08,
        "near_collision_rate_005": 0.19,
        "p05_min_group_distance": 0.35,
        "mean_min_group_distance": 0.38,
        "smoothness": {"jagged_rate": 0.02, "mean_max_normalized_step": 0.45},
    }
    delta = cp._delta_stats(a, b)
    assert delta["near_collision_rate_005_delta"] > 0.0
    assert delta["p05_min_group_distance_delta"] < 0.0
    assert delta["jagged_rate_delta"] < 0.0


def test_gate_passes_with_safe_bootstrap_and_joint_stats() -> None:
    boot = {
        "all": {"low": 0.01, "mid": 0.02, "high": 0.03, "n": 100, "bootstrap_n": cp.BOOTSTRAP_N},
        "t50": {"low": -0.01, "mid": 0.01, "high": 0.03, "n": 50, "bootstrap_n": cp.BOOTSTRAP_N},
        "t100": {"low": 0.02, "mid": 0.04, "high": 0.06, "n": 50, "bootstrap_n": cp.BOOTSTRAP_N},
        "hard_failure": {"low": -0.01, "mid": 0.02, "high": 0.04, "n": 60, "bootstrap_n": cp.BOOTSTRAP_N},
    }
    payload = {
        "inputs": {"stage42_co": {"stage42_co_gate": {"passed": 14, "total": 14}}},
        "test_metric_vs_endpoint_ade": {
            "t50_improvement": 0.01,
            "hard_failure_improvement": 0.02,
            "easy_degradation": 0.002,
        },
        "bootstrap_vs_endpoint_ade": boot,
        "joint_safety": {
            "composer_minus_floor": {"near_collision_rate_005_delta": -0.001},
            "composer_minus_endpoint": {
                "near_collision_rate_005_delta": 0.003,
                "jagged_rate_delta": 0.0,
            },
        },
        "paper_file_status": [{"contains_stage42_cp": True}],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = cp._gate(payload)
    assert gate["verdict"] == "stage42_cp_common_validation_composer_safety_pass"
    assert gate["passed"] == gate["total"]
