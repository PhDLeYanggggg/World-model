import numpy as np

from src.stage42_t50_ensemble_source_robustness import _gate, _group_bootstrap_ci, _slice_rows


def test_group_bootstrap_ci_reports_group_count():
    selected = np.array([0.5, 0.5, 1.0, 1.0])
    floor = np.array([1.0, 1.0, 1.0, 1.0])
    mask = np.ones(4, dtype=bool)
    groups = np.array(["a", "a", "b", "b"])
    ci = _group_bootstrap_ci(selected, floor, mask, groups, seed=1, n=20)
    assert ci["n_groups"] == 2
    assert ci["n_rows"] == 4
    assert ci["mid"] >= 0.0


def test_slice_rows_computes_powered_source_metrics():
    labels = {
        "horizon": np.array([50, 50, 100, 10]),
        "hard": np.array([True, False, False, False]),
        "failure": np.array([False, False, True, False]),
        "easy": np.array([False, True, False, True]),
        "source_file": np.array(["s1", "s1", "s1", "s1"]),
    }
    selected = np.array([0.5, 0.8, 0.7, 1.0])
    floor = np.array([1.0, 1.0, 1.0, 1.0])
    rows = _slice_rows(selected, floor, labels, "source_file", min_rows=1)
    assert rows[0]["source_file"] == "s1"
    assert rows[0]["t50_rows"] == 2
    assert rows[0]["t50_improvement"] > 0.0


def test_gate_passes_source_robust_payload():
    payload = {
        "source_labels": {"stage42ii_policy": "cached_verified", "source_scene_robustness_eval": "fresh_run"},
        "summary": {
            "source_count": 4,
            "scene_count": 4,
            "all_improvement": 0.1,
            "t50_improvement": 0.08,
            "t50_source_group_ci_low": 0.01,
            "t50_scene_group_ci_low": 0.0,
            "negative_powered_t50_source_count": 0,
            "hard_failure_improvement": 0.1,
            "easy_degradation": 0.0,
        },
        "no_leakage": {"future_endpoint_input": False, "future_waypoints_input": False, "test_endpoint_goals": False},
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = _gate(payload)
    assert gate["verdict"] == "stage42_ij_t50_ensemble_source_robustness_pass"
    assert gate["passed"] == gate["total"]
