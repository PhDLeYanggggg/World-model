import numpy as np

from src import stage42_source_level_nonlinear_context_slice_audit as iz


def test_train_quantile_uses_train_rows_only():
    values = np.asarray([0.0, 10.0, 1000.0, 2000.0])
    train = np.asarray([True, True, False, False])
    assert iz._train_quantile(values, train, 0.5) == 5.0


def test_slice_masks_include_train_only_thresholds():
    data = {
        "horizon": np.asarray([10, 25, 50, 100, 50]),
        "dataset": np.asarray(["A", "A", "B", "B", "B"]),
        "hard": np.asarray([True, False, True, False, False]),
        "failure": np.asarray([False, False, True, False, False]),
        "easy": np.asarray([False, True, False, True, True]),
        "history_scalar": np.asarray(
            [
                [1.0, 1.0, 5.0, 0.1, 9.0, 0.0, 0.1, 0.1, 8.0],
                [2.0, 2.0, 4.0, 0.2, 8.0, 0.0, 0.2, 0.2, 8.0],
                [3.0, 3.0, 3.0, 0.3, 7.0, 0.0, 0.3, 0.3, 8.0],
                [4.0, 4.0, 2.0, 0.4, 6.0, 0.0, 0.4, 0.4, 8.0],
                [5.0, 5.0, 1.0, 0.5, 5.0, 0.0, 0.5, 0.5, 8.0],
            ],
            dtype=float,
        ),
        "goal_ambiguity": np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
    }
    split = np.asarray(["train", "train", "test", "test", "test"])
    slices, thresholds = iz._slice_masks(data, split)
    assert thresholds["path_length_q75"] < 3.0
    assert "domain_horizon:B|50" in slices
    assert np.sum(slices["domain_horizon:B|50"]) == 2
    assert np.sum(slices["hard_failure"]) == 2


def test_gate_records_negative_slice_result():
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "trial_overview": {t["name"]: {} for t in iz.TRIALS},
        "slice_threshold_source": "train_split_quantiles_only",
        "slice_rows_total": 10,
        "summary": {
            "powered_slice_rows": 10,
            "supported_context_slice_count": 0,
            "decision": "context_slice_level_support_not_found",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_slice_thresholds": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = iz._gate(result)
    assert not gate["gates"]["context_slice_claim_supported"]
    assert gate["verdict"] == "stage42_iz_context_slice_audit_completed_context_not_proven"


def test_stage42_iz_run_completes_or_uses_cache():
    result = iz.run_stage42_source_level_nonlinear_context_slice_audit(use_cached=True)
    if not result:
        result = iz.run_stage42_source_level_nonlinear_context_slice_audit()
    assert result["stage42_iz_gate"]["gates"]["slice_audit_complete"]
    assert result["stage42_iz_gate"]["gates"]["slice_thresholds_train_only"]
    assert result["summary"]["decision"] in {
        "context_has_powered_slice_level_support",
        "context_slice_level_support_not_found",
    }
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
