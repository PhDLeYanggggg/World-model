from __future__ import annotations

from src import stage43_t100_residual_admissibility_statistical_confirmation as cu


def test_parse_seeds() -> None:
    assert cu._parse_seeds("1, 2,3") == [1, 2, 3]


def test_aggregate_detects_stable_positive() -> None:
    runs = []
    for value in [0.001, 0.002, 0.003]:
        runs.append(
            {
                "test_metrics_with_floor": {
                    "full_waypoint_ade_improvement_vs_floor": value,
                    "t100_raw_frame_full_waypoint_diagnostic_vs_floor": value,
                    "hard_failure_full_waypoint_ade_improvement_vs_floor": value,
                    "easy_degradation_vs_floor": 0.0,
                    "switch_rate": 0.1,
                },
                "bootstrap_ci": {
                    "metrics": {
                        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": {
                            "low": value / 2.0,
                        }
                    }
                },
            }
        )
    agg = cu._aggregate(runs)
    assert agg["all_seed_t100_positive"] is True
    assert agg["all_seed_bootstrap_low_positive"] is True
    assert agg["all_seed_easy_safe"] is True


def test_gate_fails_if_bootstrap_not_confirmed() -> None:
    payload = {
        "stage43_ct_precondition": {"verdict": "stage43_ct_t100_residual_admissibility_positive_diagnostic"},
        "result_source": "fresh_torch_t100_residual_admissibility_multiseed_confirmation",
        "seed_runs": [
            {
                "checkpoint": "README_RESULTS.md",
                "checkpoint_committed": False,
                "test_metrics_with_floor": {"rows": 10},
            }
            for _ in range(3)
        ],
        "feature_contract": {"denied_feature_name_hits": []},
        "aggregate": {
            "all_seed_t100_positive": True,
            "all_seed_bootstrap_low_positive": False,
            "all_seed_easy_safe": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "long_objective_complete": False,
    }
    gate = cu._gate(payload)
    assert gate["passed"] == gate["total"] - 1
    assert gate["verdict"] == "stage43_cu_t100_admissibility_multiseed_inconclusive_keep_floor"
