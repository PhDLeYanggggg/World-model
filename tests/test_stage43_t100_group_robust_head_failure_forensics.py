from __future__ import annotations

from src import stage43_t100_group_robust_head_failure_forensics as db


def _cz_seed(seed: int, t100: float, min_without: float, switch: float) -> dict:
    return {
        "seed": seed,
        "selected_validation_candidate": {"policy": {"alpha_index": 1}},
        "robust_test_metrics": {
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": t100,
            "switch_rate": switch,
        },
        "robust_test_group_summary": {
            "min_without_any_group_t100": min_without,
            "scene_group_flip_count": 0,
        },
    }


def _da_seed(seed: int, t100: float, min_without: float, val_min: float, switch: float) -> dict:
    return {
        "seed": seed,
        "validation_selected_policy": {
            "policy": {"alpha_index": 2},
            "group_summary": {"min_without_any_group_t100": val_min},
            "safe_candidates": 10,
        },
        "test_metrics_with_floor": {
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": t100,
            "switch_rate": switch,
            "easy_degradation_vs_floor": 0.0,
        },
        "test_group_summary": {
            "min_without_any_group_t100": min_without,
            "scene_group_flip_count": 1,
        },
        "bootstrap_ci": {
            "metrics": {
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": {"low": 0.0001}
            }
        },
        "best_epoch": 3,
    }


def test_align_seed_runs_computes_da_minus_cz() -> None:
    cz_report = {"seed_runs": [_cz_seed(1, 0.003, 0.002, 0.12)]}
    da_report = {"seed_runs": [_da_seed(1, 0.001, -0.001, 0.002, 0.05)]}
    rows = db._align_seed_runs(cz_report, da_report)
    assert len(rows) == 1
    assert rows[0]["delta_t100_da_minus_cz"] < 0.0
    assert rows[0]["delta_min_without_group_da_minus_cz"] < 0.0
    assert rows[0]["delta_switch_rate_da_minus_cz"] < 0.0


def test_aggregate_identifies_primary_root_causes() -> None:
    rows = [
        db._align_seed_runs(
            {"seed_runs": [_cz_seed(seed, 0.003, 0.002, 0.12)]},
            {"seed_runs": [_da_seed(seed, 0.001, -0.001, 0.002, 0.05)]},
        )[0]
        for seed in [1, 2, 3]
    ]
    agg = db._aggregate(rows)
    assert agg["root_causes"]["trained_head_underperforms_policy_only"] is True
    assert agg["root_causes"]["group_worst_case_not_preserved"] is True
    assert agg["root_causes"]["under_switching_relative_to_cz"] is True
    assert agg["root_causes"]["not_an_easy_safety_failure"] is True


def test_gate_requires_forensics_and_repair_hypotheses() -> None:
    rows = [
        db._align_seed_runs(
            {"seed_runs": [_cz_seed(seed, 0.003, 0.002, 0.12)]},
            {"seed_runs": [_da_seed(seed, 0.001, -0.001, 0.002, 0.05)]},
        )[0]
        for seed in [1, 2, 3]
    ]
    agg = db._aggregate(rows)
    payload = {
        "input_reports": {"cz_gate_passed": True, "da_gate_passed": True},
        "aggregate": agg,
        "repair_hypotheses": db._repair_hypotheses(agg),
        "deploy_on_current_heldout": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    gate = db._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_db_t100_head_failure_forensics_complete_policy_distill_next"
