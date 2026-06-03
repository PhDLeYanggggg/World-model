from __future__ import annotations

from src import stage43_t100_bounded_alpha_head_failure_forensics as dg


def _seed(selected_min: float, positive_exists: bool) -> dict:
    return {
        "max_replay_diff": 0.0,
        "candidate_count": 10,
        "positive_group_candidate_count": 3 if positive_exists else 0,
        "selected_test": {
            "t100": 0.001,
            "min_without_group_t100": selected_min,
            "easy_degradation": 0.0,
            "switch_rate": 0.1,
        },
        "selection_gap": {
            "oracle_min_minus_selected_min": 0.002,
            "oracle_t100_minus_selected_t100": 0.001,
            "selected_is_test_group_positive": selected_min > 0.0,
            "positive_candidate_exists": positive_exists,
        },
    }


def test_aggregate_identifies_validation_selection_gap() -> None:
    runs = [_seed(-0.001, True), _seed(0.001, True), _seed(0.001, True)]
    agg = dg._aggregate(runs)
    assert agg["positive_candidate_exists_all_seeds"] is True
    assert agg["selected_group_positive_all_seeds"] is False
    assert agg["selection_misses_safe_candidate"] is True
    assert agg["failure_root"] == "validation_group_risk_selection_gap"


def test_gate_passes_for_complete_forensics() -> None:
    runs = [_seed(-0.001, True), _seed(0.001, True), _seed(0.001, True)]
    agg = dg._aggregate(runs)
    payload = {
        "stage43_df_precondition": {"verdict": "stage43_df_t100_bounded_alpha_distilled_head_incomplete"},
        "result_source": "fresh_bounded_alpha_head_selection_forensics",
        "seed_runs": runs,
        "aggregate": agg,
        "deploy_on_current_heldout": False,
        "analysis_protocol": {"test_oracle_used_for_diagnosis_only": True},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
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
    gate = dg._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_dg_t100_bounded_alpha_head_forensics_selection_gap_identified"
