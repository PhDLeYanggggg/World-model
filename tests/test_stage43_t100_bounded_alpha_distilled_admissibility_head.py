from __future__ import annotations

from src import stage43_t100_bounded_alpha_distilled_admissibility_head as df


def _seed(t100: float, min_without: float, checkpoint: str, easy: float = 0.0) -> dict:
    return {
        "checkpoint": checkpoint,
        "checkpoint_committed": False,
        "test_metrics_with_floor": {
            "rows": 10,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": t100,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": t100,
            "easy_degradation_vs_floor": easy,
            "switch_rate": 0.1,
        },
        "test_group_summary": {
            "min_without_any_group_t100": min_without,
        },
        "bootstrap_ci": {
            "metrics": {
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": {
                    "low": t100 - 0.0001,
                },
            },
        },
        "teacher_switch_rate_train": 0.12,
    }


def test_aggregate_tracks_bounded_head_against_de_reference(tmp_path) -> None:
    ckpt = tmp_path / "stage43_df_test.pt"
    ckpt.write_text("fixture")
    runs = [_seed(0.002, 0.001, str(ckpt)) for _ in range(3)]
    agg = df._aggregate(
        runs,
        {"aggregate": {"bounded_t100": {"mean": 0.0018}, "bounded_min_without_group_t100": {"mean": 0.0008}}},
        {"aggregate": {"t100": {"mean": 0.0022}, "min_without_group_t100": {"mean": -0.0001}}},
    )
    assert agg["all_seed_t100_positive"] is True
    assert agg["all_min_without_group_positive"] is True
    assert agg["beats_de_t100_mean"] is True
    assert agg["beats_dc_min_without_group_mean"] is True


def test_gate_passes_for_safe_bounded_alpha_distilled_head(tmp_path) -> None:
    ckpt = tmp_path / "stage43_df_test.pt"
    ckpt.write_text("fixture")
    runs = [_seed(0.002, 0.001, str(ckpt)) for _ in range(3)]
    agg = df._aggregate(
        runs,
        {"aggregate": {"bounded_t100": {"mean": 0.0018}, "bounded_min_without_group_t100": {"mean": 0.0008}}},
        {"aggregate": {"t100": {"mean": 0.0022}, "min_without_group_t100": {"mean": -0.0001}}},
    )
    payload = {
        "stage43_de_precondition": {"verdict": "stage43_de_t100_alpha_stability_policy_repairs_group_fragility_diagnostic"},
        "result_source": "fresh_torch_bounded_alpha_policy_distilled_t100_head",
        "seed_runs": runs,
        "training_protocol": {"teacher": "stage43_de_bounded_alpha_policy", "alpha_cap": 0.75},
        "selection_protocol": {"test_threshold_tuning": False},
        "feature_contract": {"denied_feature_name_hits": []},
        "aggregate": agg,
        "deploy_on_current_heldout": False,
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
    gate = df._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["gates"]["all_checkpoints_written_not_committed"] is True
    assert gate["verdict"] == "stage43_df_t100_bounded_alpha_distilled_head_beats_de_diagnostic"
