from __future__ import annotations

from src.stage42_explicit_physical_consistency_checkpoint import _gate, _summary_from_inputs


def _loss_candidate(name: str, all_i: float, hard_i: float, promotable: bool) -> dict:
    return {
        "name": name,
        "promotable_over_stage42_am": promotable,
        "metric_vs_floor": {
            "all_improvement": all_i,
            "t50_improvement": 0.2,
            "t100_raw_frame_diagnostic_improvement": 0.1,
            "hard_failure_improvement": hard_i,
            "easy_degradation": -0.1,
        },
    }


def _group_result() -> dict:
    return {
        "source": "fresh_stage42_di_group_consistency_full_waypoint_repair",
        "deployment_decision": {"promote_group_consistency_full_waypoint_repair": True},
        "repair": {
            "test": {
                "metric_vs_floor": {
                    "all_improvement": 0.247,
                    "t50_improvement": 0.223,
                    "t100_raw_frame_diagnostic_improvement": 0.143,
                    "hard_failure_improvement": 0.239,
                    "easy_degradation": -0.25,
                },
                "diagnostics": {"base_near_005": 0.02, "final_near_005": 0.01},
            }
        },
        "comparison_to_prior": {
            "delta_vs_stage42_am": {
                "all_improvement": 0.001,
                "hard_failure_improvement": 0.001,
            }
        },
    }


def test_summary_records_loss_all_advantage_and_group_hard_advantage() -> None:
    summary = _summary_from_inputs(
        [
            _loss_candidate("all_hard_weighted_loss", 0.245, 0.237, False),
            _loss_candidate("proximity_occupancy_loss", 0.255, 0.238, False),
        ],
        _group_result(),
    )

    assert summary["loss_family_any_promotable_over_stage42_am"] is False
    assert summary["best_loss_family_candidate"] == "proximity_occupancy_loss"
    assert summary["group_consistency_promotable_over_stage42_am"] is True
    assert summary["group_consistency_delta_all_vs_best_loss_family"] < 0.0
    assert summary["group_consistency_delta_hard_vs_best_loss_family"] > 0.0


def test_gate_passes_source_level_physical_consistency_checkpoint() -> None:
    group = _group_result()
    payload = {
        "dg_result": {"source": "fresh_stage42_dg_full_waypoint_all_hard_loss_repair"},
        "dh_result": {"source": "fresh_stage42_dh_full_waypoint_proximity_occupancy_loss_repair"},
        "group_consistency_result": group,
        "summary": _summary_from_inputs(
            [
                _loss_candidate("all_hard_weighted_loss", 0.245, 0.237, False),
                _loss_candidate("proximity_occupancy_loss", 0.255, 0.238, False),
            ],
            group,
        ),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "global_primary_full_waypoint_replacement_claim_allowed": False,
            "ungated_full_waypoint_deployable": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_dy_explicit_physical_consistency_checkpoint_pass_source_level_promoted"
