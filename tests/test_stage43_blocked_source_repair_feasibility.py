from __future__ import annotations

from src import stage43_blocked_source_repair_feasibility as bb


def test_validation_rows_for_family_extracts_allowed_and_reasons() -> None:
    table = {
        "TrajNet_biwi|10": {
            "rows": 255,
            "allowed": False,
            "full_waypoint_ade_improvement_vs_floor": 0.18,
            "easy_degradation_vs_floor": 0.0,
            "reason": "blocked_insufficient_validation_support",
        },
        "TrajNet_biwi|50": {
            "rows": 1200,
            "allowed": True,
            "full_waypoint_ade_improvement_vs_floor": 0.12,
            "easy_degradation_vs_floor": 0.0,
            "reason": "allowed_by_validation",
        },
        "UCY|50": {
            "rows": 2000,
            "allowed": True,
            "full_waypoint_ade_improvement_vs_floor": 0.5,
            "easy_degradation_vs_floor": 0.0,
            "reason": "allowed_by_validation",
        },
    }
    out = bb._validation_rows_for_family(table, "TrajNet_biwi")
    assert out["total_rows"] == 1455
    assert out["allowed_horizons"] == [50]
    assert out["block_reasons"] == {"blocked_insufficient_validation_support": 1}
    assert out["horizons"]["10"]["reason"] == "blocked_insufficient_validation_support"


def test_repair_decision_blocks_catastrophic_ungated_source() -> None:
    decision = bb._repair_decision(
        blocked={"ungated_improvement": -3.0, "easy_degradation": 0.0},
        validation={"total_rows": 1500, "allowed_horizons": [50]},
        support={
            "train": {"family_rows": 10000},
            "val": {"family_rows": 1500},
            "test": {"source_rows": 500},
        },
        min_validation_rows=1000,
        max_easy_degradation=0.02,
    )
    assert decision["repairable_now"] is False
    assert decision["status"] == "not_repairable_now_keep_floor"
    assert "ungated_transfer_catastrophic_negative" in decision["blockers"]


def test_repair_decision_allows_only_when_every_support_gate_clears() -> None:
    decision = bb._repair_decision(
        blocked={"ungated_improvement": 0.15, "easy_degradation": 0.0},
        validation={"total_rows": 2000, "allowed_horizons": [50]},
        support={
            "train": {"family_rows": 10000},
            "val": {"family_rows": 2000},
            "test": {"source_rows": 500},
        },
        min_validation_rows=1000,
        max_easy_degradation=0.02,
    )
    assert decision["repairable_now"] is True
    assert decision["status"] == "repairable_with_validation_guard"


def test_gate_passes_when_unsafe_repair_is_correctly_blocked() -> None:
    payload = {
        "input_verdicts": {
            "stage43_ba": "stage43_ba_tail_adapter_source_blocker_audit_pass",
        },
        "repair_protocol": {
            "diagnostic_test_rows_not_used_for_training": True,
            "test_threshold_tuning_allowed": False,
        },
        "blocked_source_rows": [
            {
                "split_support": {"train": {}, "val": {}, "test": {}},
                "validation_support": {"total_rows": 400, "horizons": {}},
            }
        ],
        "summary": {
            "blocked_source_count": 1,
            "repairable_now_count": 0,
            "floor_only_count": 1,
            "catastrophic_ungated_count": 1,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "uniform_positive_external_transfer_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "next_required_actions": ["a", "b", "c"],
    }
    gate = bb._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_bb_blocked_source_repair_feasibility_pass"
