from __future__ import annotations

from src import stage42_proximity_guard_ablation as cr


def test_delta_handles_none_and_numeric_values() -> None:
    after = {"all_improvement": 0.10, "near_collision_005_delta_vs_endpoint": -0.01}
    before = {"all_improvement": 0.20, "near_collision_005_delta_vs_endpoint": 0.03}
    delta = cr._delta(after, before)
    assert delta["all_improvement"] < 0.0
    assert delta["near_collision_005_delta_vs_endpoint"] < 0.0
    assert delta["p05_min_distance_delta_vs_endpoint"] is None


def test_metric_row_extracts_joint_deltas() -> None:
    row = cr._metric_row(
        "guard",
        "safe",
        {"all_improvement": 0.1, "switch_rate": 0.2},
        {
            "near_collision_rate_005_delta": -0.01,
            "p05_min_group_distance_delta": 0.02,
            "jagged_rate_delta": 0.0,
        },
    )
    assert row["name"] == "guard"
    assert row["all_improvement"] == 0.1
    assert row["near_collision_005_delta_vs_endpoint"] < 0.0


def test_gate_passes_when_guard_trades_accuracy_for_safety() -> None:
    rows = {
        "endpoint_linear_reference": {"all_improvement": 0.0},
        "no_proximity_guard": {
            "all_improvement": 0.03,
            "t50_improvement": 0.02,
            "t100_raw_frame_diagnostic_improvement": 0.06,
            "hard_failure_improvement": 0.03,
            "easy_degradation": 0.002,
            "near_collision_005_delta_vs_endpoint": 0.003,
        },
        "proximity_guard": {
            "all_improvement": 0.02,
            "t50_improvement": 0.01,
            "t100_raw_frame_diagnostic_improvement": 0.03,
            "hard_failure_improvement": 0.02,
            "easy_degradation": 0.002,
            "near_collision_005_delta_vs_endpoint": -0.001,
        },
    }
    payload = {
        "inputs": {
            "stage42_co": {"stage42_co_gate": {"passed": 14, "total": 14}},
            "stage42_cp": {"stage42_cp_gate": {"passed": 14, "total": 14}},
            "stage42_cq": {"stage42_cq_gate": {"passed": 19, "total": 19}},
        },
        "ablation_rows": rows,
        "guard_contribution": {
            "safety_delta_vs_no_guard": cr._delta(rows["proximity_guard"], rows["no_proximity_guard"]),
            "accuracy_cost_vs_no_guard": cr._delta(rows["proximity_guard"], rows["no_proximity_guard"]),
        },
        "deployment_recommendation": {"safety_sensitive_policy": "proximity_guard"},
        "paper_file_status": [{"contains_stage42_cr": True}],
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = cr._gate(payload)
    assert gate["verdict"] == "stage42_cr_proximity_guard_ablation_pass"
    assert gate["passed"] == gate["total"]
