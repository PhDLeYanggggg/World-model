from __future__ import annotations

import numpy as np

from src.stage42_group_consistency_runtime_policy import FrozenGroupConsistencyPolicy, _gate


def _repel_policy() -> FrozenGroupConsistencyPolicy:
    return FrozenGroupConsistencyPolicy(
        {
            "repair_rule": {
                "type": "repel_unsafe",
                "min_sep": 0.08,
                "margin": 0.0,
                "strength": 0.5,
            }
        },
        policy_hash="hash",
    )


def test_runtime_repel_policy_improves_min_distance() -> None:
    policy = _repel_policy()
    base = np.asarray([[[0.02, 0.0], [0.02, 0.0]], [[0.03, 0.0], [0.03, 0.0]]], dtype=np.float32)
    floor = np.asarray([[[0.0, 0.0], [0.0, 0.0]], [[0.2, 0.0], [0.2, 0.0]]], dtype=np.float32)
    result = policy.apply(
        base_xy=base,
        floor_xy=floor,
        pred_xy=base.copy(),
        base_switch=np.asarray([True, True]),
        group_key=np.asarray(["g", "g"], dtype=object),
        normalizer=np.ones(2, dtype=np.float64),
        agent_id=np.asarray([1, 2], dtype=np.int64),
        current_xy=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    diag = result.diagnostics()
    assert diag["final_near_005"] <= diag["base_near_005"]
    assert diag["final_p05_min_distance"] > diag["base_p05_min_distance"]


def test_runtime_fallback_policy_disables_unsafe_switch() -> None:
    policy = FrozenGroupConsistencyPolicy(
        {"repair_rule": {"type": "fallback_unsafe", "min_sep": 0.08, "margin": 0.0}},
        policy_hash="hash",
    )
    base = np.asarray([[[0.02, 0.0], [0.02, 0.0]], [[0.03, 0.0], [0.03, 0.0]]], dtype=np.float32)
    floor = np.asarray([[[0.0, 0.0], [0.0, 0.0]], [[0.2, 0.0], [0.2, 0.0]]], dtype=np.float32)
    result = policy.apply(
        base_xy=base,
        floor_xy=floor,
        pred_xy=base.copy(),
        base_switch=np.asarray([True, True]),
        group_key=np.asarray(["g", "g"], dtype=object),
        normalizer=np.ones(2, dtype=np.float64),
        agent_id=np.asarray([1, 2], dtype=np.int64),
        current_xy=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    assert result.switch.tolist() == [False, False]
    assert np.allclose(result.selected_xy, floor)


def test_gate_passes_for_runtime_replay_payload() -> None:
    payload = {
        "policy_artifact": {"exists": True},
        "policy_hash": "hash",
        "inputs": {"stage42_dk": {"stage42_dk_gate": {"passed": 34, "total": 34}}},
        "runtime_policy": {"mode": "repel_unsafe", "min_sep": 0.08, "margin": 0.0, "strength": 0.5},
        "policy_artifact_payload": {"repair_rule": {"type": "repel_unsafe", "min_sep": 0.08, "strength": 0.5}},
        "smoke_case": {"passes": True},
        "real_batch_replay": {
            "rows": 10,
            "selected_xy_max_abs_diff": 0.0,
            "switch_exact_match": True,
            "selected_ade_max_abs_diff": 0.0,
            "selected_fde_max_abs_diff": 0.0,
            "metric_abs_diff": {
                "all_improvement": 0.0,
                "t50_improvement": 0.0,
                "t100_raw_frame_diagnostic_improvement": 0.0,
                "hard_failure_improvement": 0.0,
                "easy_degradation": 0.0,
                "switch_rate": 0.0,
            },
            "diagnostic_abs_diff": {"base_near_005": 0.0, "final_near_005": 0.0, "floor_near_005": 0.0},
            "metric": {
                "all_improvement": 0.2,
                "t50_improvement": 0.1,
                "t100_raw_frame_diagnostic_improvement": 0.1,
                "hard_failure_improvement": 0.2,
                "easy_degradation": 0.0,
            },
            "diagnostics": {"base_near_005": 0.02, "final_near_005": 0.01},
        },
        "runtime_inputs": [
            "base_xy_predicted_full_waypoint_candidate",
            "floor_xy_train_horizon_causal_rollout",
            "pred_xy_model_rollout_diagnostic",
            "base_switch_from_validation_selected_policy",
            "source_frame_horizon_group_key",
            "normalizer",
            "agent_id",
            "current_xy",
        ],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "paper_file_status": [{"exists": True, "contains_stage42_dl": True}],
    }
    gate = _gate(payload)
    assert gate["verdict"] == "stage42_dl_group_consistency_runtime_policy_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_runtime_does_not_exactly_replay() -> None:
    payload = {
        "policy_artifact": {"exists": True},
        "policy_hash": "hash",
        "inputs": {"stage42_dk": {"stage42_dk_gate": {"passed": 34, "total": 34}}},
        "runtime_policy": {"mode": "repel_unsafe", "min_sep": 0.08, "margin": 0.0, "strength": 0.5},
        "policy_artifact_payload": {"repair_rule": {"type": "repel_unsafe", "min_sep": 0.08, "strength": 0.5}},
        "smoke_case": {"passes": True},
        "real_batch_replay": {
            "rows": 10,
            "selected_xy_max_abs_diff": 0.01,
            "switch_exact_match": True,
            "selected_ade_max_abs_diff": 0.0,
            "selected_fde_max_abs_diff": 0.0,
            "metric_abs_diff": {"all_improvement": 0.0},
            "diagnostic_abs_diff": {"base_near_005": 0.0},
            "metric": {
                "all_improvement": 0.2,
                "t50_improvement": 0.1,
                "t100_raw_frame_diagnostic_improvement": 0.1,
                "hard_failure_improvement": 0.2,
                "easy_degradation": 0.0,
            },
            "diagnostics": {"base_near_005": 0.02, "final_near_005": 0.01},
        },
        "runtime_inputs": [
            "base_xy_predicted_full_waypoint_candidate",
            "floor_xy_train_horizon_causal_rollout",
            "pred_xy_model_rollout_diagnostic",
            "base_switch_from_validation_selected_policy",
            "source_frame_horizon_group_key",
            "normalizer",
            "agent_id",
            "current_xy",
        ],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "paper_file_status": [{"exists": True, "contains_stage42_dl": True}],
    }
    gate = _gate(payload)
    assert gate["gates"]["real_batch_selected_xy_exact"] is False
    assert gate["passed"] == gate["total"] - 1
