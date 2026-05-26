from __future__ import annotations

from src.stage42_proximity_guard_runtime_policy import FrozenProximityGuardPolicy, _gate


def _policy() -> FrozenProximityGuardPolicy:
    return FrozenProximityGuardPolicy(
        {
            "guard_rule": {"min_sep": 0.2, "margin": 0.005},
            "base_choices": {"ETH_UCY|50": True, "ETH_UCY|100": True, "TrajNet|50": False},
        },
        policy_hash="hash",
    )


def test_runtime_decision_uses_full_when_guard_clear() -> None:
    decision = _policy().decide(
        domain="ETH_UCY",
        horizon=50,
        endpoint_min_group_distance=0.4,
        full_min_group_distance=0.3,
    )
    assert decision.use_full_waypoint is True
    assert decision.guarded_off is False
    assert decision.reason == "base_choice_full_waypoint_guard_clear"


def test_runtime_decision_guards_off_when_full_waypoint_is_too_close() -> None:
    decision = _policy().decide(
        domain="ETH_UCY",
        horizon=50,
        endpoint_min_group_distance=0.4,
        full_min_group_distance=0.1,
    )
    assert decision.use_full_waypoint is False
    assert decision.guarded_off is True
    assert decision.reason == "proximity_guard_fallback_to_endpoint_linear"


def test_runtime_decision_keeps_endpoint_for_base_endpoint_slice() -> None:
    decision = _policy().decide(
        domain="TrajNet",
        horizon=50,
        endpoint_min_group_distance=0.4,
        full_min_group_distance=0.1,
    )
    assert decision.use_full_waypoint is False
    assert decision.guarded_off is False
    assert decision.reason == "base_choice_endpoint_linear"


def test_runtime_decision_replays_nonfinite_geometry_behavior() -> None:
    decision = _policy().decide(
        domain="ETH_UCY",
        horizon=100,
        endpoint_min_group_distance=None,
        full_min_group_distance=0.1,
    )
    assert decision.use_full_waypoint is True
    assert decision.guarded_off is False
    assert decision.reason == "base_choice_full_waypoint_geometry_nonfinite_replay_no_guard"


def test_gate_passes_for_runtime_smoke_payload() -> None:
    payload = {
        "policy_artifact": {"exists": True},
        "policy_hash": "hash",
        "inputs": {"stage42_ct": {"stage42_ct_gate": {"passed": 30, "total": 30}}},
        "runtime_policy": {
            "min_sep": 0.2,
            "margin": 0.005,
            "base_choices": {"ETH_UCY|50": True},
        },
        "policy_artifact_payload": {
            "guard_rule": {"min_sep": 0.2, "margin": 0.005},
            "base_choices": {"ETH_UCY|50": True},
        },
        "smoke_cases": {
            "full_slice_guard_clear": {"passed": True},
            "full_slice_guarded_off": {"passed": True},
            "endpoint_slice_never_switches": {"passed": True},
            "full_slice_nonfinite_geometry_replays_no_guard": {"passed": True},
        },
        "runtime_inputs": [
            "domain",
            "horizon",
            "endpoint_min_group_distance_from_predicted_endpoint_rollout",
            "full_min_group_distance_from_predicted_full_waypoint_rollout",
        ],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = _gate(payload)
    assert gate["verdict"] == "stage42_cu_runtime_policy_api_pass"
    assert gate["passed"] == gate["total"]
