from __future__ import annotations

import numpy as np

from src import stage42_group_consistency_t100_easy_guard_runtime as ht


def _policy_payload() -> dict:
    return {
        "decision_rule": {"threshold_easy_degradation": 0.0},
        "decision_table": {
            "guarded_slices": {"TrajNet|100": {"keep": False}},
            "kept_slices": {"UCY|100": {"keep": True}},
        },
        "test_summary_vs_train_horizon_causal_floor": {
            "all_improvement": 0.2772,
            "t50_improvement": 0.2699,
            "t100_raw_frame_diagnostic_improvement": 0.0679,
            "hard_failure_improvement": 0.2593,
            "easy_degradation": -0.3233,
            "t100_easy_degradation": -0.0031,
            "switch_rate": 0.6816,
        },
    }


def _policy() -> ht.FrozenT100EasyGuardPolicy:
    return ht.FrozenT100EasyGuardPolicy(_policy_payload(), policy_hash="hash")


def _gate_payload() -> dict:
    policy_payload = _policy_payload()
    policy = ht.FrozenT100EasyGuardPolicy(policy_payload, policy_hash="hash")
    return {
        "policy_artifact": {"exists": True},
        "policy_hash": "hash",
        "inputs": {"stage42_hs": {"stage42_hs_gate": {"passed": 27, "total": 27}}},
        "runtime_policy": {
            "guarded_slices": policy.guarded_slices,
            "kept_slices": policy.kept_slices,
            "threshold_easy_degradation": policy.threshold_easy_degradation,
        },
        "policy_artifact_payload": policy_payload,
        "runtime_inputs": [
            "domain",
            "horizon",
            "candidate_xy_predicted_rollout",
            "floor_xy_train_horizon_causal_rollout",
            "candidate_switch_optional",
        ],
        "smoke_case": ht._smoke_cases(policy),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }


def test_decide_handles_supported_guarded_and_unknown_t100_domains() -> None:
    policy = _policy()

    assert policy.decide(domain="TrajNet", horizon=100).fallback_to_floor is True
    assert policy.decide(domain="TrajNet", horizon=100).reason == "validation_easy_harm_t100_fallback_floor"
    assert policy.decide(domain="UCY", horizon=100).use_candidate is True
    assert policy.decide(domain="UCY", horizon=100).reason == "validation_supported_t100_keep_candidate"
    assert policy.decide(domain="TrajNet", horizon=50).reason == "non_t100_not_guarded"
    assert policy.decide(domain="NewDomain", horizon=100).reason == "unknown_t100_domain_no_validation_support_fallback_floor"


def test_apply_falls_back_only_for_guarded_or_unknown_t100_rows() -> None:
    policy = _policy()
    candidate = np.asarray(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[3.0, 0.0], [4.0, 0.0]],
            [[5.0, 0.0], [6.0, 0.0]],
            [[7.0, 0.0], [8.0, 0.0]],
        ],
        dtype=np.float32,
    )
    floor = np.zeros_like(candidate)

    out = policy.apply(
        domains=np.asarray(["TrajNet", "UCY", "TrajNet", "NewDomain"], dtype=object),
        horizons=np.asarray([100, 100, 50, 100], dtype=np.int64),
        candidate_xy=candidate,
        floor_xy=floor,
    )

    assert out.switch.tolist() == [False, True, True, False]
    assert np.allclose(out.selected_xy[[0, 3]], floor[[0, 3]])
    assert np.allclose(out.selected_xy[[1, 2]], candidate[[1, 2]])
    assert out.diagnostics()["fallback_rows"] == 2


def test_gate_passes_for_runtime_policy_payload() -> None:
    gate = ht._gate(_gate_payload())

    assert gate["verdict"] == "stage42_ht_t100_easy_guard_runtime_policy_pass"
    assert gate["passed"] == gate["total"]
    assert gate["gates"]["trajnet_t100_fallbacks"] is True
    assert gate["gates"]["ucy_t100_keeps_candidate"] is True
    assert gate["gates"]["unknown_t100_fallbacks"] is True


def test_gate_fails_if_unknown_t100_domain_does_not_fallback() -> None:
    payload = _gate_payload()
    payload["smoke_case"]["actual_reasons"][3] = "validation_supported_t100_keep_candidate"

    gate = ht._gate(payload)

    assert gate["gates"]["unknown_t100_fallbacks"] is False
    assert gate["passed"] == gate["total"] - 1
