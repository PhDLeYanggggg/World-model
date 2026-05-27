from __future__ import annotations

from src import stage42_group_consistency_t100_easy_guard_freeze as hs


def _hr_payload() -> dict:
    return {
        "stage42_hr_gate": {"passed": 23, "total": 23, "verdict": "stage42_hr_t100_easy_guard_pass"},
        "pre_guard": {"metric": {}, "t100_easy_degradation": 0.025},
        "guarded": {
            "threshold": 0.0,
            "guarded_slices": {"TrajNet|100": {"keep": False}},
            "kept_slices": {"UCY|100": {"keep": True}},
            "metric": {
                "rows": 10,
                "all_improvement": 0.2,
                "t50_improvement": 0.1,
                "t100_raw_frame_diagnostic_improvement": 0.03,
                "hard_failure_improvement": 0.12,
                "easy_degradation": 0.0,
                "switch_rate": 0.4,
                "harm_over_fallback": -0.2,
            },
            "t100_easy_degradation": -0.01,
            "by_domain": {
                "TrajNet": {"all_improvement": 0.1, "t50_improvement": 0.1},
                "UCY": {"all_improvement": 0.2, "t50_improvement": 0.1},
            },
            "by_horizon": {"100": {"all_improvement": 0.03}},
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
            "source_overlap_pass": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def _payload() -> dict:
    hr = _hr_payload()
    policy = hs._policy_payload(hr)
    replay = hs._replay_from_policy(policy, hr)
    return {
        "inputs": {"hr_artifact_exists": True, "stage42_hr_gate": hr["stage42_hr_gate"]},
        "frozen_policy": policy,
        "policy_artifact": {"sha256": "a" * 64},
        "policy_hash": "b" * 64,
        "replay": replay,
        "paper_file_status": [{"exists": True, "contains_stage42_hs": True}],
    }


def test_policy_payload_freezes_validation_only_decision_table() -> None:
    policy = hs._policy_payload(_hr_payload())

    assert policy["selection_scope"] == "validation_only_domain_horizon_t100"
    assert policy["decision_rule"]["uses_test_metrics_for_guard"] is False
    assert "TrajNet|100" in policy["decision_table"]["guarded_slices"]
    assert "UCY|100" in policy["decision_table"]["kept_slices"]


def test_replay_is_exact_for_compact_hr_artifact() -> None:
    hr = _hr_payload()
    policy = hs._policy_payload(hr)
    replay = hs._replay_from_policy(policy, hr)

    assert replay["decision_table_exact_replay"] is True
    assert replay["metric_summary_exact_replay"] is True
    assert replay["max_metric_abs_diff"] == 0.0


def test_gate_passes_for_safe_frozen_hr_policy() -> None:
    gate = hs._gate(_payload())

    assert gate["verdict"] == "stage42_hs_t100_easy_guard_freeze_pass"
    assert gate["passed"] == gate["total"]
    assert gate["gates"]["trajnet_t100_guarded"] is True
    assert gate["gates"]["ucy_t100_kept"] is True


def test_gate_fails_when_t100_easy_is_not_repaired() -> None:
    payload = _payload()
    payload["frozen_policy"]["test_summary_vs_train_horizon_causal_floor"]["t100_easy_degradation"] = 0.03

    gate = hs._gate(payload)

    assert gate["gates"]["t100_easy_repaired"] is False
    assert gate["passed"] == gate["total"] - 1
