from __future__ import annotations

from src import stage42_group_consistency_policy_replay as dk


def test_dict_close_reports_match_and_mismatch() -> None:
    rows = dk._dict_close({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 3.0}, ["a", "b"])
    assert rows["a"]["match"] is True
    assert rows["b"]["match"] is False


def test_policy_matches_di_metric_safety_and_leakage() -> None:
    selected = {
        "candidate": {"mode": "repel_unsafe", "min_sep": 0.08, "margin": 0.0, "strength": 0.5},
        "val_score": 1.23,
        "val_metric": {
            "all_improvement": 0.1,
            "t50_improvement": 0.2,
            "t100_raw_frame_diagnostic_improvement": 0.3,
            "hard_failure_improvement": 0.4,
            "easy_degradation": 0.0,
            "switch_rate": 0.5,
        },
    }
    metric = {
        "all_improvement": 0.11,
        "t50_improvement": 0.22,
        "t100_raw_frame_diagnostic_improvement": 0.33,
        "hard_failure_improvement": 0.44,
        "easy_degradation": -0.01,
        "switch_rate": 0.55,
    }
    safety = {
        "base_near_005": 0.02,
        "final_near_005": 0.01,
        "floor_near_005": 0.03,
        "base_p05_min_distance": 0.07,
        "final_p05_min_distance": 0.08,
        "floor_p05_min_distance": 0.06,
    }
    no_leak = {"future_endpoint_input": False}
    claim = {"stage5c_executed": False}
    di = {
        "repair": {
            "selected": selected,
            "test": {"metric_vs_floor": metric, "diagnostics": safety, "bootstrap": {"all": {"low": 0.1}}},
        },
        "no_leakage": no_leak,
        "claim_boundary": claim,
    }
    policy = {
        "repair_rule": {"type": "repel_unsafe", "min_sep": 0.08, "margin": 0.0, "strength": 0.5},
        "validation_selection": {"val_score": 1.23, "val_metric": selected["val_metric"]},
        "test_summary_vs_train_horizon_causal_floor": metric,
        "test_group_safety": safety,
        "bootstrap": {"all": {"low": 0.1}},
        "no_leakage": no_leak,
        "claim_boundary": claim,
    }
    replay = dk._policy_matches_di(policy, di)
    assert all(replay["repair_rule_matches_selected_candidate"].values())
    assert replay["validation_selection_replays_di"]["val_score"] is True
    assert all(row["match"] for row in replay["validation_selection_replays_di"]["val_metric"].values())
    assert all(row["match"] for row in replay["metric_matches"].values())
    assert all(row["match"] for row in replay["safety_matches"].values())
    assert replay["bootstrap_matches"] is True
    assert replay["no_leakage_matches"] is True
    assert replay["claim_boundary_matches"] is True


def _passing_payload() -> dict:
    policy = {
        "test_summary_vs_train_horizon_causal_floor": {
            "all_improvement": 0.1,
            "t50_improvement": 0.1,
            "t100_raw_frame_diagnostic_improvement": 0.05,
            "hard_failure_improvement": 0.2,
            "easy_degradation": 0.0,
        },
        "test_group_safety": {"base_near_005": 0.02, "final_near_005": 0.01, "floor_near_005": 0.03},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    return {
        "policy_artifact": {"exists": True},
        "policy_artifact_payload": policy,
        "inputs": {
            "stage42_di": {"stage42_di_gate": {"passed": 17, "total": 17}},
            "stage42_dj": {"stage42_dj_gate": {"passed": 22, "total": 22}},
        },
        "replay_checks": {
            "policy_hash_recomputed_matches_dj": True,
            "policy_json_matches_dj_embedded_policy": True,
            "di_replay": {
                "repair_rule_matches_selected_candidate": {"mode": True, "min_sep": True, "margin": True, "strength": True},
                "validation_selection_replays_di": {
                    "val_score": True,
                    "val_metric": {key: {"match": True} for key in dk.METRIC_KEYS},
                },
                "metric_matches": {key: {"match": True} for key in dk.METRIC_KEYS},
                "safety_matches": {key: {"match": True} for key in dk.SAFETY_KEYS},
                "bootstrap_matches": True,
                "no_leakage_matches": True,
                "claim_boundary_matches": True,
            },
        },
    }


def test_gate_passes_for_exact_group_consistency_replay() -> None:
    gate = dk._gate(_passing_payload())
    assert gate["verdict"] == "stage42_dk_group_consistency_policy_replay_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_policy_hash_does_not_replay() -> None:
    payload = _passing_payload()
    payload["replay_checks"]["policy_hash_recomputed_matches_dj"] = False
    gate = dk._gate(payload)
    assert gate["passed"] == gate["total"] - 1
    assert gate["gates"]["policy_hash_recomputed_matches_dj"] is False
