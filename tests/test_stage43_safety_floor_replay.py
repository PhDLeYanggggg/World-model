from __future__ import annotations

from src.stage43_safety_floor_replay import main


_PAYLOAD = None


def _payload():
    global _PAYLOAD
    if _PAYLOAD is None:
        _PAYLOAD = main()
    return _PAYLOAD


def test_stage43_safety_floor_gate_passes():
    payload = _payload()
    gate = payload["stage43_a_gate"]
    assert gate["verdict"] == "stage43_a_safety_floor_replay_pass"
    assert gate["passed"] == gate["total"]
    assert gate["latent_state_training_precondition"] is True


def test_stage43_replays_current_stage42_floor_exactly():
    payload = _payload()
    replay = payload["stage42_floors"]["stage42_source_full_waypoint_current_floor"]["fresh_replay"]
    metrics = replay["metrics"]
    assert replay["status"] == "fresh_run"
    assert replay["exact_replay_pass"] is True
    assert replay["max_replay_diff"] <= 1e-7
    assert metrics["rows"] == 47458
    assert metrics["fallback_exact_floor_rate"] >= 0.999
    assert metrics["ade_all"] > 0.0
    assert metrics["ade_t50"] > 0.0
    assert metrics["ade_hard_failure"] > 0.0
    assert metrics["easy_degradation"] <= 0.02


def test_stage43_freezes_historical_floors_and_boundaries():
    payload = _payload()
    floors = payload["historical_floors"]
    assert floors["stage26_sdd_selector"]["report_file"]["exists"]
    assert floors["stage37_external_t50_selector"]["policy_hash"]
    assert floors["m3w_neural_v1_protected_composite"]["package_hash"]
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
    assert payload["claim_boundary"]["metric_or_seconds_claim"] is False
    assert payload["no_leakage"]["future_endpoint_input"] is False
    assert payload["no_leakage"]["future_waypoint_input"] is False
    assert payload["no_leakage"]["central_velocity_official_input"] is False
    assert payload["no_leakage"]["test_endpoint_goal_construction"] is False
    assert payload["no_leakage"]["test_metric_threshold_tuning"] is False
    assert payload["no_leakage"]["stage42_cache_waypoints_are_labels_only"] is True
