from __future__ import annotations

from src import stage43_bounded_residual_policy_freeze as an


def test_stable_hash_is_order_independent() -> None:
    assert an._stable_hash({"a": 1, "b": {"c": 2}}) == an._stable_hash({"b": {"c": 2}, "a": 1})


def _payload(*, confirmed: bool = True, easy: float = 0.0) -> dict:
    return {
        "source": an.SOURCE,
        "policy": {
            "policy_hash": "a" * 64,
            "evidence": {"stage43_m_replay_max_abs_diff": 0.0},
            "deployment_rule": {"global_floor_removal": False},
        },
        "policy_artifact": str(an.POLICY_JSON),
        "evidence_sources": {
            "stage43_al": {"deploy_bounded_residual": True},
            "stage43_am": {"statistically_confirmed": confirmed},
            "stage43_m": {"checkpoint_tracked_by_git": False},
        },
        "frozen_metrics": {
            "all_delta_ci": {"low": 0.01},
            "t50_delta_ci": {"low": 0.01},
            "hard_failure_delta_ci": {"low": 0.01},
            "easy_degradation_ci": {"high": easy},
            "easy": easy,
            "t100": 0.0,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "thresholds_selected_on_test": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }


def test_gate_passes_for_confirmed_safe_policy_after_artifact_exists(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "policy.json"
    artifact.write_text("{}", encoding="utf-8")
    payload = _payload(confirmed=True, easy=0.0)
    payload["policy_artifact"] = str(artifact)
    gate = an._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["policy_frozen"] is True


def test_gate_fails_without_statistical_confirmation(tmp_path) -> None:
    artifact = tmp_path / "policy.json"
    artifact.write_text("{}", encoding="utf-8")
    payload = _payload(confirmed=False, easy=0.0)
    payload["policy_artifact"] = str(artifact)
    gate = an._gate(payload)
    assert gate["gates"]["stage43_am_statistically_confirmed"] is False
    assert gate["policy_frozen"] is False


def test_gate_fails_if_easy_metric_is_unsafe(tmp_path) -> None:
    artifact = tmp_path / "policy.json"
    artifact.write_text("{}", encoding="utf-8")
    payload = _payload(confirmed=True, easy=0.05)
    payload["policy_artifact"] = str(artifact)
    gate = an._gate(payload)
    assert gate["gates"]["bootstrap_ci_supports_policy"] is False
    assert gate["gates"]["frozen_metrics_safe"] is False
