from __future__ import annotations

from src import stage43_scene_proxy_augmented_latent_dynamics as ab


def _payload(*, lift: bool = True, easy: float = 0.0) -> dict:
    current = {
        "rows": 10,
        "full_waypoint_ade_improvement_vs_floor": 0.31 if lift else 0.29,
        "endpoint_fde_improvement_vs_floor": 0.40,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.18 if lift else 0.15,
        "t50_endpoint_fde_improvement_vs_floor": 0.29,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": -0.1,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.30,
        "easy_degradation_vs_floor": easy,
        "switch_rate": 0.5,
    }
    baseline = {
        "full_waypoint_ade_improvement_vs_floor": 0.30,
        "endpoint_fde_improvement_vs_floor": 0.39,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.16,
        "t50_endpoint_fde_improvement_vs_floor": 0.28,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": -0.12,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.28,
        "easy_degradation_vs_floor": 0.0,
        "switch_rate": 0.6,
    }
    delta = ab._delta_metrics(current, baseline)
    return {
        "source": ab.SOURCE,
        "result_source": "fresh_run_scene_proxy_augmented_torch_training",
        "checkpoint": "dummy.pt",
        "checkpoint_committed": False,
        "stage43_m_baseline_verdict": "stage43_m_protected_full_waypoint_latent_candidate_pass",
        "stage43_aa_precondition": {"verdict": "stage43_aa_scene_raster_proxy_tokens_pass"},
        "scene_proxy_feature_count": 14,
        "base_feature_count": 162,
        "feature_count": 176,
        "scene_proxy_feature_hashes": {"train": "a", "val": "b", "test": "c"},
        "test_metrics_with_floor": current,
        "delta_vs_stage43_m": delta,
        "scene_proxy_lift_over_stage43_m": lift,
        "deploy_scene_proxy_augmented_neural": lift and easy <= 0.02,
        "no_leakage": {
            "future_waypoint_input": False,
            "future_endpoint_input": False,
            "test_endpoint_goal_construction": False,
            "scene_proxy_train_only": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_delta_metrics_compare_against_stage43_m() -> None:
    delta = ab._delta_metrics(
        {"full_waypoint_ade_improvement_vs_floor": 0.31, "easy_degradation_vs_floor": 0.01},
        {"full_waypoint_ade_improvement_vs_floor": 0.30, "easy_degradation_vs_floor": 0.00},
    )
    assert abs(delta["full_waypoint_ade_improvement_vs_floor"] - 0.01) < 1e-8
    assert abs(delta["easy_degradation_vs_floor"] - 0.01) < 1e-8


def test_gate_promotes_only_when_scene_proxy_lifts_and_easy_safe(monkeypatch) -> None:
    monkeypatch.setattr(ab.Path, "exists", lambda self: True)
    gate = ab._gate(_payload(lift=True, easy=0.0))
    assert gate["passed"] == gate["total"]
    assert gate["deploy_scene_proxy_augmented_neural"] is True


def test_gate_keeps_stage43_m_when_no_scene_proxy_lift(monkeypatch) -> None:
    monkeypatch.setattr(ab.Path, "exists", lambda self: True)
    gate = ab._gate(_payload(lift=False, easy=0.0))
    assert gate["passed"] == gate["total"]
    assert gate["deploy_scene_proxy_augmented_neural"] is False
    assert gate["verdict"] == "stage43_ab_scene_proxy_augmented_latent_diagnostic_keep_stage43_m"


def test_gate_blocks_easy_degradation(monkeypatch) -> None:
    monkeypatch.setattr(ab.Path, "exists", lambda self: True)
    gate = ab._gate(_payload(lift=True, easy=0.03))
    assert gate["gates"]["easy_preserved"] is False
    assert gate["deploy_scene_proxy_augmented_neural"] is False
