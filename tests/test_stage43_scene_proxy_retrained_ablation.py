from __future__ import annotations

from src import stage43_scene_proxy_retrained_ablation as ag


def test_scene_proxy_groups_are_disjoint_enough_for_ablation() -> None:
    assert ag.SCENE_GROUPS["no_scene"] == []
    assert 8 in ag.SCENE_GROUPS["goal_only"]
    assert 6 in ag.SCENE_GROUPS["geometry_route"]
    assert len(ag.SCENE_GROUPS["full_scene"]) == 14
    assert set(ag.SCENE_GROUPS["goal_only"]) != set(ag.SCENE_GROUPS["geometry_route"])


def _variant(name: str, *, t50: float, hard: float, all_: float, easy: float = 0.0, features: int = 0) -> dict:
    metrics = {
        "rows": 10,
        "full_waypoint_ade_improvement_vs_floor": all_,
        "endpoint_fde_improvement_vs_floor": all_,
        "t50_full_waypoint_ade_improvement_vs_floor": t50,
        "t50_endpoint_fde_improvement_vs_floor": t50,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": hard,
        "easy_degradation_vs_floor": easy,
        "switch_rate": 0.3,
    }
    baseline = {
        "full_waypoint_ade_improvement_vs_floor": 0.1,
        "endpoint_fde_improvement_vs_floor": 0.1,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.1,
        "t50_endpoint_fde_improvement_vs_floor": 0.1,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.1,
        "easy_degradation_vs_floor": 0.0,
        "switch_rate": 0.3,
    }
    return {
        "variant": name,
        "scene_feature_count": features,
        "test_metrics_with_floor": metrics,
        "delta_vs_retrained_no_scene": ag._delta_metrics(metrics, baseline),
        "latent_variance": 0.2,
        "checkpoint_committed": False,
    }


def _payload(*, easy_harm: bool = False, t50_lift: bool = True) -> dict:
    no_scene = _variant("no_scene", t50=0.1, hard=0.1, all_=0.1, features=0)
    geom = _variant("geometry_route", t50=0.16 if t50_lift else 0.1, hard=0.13, all_=0.12, easy=0.03 if easy_harm else 0.0, features=9)
    goal = _variant("goal_only", t50=0.12 if t50_lift else 0.1, hard=0.11, all_=0.11, features=9)
    safe_best = None if easy_harm or not t50_lift else "geometry_route"
    return {
        "source": ag.SOURCE,
        "result_source": "fresh_retrained_scene_proxy_subset_ablation",
        "variants": [no_scene, geom, goal],
        "best_variant_by_t50_delta": "geometry_route",
        "best_safe_variant_by_t50_delta": safe_best,
        "best_variant_by_hard_delta": "geometry_route",
        "best_variant_by_all_delta": "geometry_route",
        "ablation_type": {"not_full_stage43_factorial_all_modules": True},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_train_only": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }


def test_gate_passes_when_retrained_scene_subset_lifts_t50_safely() -> None:
    gate = ag._gate(_payload(easy_harm=False, t50_lift=True))
    assert gate["passed"] == gate["total"]
    assert gate["scene_proxy_retrained_ablation_supports_contribution"] is True


def test_gate_rejects_no_t50_lift_or_easy_harm() -> None:
    no_lift = ag._gate(_payload(easy_harm=False, t50_lift=False))
    assert no_lift["gates"]["scene_subset_t50_lift_found"] is False
    assert no_lift["scene_proxy_retrained_ablation_supports_contribution"] is False

    easy_harm = ag._gate(_payload(easy_harm=True, t50_lift=True))
    assert easy_harm["gates"]["safe_t50_scene_variant_available"] is False
    assert easy_harm["scene_proxy_retrained_ablation_supports_contribution"] is False
