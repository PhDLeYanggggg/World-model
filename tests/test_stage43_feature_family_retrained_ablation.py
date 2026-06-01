from __future__ import annotations

from src import stage43_feature_family_retrained_ablation as ah


def test_feature_family_masks_remove_expected_groups() -> None:
    names = [
        "current_x_over_scale",
        "history_dx_tail0",
        "history_neighbor_count",
        "prototype_distance_0",
        "goal_ambiguity",
        "baseline_endpoint_rel_0",
        "floor_endpoint_rel_x",
        "domain_UCY",
        "horizon_50",
    ]
    assert ah._feature_mask(names, "full_features").tolist() == [True] * len(names)
    assert ah._feature_mask(names, "no_history").tolist()[1] is False
    assert ah._feature_mask(names, "no_history").tolist()[2] is False
    assert ah._feature_mask(names, "no_neighbor_interaction").tolist()[2] is False
    assert ah._feature_mask(names, "no_neighbor_interaction").tolist()[1] is True
    assert ah._feature_mask(names, "no_goal").tolist()[3] is False
    assert ah._feature_mask(names, "no_goal").tolist()[4] is False
    assert ah._feature_mask(names, "no_baseline_floor").tolist()[5] is False
    assert ah._feature_mask(names, "no_baseline_floor").tolist()[6] is False
    assert ah._feature_mask(names, "no_domain").tolist()[7] is False


def _row(name: str, *, t50_delta: float = 0.0, all_delta: float = 0.0, hard_delta: float = 0.0) -> dict:
    full_metrics = {
        "rows": 10,
        "full_waypoint_ade_improvement_vs_floor": 0.20,
        "endpoint_fde_improvement_vs_floor": 0.20,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.20,
        "t50_endpoint_fde_improvement_vs_floor": 0.20,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.20,
        "easy_degradation_vs_floor": 0.0,
        "switch_rate": 0.2,
    }
    metrics = dict(full_metrics)
    if name != "full_features":
        metrics["full_waypoint_ade_improvement_vs_floor"] -= all_delta
        metrics["t50_full_waypoint_ade_improvement_vs_floor"] -= t50_delta
        metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] -= hard_delta
    return {
        "variant": name,
        "feature_count": 10,
        "checkpoint_committed": False,
        "latent_variance": 0.2,
        "test_metrics_with_floor": metrics,
        "delta_full_minus_variant": ah._metric_delta(full_metrics, metrics),
        "bootstrap_contribution_ci": {
            "n": 500,
            "metrics": {
                "t50_full_waypoint_ade_contribution": {
                    "rows": 10,
                    "mean": t50_delta,
                    "low": t50_delta / 2,
                    "high": t50_delta * 1.5,
                }
            },
        },
    }


def _payload(*, enough_modules: bool = True) -> dict:
    rows = [
        _row("full_features"),
        _row("no_history", t50_delta=0.05 if enough_modules else 0.0, hard_delta=0.02),
        _row("no_goal", t50_delta=0.03 if enough_modules else 0.0),
        _row("no_neighbor_interaction", all_delta=0.01),
        _row("no_baseline_floor", t50_delta=0.02 if enough_modules else 0.0),
        _row("no_domain"),
    ]
    return {
        "source": ah.SOURCE,
        "result_source": "fresh_retrained_feature_family_ablation",
        "variants": rows,
        "positive_t50_contribution_variants": ["no_history", "no_goal"] if enough_modules else [],
        "positive_hard_or_all_contribution_variants": ["no_history", "no_neighbor_interaction"] if enough_modules else [],
        "ablation_type": {
            "not_inference_masking": True,
            "not_full_all_module_factorial": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }


def test_gate_requires_multiple_positive_retrained_feature_families() -> None:
    passing = ah._gate(_payload(enough_modules=True))
    assert passing["passed"] == passing["total"]
    assert passing["feature_family_retrained_ablation_supports_modules"] is True

    failing = ah._gate(_payload(enough_modules=False))
    assert failing["gates"]["at_least_two_feature_families_show_contribution"] is False
    assert failing["feature_family_retrained_ablation_supports_modules"] is False
