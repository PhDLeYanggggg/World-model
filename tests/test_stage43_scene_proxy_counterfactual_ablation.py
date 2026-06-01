from __future__ import annotations

import numpy as np

from src import stage43_scene_proxy_counterfactual_ablation as af
from src import stage43_scene_proxy_slice_safe_policy as ae


def test_counterfactual_routes_replace_only_scene_proxy_branch() -> None:
    actual = np.asarray(
        [
            ae.ROUTE_FLOOR,
            ae.ROUTE_STAGE43_M,
            ae.ROUTE_STAGE43_AB,
            ae.ROUTE_STAGE43_AB,
        ],
        dtype=np.int8,
    )
    routes = af._counterfactual_routes(actual)
    assert routes["actual_slice_safe"].tolist() == actual.tolist()
    assert routes["no_scene_proxy_to_stage43_m"].tolist() == [
        ae.ROUTE_FLOOR,
        ae.ROUTE_STAGE43_M,
        ae.ROUTE_STAGE43_M,
        ae.ROUTE_STAGE43_M,
    ]
    assert routes["no_scene_proxy_to_floor"].tolist() == [
        ae.ROUTE_FLOOR,
        ae.ROUTE_STAGE43_M,
        ae.ROUTE_FLOOR,
        ae.ROUTE_FLOOR,
    ]


def _payload(*, t50_lift: float = 0.05, easy: float = 0.0) -> dict:
    actual = {
        "rows": 10,
        "full_waypoint_ade_improvement_vs_floor": 0.20,
        "endpoint_fde_improvement_vs_floor": 0.25,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.30,
        "t50_endpoint_fde_improvement_vs_floor": 0.40,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.22,
        "easy_degradation_vs_floor": easy,
        "switch_rate": 0.5,
        "h100_floor_rate": 1.0,
    }
    no_scene = {
        "rows": 10,
        "full_waypoint_ade_improvement_vs_floor": 0.18,
        "endpoint_fde_improvement_vs_floor": 0.22,
        "t50_full_waypoint_ade_improvement_vs_floor": actual["t50_full_waypoint_ade_improvement_vs_floor"] - t50_lift,
        "t50_endpoint_fde_improvement_vs_floor": actual["t50_endpoint_fde_improvement_vs_floor"] - t50_lift,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.20,
        "easy_degradation_vs_floor": 0.0,
        "switch_rate": 0.5,
        "h100_floor_rate": 1.0,
    }
    return {
        "source": af.SOURCE,
        "result_source": "fresh_replay_same_route_counterfactual_model_family_ablation",
        "stage43_ae_verdict": "stage43_ae_slice_safe_scene_proxy_candidate",
        "stage43_ae_deploy": True,
        "route_counts": {"floor": 2, "stage43_m": 2, "stage43_ab": 6},
        "actual_slice_safe": {"metrics": actual},
        "counterfactuals": {"no_scene_proxy_to_stage43_m": {"metrics": no_scene}},
        "scene_proxy_contribution_vs_stage43_m_counterfactual": af._delta_metrics(actual, no_scene),
        "ablation_type": {"not_full_retrained_factorial_ablation": True},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "not_uniform_all_metric_improvement": True,
        },
    }


def test_gate_accepts_positive_scene_proxy_counterfactual_lift() -> None:
    gate = af._gate(_payload(t50_lift=0.05, easy=0.0))
    assert gate["passed"] == gate["total"]
    assert gate["scene_proxy_counterfactual_contribution_supported"] is True


def test_gate_rejects_missing_t50_lift_or_easy_harm() -> None:
    no_lift = af._gate(_payload(t50_lift=0.0, easy=0.0))
    assert no_lift["gates"]["scene_proxy_t50_lift_positive"] is False
    assert no_lift["scene_proxy_counterfactual_contribution_supported"] is False

    easy_harm = af._gate(_payload(t50_lift=0.05, easy=0.03))
    assert easy_harm["gates"]["actual_easy_preserved"] is False
    assert easy_harm["scene_proxy_counterfactual_contribution_supported"] is False
