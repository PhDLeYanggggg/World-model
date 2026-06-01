from __future__ import annotations

import numpy as np

from src import stage43_scene_proxy_slice_safe_policy as ae


class _TinySplit:
    def __init__(self) -> None:
        self.x = np.zeros((8, 3), dtype=np.float32)
        self.horizon = np.asarray([10, 25, 50, 100, 10, 25, 50, 100], dtype=np.int64)
        self.domain = np.asarray(["TrajNet", "TrajNet", "TrajNet", "TrajNet", "UCY", "UCY", "UCY", "UCY"])
        self.floor_ade = np.ones(8, dtype=np.float32)
        self.floor_fde = np.ones(8, dtype=np.float32)


def test_domain_safe_v1_routes_unsafe_trajnet_slices_to_floor_or_m() -> None:
    ds = _TinySplit()
    route = ae._route_codes(ds, "domain_safe_v1")
    assert route.tolist() == [
        ae.ROUTE_FLOOR,
        ae.ROUTE_STAGE43_M,
        ae.ROUTE_STAGE43_M,
        ae.ROUTE_FLOOR,
        ae.ROUTE_STAGE43_AB,
        ae.ROUTE_STAGE43_AB,
        ae.ROUTE_STAGE43_AB,
        ae.ROUTE_FLOOR,
    ]


def test_candidate_pack_rejects_scene_proxy_to_configured_fallback() -> None:
    ds = _TinySplit()
    route = np.full(8, ae.ROUTE_STAGE43_AB, dtype=np.int8)
    pack = {
        "m_ade": np.full(8, 0.7, dtype=np.float32),
        "m_fde": np.full(8, 0.7, dtype=np.float32),
        "ab_ade": np.full(8, 0.4, dtype=np.float32),
        "ab_fde": np.full(8, 0.4, dtype=np.float32),
    }
    ab_allowed = np.asarray([True, False, True, False, True, True, False, False])
    selected_ade, selected_fde, switched, final_route = ae._candidate_pack(
        ds,
        pack,
        route=route,
        ab_allowed=ab_allowed,
        ab_reject_fallback="stage43_m",
    )
    assert final_route.tolist() == [
        ae.ROUTE_STAGE43_AB,
        ae.ROUTE_STAGE43_M,
        ae.ROUTE_STAGE43_AB,
        ae.ROUTE_STAGE43_M,
        ae.ROUTE_STAGE43_AB,
        ae.ROUTE_STAGE43_AB,
        ae.ROUTE_STAGE43_M,
        ae.ROUTE_STAGE43_M,
    ]
    assert selected_ade.tolist() == [0.4000000059604645, 0.699999988079071, 0.4000000059604645, 0.699999988079071, 0.4000000059604645, 0.4000000059604645, 0.699999988079071, 0.699999988079071]
    assert selected_fde.tolist() == selected_ade.tolist()
    assert switched.tolist() == [True] * 8


def _valid_payload() -> dict:
    metrics = {
        "full_waypoint_ade_improvement_vs_floor": 0.25,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.2,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.22,
        "easy_degradation_vs_floor": 0.0,
        "h10_floor_rate": 1.0,
        "h100_floor_rate": 1.0,
    }
    return {
        "source": ae.SOURCE,
        "result_source": "fresh_validation_selected_slice_safe_three_route_policy",
        "stage43_ac_verdict": "stage43_ac_guarded_scene_proxy_latent_candidate",
        "stage43_ad_verdict": "stage43_ad_guarded_scene_proxy_caveated_audit_pass",
        "validation_selected_policy": {
            "policy": {
                "selected_on": "validation_only",
                "test_threshold_tuning": False,
                "uses_easy_label_at_inference": False,
                "stage43_ad_structural_caveat_guard": True,
            }
        },
        "route_counts": {"floor": 3, "stage43_m": 2, "stage43_ab": 4},
        "test_metrics_slice_safe": metrics,
        "test_diagnostics_slice_safe": {
            "max_domain_easy_degradation": 0.0,
            "max_horizon_easy_degradation": 0.0,
            "min_domain_all_improvement": 0.05,
        },
        "delta_vs_stage43_m": {
            "full_waypoint_ade_improvement_vs_floor": 0.01,
            "t50_full_waypoint_ade_improvement_vs_floor": 0.0,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.0,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "uses_easy_label_at_inference": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_gate_passes_only_when_three_route_slice_safe_policy_is_clean() -> None:
    gate = ae._gate(_valid_payload())
    assert gate["passed"] == gate["total"]
    assert gate["deploy_slice_safe_scene_proxy"] is True


def test_gate_rejects_easy_label_or_missing_floor_route() -> None:
    payload = _valid_payload()
    payload["no_leakage"]["uses_easy_label_at_inference"] = True
    gate = ae._gate(payload)
    assert gate["gates"]["no_future_or_test_leakage"] is False
    assert gate["deploy_slice_safe_scene_proxy"] is False

    payload = _valid_payload()
    payload["route_counts"]["floor"] = 0
    gate = ae._gate(payload)
    assert gate["gates"]["three_route_policy_used"] is False
    assert gate["deploy_slice_safe_scene_proxy"] is False
