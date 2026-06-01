from __future__ import annotations

from src import stage43_safety_floor_necessity_audit as aj


def _payload(*, unsafe_ungated: bool = True) -> dict:
    return {
        "source": aj.SOURCE,
        "evidence_sources": {
            "stage43_a_safety_floor_replay": {"verdict": "stage43_a_safety_floor_replay_pass"},
        },
        "protected_vs_ungated": {
            "protected_easy": 0.0,
            "ungated_easy": 0.55 if unsafe_ungated else 0.0,
            "easy_harm_reduction": 0.55 if unsafe_ungated else 0.0,
            "protected_t100": -0.18,
            "ungated_t100": -0.72,
            "t100_harm_reduction": 0.54,
        },
        "multiseed_floor_feature_evidence": {
            "no_baseline_floor_t50_delta_mean": 0.12,
            "no_baseline_floor_t50_positive_seed_count": 3,
            "seed_count": 3,
        },
        "scene_proxy_floor_guard_evidence": {
            "raw_best_easy": 0.09,
            "safe_best_easy": 0.0,
            "slice_safe_t50": 0.37,
            "slice_safe_easy": 0.0,
            "h100_floor_rate": 1.0,
        },
        "conclusion": {
            "global_floor_removable": False,
            "floor_is_core_safety_mechanism": True,
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


def test_gate_confirms_floor_necessity_from_harm_and_multiseed_evidence() -> None:
    gate = aj._gate(_payload(unsafe_ungated=True))
    assert gate["passed"] == gate["total"]
    assert gate["floor_necessity_confirmed"] is True


def test_gate_rejects_if_ungated_easy_harm_is_not_material() -> None:
    gate = aj._gate(_payload(unsafe_ungated=False))
    assert gate["gates"]["protected_vs_ungated_reported"] is False
    assert gate["gates"]["protected_reduces_easy_harm_materially"] is False
    assert gate["floor_necessity_confirmed"] is False
