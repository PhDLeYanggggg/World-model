from __future__ import annotations

import numpy as np

from src.stage42_dual_domain_group_consistency_statistics import _bootstrap_degradation, _gate, _positive_ci


def _ci(low: float = 0.1, high: float = 0.2) -> dict:
    return {"low": low, "mid": (low + high) / 2.0, "high": high, "n": 100, "bootstrap_n": 2000}


def _slice_ci() -> dict:
    return {
        "all": _ci(),
        "t50": _ci(),
        "t100_raw_frame_diagnostic": _ci(),
        "hard_failure": _ci(),
        "easy_degradation": _ci(-0.2, 0.0),
    }


def test_bootstrap_degradation_is_negative_improvement_interval() -> None:
    selected = np.ones(100, dtype=float)
    floor = np.ones(100, dtype=float) * 2.0
    mask = np.ones(100, dtype=bool)
    ci = _bootstrap_degradation(selected, floor, mask, seed=1, n=100)
    assert ci["high"] < 0.0
    assert ci["bootstrap_n"] == 100


def test_positive_ci_requires_positive_lows_and_easy_high_under_gate() -> None:
    assert _positive_ci(_slice_ci())
    bad = _slice_ci()
    bad["t50"] = _ci(-0.01, 0.1)
    assert not _positive_ci(bad)
    bad_easy = _slice_ci()
    bad_easy["easy_degradation"] = _ci(0.01, 0.03)
    assert not _positive_ci(bad_easy)


def test_gate_passes_dual_domain_statistical_evidence() -> None:
    payload = {
        "source": "fresh_stage42_ea_dual_domain_group_consistency_statistics",
        "bootstrap_ci": {
            "global": _slice_ci(),
            "by_domain": {"UCY": _slice_ci(), "TrajNet": _slice_ci()},
        },
        "near_collision_ci": {
            "global": {
                "delta_final_minus_base": {"high": -0.001},
            }
        },
        "summary": {
            "ci_positive_safe_domains": 2,
            "near005_delta_high": -0.001,
        },
        "no_leakage": {
            "test_sources_unchanged": True,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ea_dual_domain_group_consistency_statistics_pass"
