from __future__ import annotations

from src.stage42_ucy_supported_group_consistency import _gate, _positive_safe


def _metric(all_i: float = 0.2, t50: float = 0.1, hard: float = 0.2, easy: float = 0.0) -> dict:
    return {
        "rows": 10,
        "all_improvement": all_i,
        "t10_improvement": 0.1,
        "t25_improvement": 0.1,
        "t50_improvement": t50,
        "t100_raw_frame_diagnostic_improvement": 0.1,
        "hard_failure_improvement": hard,
        "easy_degradation": easy,
        "switch_rate": 0.5,
    }


def test_positive_safe_requires_all_t50_hard_and_easy() -> None:
    assert _positive_safe(_metric())
    assert not _positive_safe(_metric(all_i=0.0))
    assert not _positive_safe(_metric(t50=0.0))
    assert not _positive_safe(_metric(hard=0.0))
    assert not _positive_safe(_metric(easy=0.03))


def test_gate_passes_dual_domain_ucy_supported_group_consistency() -> None:
    payload = {
        "summary": {
            "ucy_val_rows_after": 10,
            "near005_base": 0.02,
            "near005_final": 0.01,
            "positive_safe_domains": 2,
        },
        "repair": {
            "candidate_count": 42,
            "test": {
                "metric_vs_floor": _metric(),
                "by_domain": {"UCY": _metric(), "TrajNet": _metric()},
            },
        },
        "no_leakage": {
            "test_sources_unchanged": True,
            "source_overlap_pass": True,
            "test_threshold_tuning": False,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "internal_val_from_train_only": True,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "ungated_full_waypoint_deployable": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_dz_ucy_supported_group_consistency_pass_dual_domain"
