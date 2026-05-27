from __future__ import annotations

from src import stage42_fh_horizon_weak_slice_repair as fk


def _metric(all_i: float = 0.2, hard: float = 0.1, easy: float = 0.0, rows: int = 50) -> dict:
    return {
        "rows": rows,
        "all_improvement": all_i,
        "t50_improvement": all_i,
        "t100_raw_frame_diagnostic_improvement": all_i,
        "hard_failure_improvement": hard,
        "easy_degradation": easy,
        "switch_rate": 0.5,
    }


def test_safe_validation_requires_rows_gain_and_easy_safety() -> None:
    assert fk._safe_for_validation(_metric(), 50)
    assert not fk._safe_for_validation(_metric(rows=5), 5)
    assert not fk._safe_for_validation(_metric(all_i=-0.1), 50)
    assert not fk._safe_for_validation(_metric(hard=-0.1), 50)
    assert not fk._safe_for_validation(_metric(easy=0.03), 50)


def test_choose_candidate_prefers_safe_best_score() -> None:
    candidates = {
        "fh": {"metric": _metric(all_i=0.1, hard=0.1, easy=0.0)},
        "di": {"metric": _metric(all_i=0.2, hard=0.2, easy=0.0)},
        "fc": {"metric": _metric(all_i=0.5, hard=0.5, easy=0.5)},
        "floor": {"metric": _metric(all_i=0.0, hard=0.0, easy=0.0)},
    }

    choice = fk._choose_candidate(candidates)

    assert choice["candidate"] == "di"
    assert choice["validation_safe"] is True


def test_choose_candidate_falls_back_when_no_safe_positive_candidate() -> None:
    candidates = {
        "fh": {"metric": _metric(all_i=-0.1, hard=-0.1, easy=0.0)},
        "fc": {"metric": _metric(all_i=0.5, hard=0.5, easy=0.5)},
    }

    choice = fk._choose_candidate(candidates)

    assert choice["candidate"] == "floor"
    assert choice["reason"] == "no_validation_safe_positive_candidate"


def test_gate_blocks_uniform_horizon_overclaim_when_weak_horizons_remain() -> None:
    payload = {
        "source": fk.SOURCE,
        "input_reports": {
            "stage42_fi_verdict": "stage42_fi_fh_policy_freeze_replay_pass",
            "stage42_fj_verdict": "stage42_fj_fh_source_robustness_pass",
        },
        "selection_rule": {"target_keys": ["UCY|100"], "uses_test_metrics_for_selection": False},
        "summary": {
            "weak_horizon_count_before": 3,
            "weak_horizon_count_after": 1,
        },
        "metric_vs_floor": _metric(),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "uniform_horizon_claim": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fk._gate(payload)

    assert gate["verdict"] == "stage42_fk_fh_horizon_weak_slice_repair_partial"
    assert gate["gates"]["uniform_horizon_claim_only_if_no_weak_horizons"] is False
