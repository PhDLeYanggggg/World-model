from __future__ import annotations

from src.stage42_ucy_supported_fe_composer import _gate, _positive_safe, _summary_section


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


def _payload() -> dict:
    return {
        "source": "fresh_stage42_ucy_supported_fe_composer",
        "summary": {
            "ucy_val_rows_after": 10,
            "test_rows_unchanged": True,
            "positive_safe_domains": ["TrajNet", "UCY"],
            "weak_domains": [],
            "deployment_decision": "promote_stage42_fh_ucy_supported_fe_composer",
            "selected_candidate": {"mode": "fc_to_safety"},
        },
        "repair": {
            "candidate_count": 42,
            "selected": {"candidate": {"mode": "fc_to_safety"}},
            "test": {
                "metric_vs_floor": _metric(),
                "by_domain": {"UCY": _metric(), "TrajNet": _metric()},
                "near_delta_vs_fc": -0.01,
                "near_delta_vs_di": 0.0,
                "bootstrap": {
                    "all": {"low": 0.1, "mid": 0.2, "high": 0.3, "n": 10, "bootstrap_n": 100},
                    "t50": {"low": 0.1, "mid": 0.2, "high": 0.3, "n": 10, "bootstrap_n": 100},
                    "hard_failure": {"low": 0.1, "mid": 0.2, "high": 0.3, "n": 10, "bootstrap_n": 100},
                    "easy_degradation": {"low": -0.1, "mid": 0.0, "high": 0.01, "n": 10, "bootstrap_n": 100},
                },
            },
        },
        "no_leakage": {
            "test_sources_unchanged": True,
            "source_overlap_pass": True,
            "test_threshold_tuning": False,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "validation_only_policy_selection": True,
            "internal_val_from_train_only": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_positive_safe_requires_all_t50_hard_and_easy() -> None:
    assert _positive_safe(_metric())
    assert not _positive_safe(_metric(all_i=0.0))
    assert not _positive_safe(_metric(t50=0.0))
    assert not _positive_safe(_metric(hard=0.0))
    assert not _positive_safe(_metric(easy=0.03))


def test_gate_passes_dual_domain_ucy_supported_fe_composer() -> None:
    gate = _gate(_payload())

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_fh_ucy_supported_fe_composer_pass"


def test_gate_fails_when_ucy_is_not_positive() -> None:
    payload = _payload()
    payload["repair"]["test"]["by_domain"]["UCY"] = _metric(all_i=0.0, t50=0.0, hard=0.0)
    payload["summary"]["positive_safe_domains"] = ["TrajNet"]
    payload["summary"]["weak_domains"] = ["UCY"]

    gate = _gate(payload)

    assert gate["passed"] < gate["total"]
    assert gate["gates"]["ucy_positive_safe"] is False
    assert gate["gates"]["at_least_two_positive_safe_domains"] is False


def test_summary_section_records_boundary_and_decision() -> None:
    payload = _payload()
    payload["stage42_fh_gate"] = _gate(payload)

    section = _summary_section(payload)

    assert "Stage42-FH UCY-Supported FE Composer" in section
    assert "promote_stage42_fh_ucy_supported_fe_composer" in section
    assert "no metric/seconds claim" in section
