from __future__ import annotations

import numpy as np

from src import stage42_adaptive_group_repair as ew


def test_adaptive_key_modes() -> None:
    data = {
        "dataset": np.asarray(["UCY", "TrajNet", "UCY"], dtype=object),
        "horizon": np.asarray([50, 100, 50], dtype=np.int64),
    }
    ids = np.asarray([0, 1, 2], dtype=np.int64)
    risk = np.asarray([False, True, True])
    assert ew._adaptive_key(data, ids, risk, "global").tolist() == ["global", "global", "global"]
    assert ew._adaptive_key(data, ids, risk, "domain_horizon").tolist() == ["UCY|50", "TrajNet|100", "UCY|50"]
    assert ew._adaptive_key(data, ids, risk, "domain_horizon_risk").tolist() == [
        "UCY|50|clear",
        "TrajNet|100|risk",
        "UCY|50|risk",
    ]


def test_candidate_score_penalizes_easy_harm_and_near_worsening() -> None:
    metric = {
        "all_improvement": 0.10,
        "hard_failure_improvement": 0.10,
        "t50_improvement": 0.10,
        "t100_raw_frame_diagnostic_improvement": 0.05,
        "easy_degradation": 0.00,
        "switch_rate": 0.20,
    }
    diag = {"base_near_005": 0.03, "final_near_005": 0.02}
    safe = ew._candidate_score(metric, diag)
    harmed = ew._candidate_score({**metric, "easy_degradation": 0.10}, diag)
    near_bad = ew._candidate_score(metric, {"base_near_005": 0.03, "final_near_005": 0.10})
    assert safe > harmed
    assert safe > near_bad


def test_mixed_group_selection_rate_detects_mixed_repairs() -> None:
    chosen = np.asarray(["a", "a", "b", "a", "b"], dtype=object)
    group = np.asarray(["g1", "g1", "g2", "g2", "g3"], dtype=object)
    out = ew._mixed_group_selection_rate(chosen, group)
    assert out["mixed_group_count"] == 1
    assert out["group_count"] == 3
    assert out["mixed_row_count"] == 2


def test_deployment_decision_requires_beating_di_and_group_consistency() -> None:
    metric = {"all_improvement": 0.20, "hard_failure_improvement": 0.20, "easy_degradation": 0.0}
    comparison = {"delta_vs_stage42_di": {"all_improvement": 0.01, "hard_failure_improvement": 0.02}}
    selected = {
        "mode": "domain_horizon",
        "test_diagnostics": {"final_near_005": 0.01, "base_near_005": 0.02},
        "mixed_group_selection": {"mixed_group_count": 0},
    }
    assert ew._deployment_decision(metric, comparison, selected)["promote_adaptive_group_repair"] is True
    selected_bad = {**selected, "mode": "domain_horizon_risk", "mixed_group_selection": {"mixed_group_count": 2}}
    assert ew._deployment_decision(metric, comparison, selected_bad)["promote_adaptive_group_repair"] is False


def test_gate_promotable_payload() -> None:
    payload = {
        "adaptive_repair": {
            "candidate_count": 84,
            "mode_rows": [{"mode": "global"}, {"mode": "domain_horizon"}, {"mode": "domain_horizon_risk"}],
            "selected": {
                "mode": "domain_horizon",
                "test_metric_vs_floor": {
                    "all_improvement": 0.25,
                    "t50_improvement": 0.2,
                    "hard_failure_improvement": 0.24,
                    "easy_degradation": 0.0,
                },
                "test_diagnostics": {"final_near_005": 0.01, "base_near_005": 0.02},
            },
        },
        "comparison_to_prior": {"delta_vs_stage42_di": {"all_improvement": 0.01, "hard_failure_improvement": 0.01}},
        "deployment_decision": {"group_consistent_selection": True},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_rule_selection": True,
            "validation_only_mode_selection": True,
            "source_overlap_pass": True,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = ew._gate(payload)
    assert gate["passed"] == gate["total"]
