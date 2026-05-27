from __future__ import annotations

import numpy as np

from src import stage42_group_level_risk_repair as ex


def test_group_level_risk_lifts_any_row_risk_to_group() -> None:
    row_risk = np.asarray([False, True, False, False, True])
    group_key = np.asarray(["g1", "g1", "g2", "g3", "g3"], dtype=object)
    out = ex._group_level_risk(row_risk, group_key)
    assert out.tolist() == [True, True, False, True, True]


def test_group_level_key_modes() -> None:
    data = {
        "dataset": np.asarray(["UCY", "TrajNet", "UCY"], dtype=object),
        "horizon": np.asarray([50, 100, 50], dtype=np.int64),
    }
    ids = np.asarray([0, 1, 2], dtype=np.int64)
    risk = np.asarray([False, True, True])
    assert ex._group_level_key(data, ids, risk, "global").tolist() == ["global", "global", "global"]
    assert ex._group_level_key(data, ids, risk, "domain_horizon").tolist() == ["UCY|50", "TrajNet|100", "UCY|50"]
    assert ex._group_level_key(data, ids, risk, "domain_horizon_group_risk").tolist() == [
        "UCY|50|group_clear",
        "TrajNet|100|group_risk",
        "UCY|50|group_risk",
    ]


def test_deployment_decision_requires_di_lift() -> None:
    metric = {"all_improvement": 0.2, "hard_failure_improvement": 0.2, "easy_degradation": 0.0}
    selected = {
        "test_diagnostics": {"final_near_005": 0.01, "base_near_005": 0.02},
        "mixed_group_selection": {"mixed_group_count": 0},
    }
    ok = ex._deployment_decision(metric, {"delta_vs_stage42_di": {"all_improvement": 0.01, "hard_failure_improvement": 0.02}}, selected)
    not_ok = ex._deployment_decision(metric, {"delta_vs_stage42_di": {"all_improvement": -0.01, "hard_failure_improvement": 0.02}}, selected)
    assert ok["promote_group_level_risk_repair"] is True
    assert not_ok["promote_group_level_risk_repair"] is False


def test_gate_promotable_payload() -> None:
    payload = {
        "group_level_repair": {
            "candidate_count": 84,
            "risk_stats": {"group_risk_val_rate": 0.5, "row_risk_val_rate": 0.4},
            "mode_rows": [{"mode": "global"}, {"mode": "domain_horizon"}, {"mode": "domain_horizon_group_risk"}],
            "selected": {
                "mode": "domain_horizon_group_risk",
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
    gate = ex._gate(payload)
    assert gate["passed"] == gate["total"]
