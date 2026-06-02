from __future__ import annotations

from src import stage43_source_family_coverage_guard as m


def test_coverage_policy_names_and_modes() -> None:
    support = {"global_families": ["biwi"], "domain_families": {"TrajNet": ["biwi"]}}
    base = {"gain_threshold": 0.5, "harm_threshold": 0.1, "failure_threshold": 0.5}
    policy = m._coverage_policy(base, support, mode="global")
    assert policy["name"] == "global_source_family_coverage_guard"
    assert policy["source_family_support_mode"] == "global"
    assert policy["supported_global_families"] == ["biwi"]


def test_objective_penalizes_easy_degradation() -> None:
    good = {
        "full_waypoint_ade_improvement_vs_floor": 0.1,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.1,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.0,
        "switch_rate": 0.1,
        "easy_degradation_vs_floor": 0.0,
    }
    bad = dict(good)
    bad["easy_degradation_vs_floor"] = 0.1
    assert m._objective(good) > m._objective(bad)


def test_gate_passes_safe_lift_coverage_guard() -> None:
    payload = {
        "stage43_cc_precondition": {"verdict": "stage43_cc_shadow_easy_guard_shadow_safe_test_mismatch"},
        "result_source": "fresh_validation_source_family_coverage_guard",
        "protocol": {"train_only_heads_refit": True},
        "coverage_policy": {
            "selection_uses_test_metrics": False,
            "selected_shadow_policy": {
                "policy": {"source_family_support_mode": "global"},
                "shadow_holdout_metrics": {"easy_degradation_vs_floor": 0.0},
            },
        },
        "test_source_support_summary": {"global_unsupported_family_rows": {"pets": 2}},
        "test_once": {
            "metrics": {
                "easy_degradation_vs_floor": 0.0,
                "full_waypoint_ade_improvement_vs_floor": 0.01,
                "t50_full_waypoint_ade_improvement_vs_floor": 0.0,
                "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.0,
            }
        },
        "no_leakage": {
            "future_labels_as_inputs": False,
            "future_labels_train_eval_only": True,
            "test_threshold_tuning": False,
            "guard_uses_test_endpoints": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "long_objective_complete": False,
    }
    gate = m._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_cd_source_family_coverage_guard_pass"
