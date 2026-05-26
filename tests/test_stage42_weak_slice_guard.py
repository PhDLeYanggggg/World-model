from src import stage42_weak_slice_guard as s42af


def test_stage42af_validation_guard_uses_val_margin_and_excludes_ucy() -> None:
    row = {
        "choices": {
            "ETH_UCY|25": {"val_score": 0.01},
            "TrajNet|50": {"val_score": 0.08},
            "UCY|25": {"val_score": 0.0},
        }
    }
    assert s42af.validation_margin_guard_keys(row, threshold=0.02) == ["ETH_UCY|25"]


def test_stage42af_validation_guard_respects_threshold_boundary() -> None:
    row = {
        "choices": {
            "ETH_UCY|25": {"val_score": 0.02},
            "TrajNet|25": {"val_score": 0.019},
        }
    }
    assert s42af.validation_margin_guard_keys(row, threshold=0.02) == ["TrajNet|25"]


def test_stage42af_gate_passes_with_eth_t50_limitation() -> None:
    payload = {
        "stage42x_gate": {"verdict": "stage42_x_unified_row_level_full_waypoint_cache_pass"},
        "guarded_rows": [{"pair_idx": 0}, {"pair_idx": 1}, {"pair_idx": 2}],
        "guard_rule": {"uses_test_metrics_for_threshold": False},
        "summary": {
            "ade_all": {"mean": 0.08, "ci_low": 0.05, "ci_high": 0.1},
            "ade_t50": {"mean": 0.06, "ci_low": 0.02, "ci_high": 0.08},
            "ade_hard_failure": {"mean": 0.09, "ci_low": 0.03, "ci_high": 0.12},
            "ade_easy_degradation": {"mean": 0.001, "ci_low": 0.0, "ci_high": 0.01},
        },
        "repair_effect": {
            "horizon25_ade_all_after": 0.0,
            "horizon25_delta": 0.1,
            "eth_ucy_t50_limitation_remaining": True,
        },
        "no_leakage": {"future_endpoint_input": False, "test_policy_tuning": False},
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42af._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_af_weak_slice_guard_repair_pass_with_eth_t50_limitation"
