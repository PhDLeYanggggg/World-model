from src import stage42_eth_t50_fde_source_repair as s42ag


def test_eth_t50_fde_source_choice_uses_validation_fde_support() -> None:
    assert (
        s42ag.eth_t50_fde_source_choice(j_val_ade_t50=0.01, j_val_fde_t50=0.06)
        == "stage42j_static_expert"
    )
    assert s42ag.eth_t50_fde_source_choice(j_val_ade_t50=0.01, j_val_fde_t50=0.049) == "floor"
    assert s42ag.eth_t50_fde_source_choice(j_val_ade_t50=-0.001, j_val_fde_t50=0.10) == "floor"


def test_stage42ag_gate_passes_positive_eth_t50_repair() -> None:
    payload = {
        "stage42af_gate": {"verdict": "stage42_af_weak_slice_guard_repair_pass_with_eth_t50_limitation"},
        "repaired_rows": [{"pair_idx": 0}, {"pair_idx": 1}, {"pair_idx": 2}],
        "source_repair_rule": {"uses_test_metrics_for_threshold": False},
        "summary": {
            "ade_all": {"mean": 0.09, "ci_low": 0.08, "ci_high": 0.10},
            "ade_t50": {"mean": 0.06, "ci_low": 0.05, "ci_high": 0.07},
            "ade_hard_failure": {"mean": 0.09, "ci_low": 0.08, "ci_high": 0.10},
            "ade_easy_degradation": {"mean": 0.001, "ci_low": 0.0, "ci_high": 0.01},
        },
        "repair_effect": {
            "eth_ucy_t50_ade_ci_low_after": 0.001,
            "eth_ucy_fde_t50_ci_low_after": 0.002,
        },
        "no_leakage": {"future_endpoint_input": False, "test_policy_tuning": False},
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42ag._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ag_eth_t50_fde_source_repair_pass"
