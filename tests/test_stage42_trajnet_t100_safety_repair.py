from src import stage42_trajnet_t100_safety_repair as s42ai


def test_trajnet_t100_source_choice_prefers_easy_safe_positive_source() -> None:
    assert (
        s42ai.trajnet_t100_source_choice(
            j_val_all=0.09,
            j_val_easy=0.0,
            p_val_all=0.11,
            p_val_easy=0.004,
        )
        == "stage42j_static_expert"
    )
    assert (
        s42ai.trajnet_t100_source_choice(
            j_val_all=0.08,
            j_val_easy=0.0,
            p_val_all=0.13,
            p_val_easy=0.0,
        )
        == "stage42p_t50_gain_harm"
    )
    assert (
        s42ai.trajnet_t100_source_choice(
            j_val_all=-0.01,
            j_val_easy=0.0,
            p_val_all=0.10,
            p_val_easy=0.03,
        )
        == "floor"
    )


def test_stage42ai_gate_passes_t100_safety_repair() -> None:
    payload = {
        "stage42ah_gate": {"verdict": "stage42_ah_post_repair_claim_refresh_pass"},
        "source_repair_rule": {"uses_test_metrics_for_threshold": False},
        "summary": {
            "ade_all": {"ci_low": 0.01},
            "ade_t50": {"ci_low": 0.01},
            "ade_t100_raw_frame_diagnostic": {"ci_low": 0.01},
            "ade_hard_failure": {"ci_low": 0.01},
            "ade_easy_degradation": {"ci_high": 0.01},
        },
        "repair_effect": {
            "trajnet100_safety_repaired": True,
            "trajnet100_ade_ci_low_after": 0.01,
        },
        "claim_boundary": {
            "t100_seconds_claim": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42ai._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ai_trajnet_t100_safety_repair_pass"
