from src import stage42_eth_seq_t50_support_dry_run as bt


def test_feature_threshold_selector_falls_back_when_gate_false() -> None:
    selector = bt._feature_threshold_selector("constant_position", "speed_causal", "high", 2.0)
    assert selector({"speed_causal": 1.0}) == bt.FALLBACK
    assert selector({"speed_causal": 3.0}) == "constant_position"


def test_gate_passes_honest_eth_seq_blocker() -> None:
    payload = {
        "source": "fresh_eth_seq_t50_support_dry_run_terms_unverified",
        "bm_verdict": "stage42_bm_eth_person_terms_audit_pass_claim_blocked",
        "bs_verdict": "stage42_bs_ucy_zara_t50_family_policy_pass_positive",
        "summary": {
            "candidate_sources": 5,
            "source_cv_folds": 5,
            "h50_windows_total": 100,
            "eth_seq_holdout_rows": 10,
            "eth_seq_t50_support_repaired": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "eth_person_terms_confirmed": False,
            "official_converted_dataset_claim_allowed": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = bt._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["eth_seq_t50_support_repaired"] is False
    assert gate["verdict"] == "stage42_bt_eth_seq_t50_support_dry_run_pass_blocker_confirmed"
