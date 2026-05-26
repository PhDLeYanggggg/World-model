from src import stage42_source_diversity_conversion_preflight as ce


def test_next_action_blocks_terms_before_conversion() -> None:
    action = ce._next_action({"id": "x"}, independent=[{"source_name": "a"}], terms_blocked=True, schema_possible=True)
    assert "terms" in action
    assert "not legal permission" in action


def test_next_action_requires_parseable_rows() -> None:
    action = ce._next_action({"id": "x"}, independent=[], terms_blocked=False, schema_possible=False)
    assert "parseable trajectory" in action


def test_gate_prevents_conversion_overclaim() -> None:
    payload = {
        "source": "unit",
        "input_reports": {"stage42_cd_verdict": "stage42_cd_source_diversity_acquisition_package_pass"},
        "summary": {
            "targets_checked": 5,
            "targets_with_local_path": 4,
            "targets_with_schema_possible": 3,
            "source_diversity_repair_ready_now": False,
        },
        "claim_boundary": {
            "preflight_counted_as_conversion": False,
            "converted_dataset_claim": False,
            "evaluated_dataset_claim": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "target_summaries": [{"legal_terms_blocked": True}],
        "user_action_required": [{"action": "verify"}],
    }
    gate = ce._gate(payload)
    assert gate["verdict"] == "stage42_ce_source_diversity_conversion_preflight_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_preflight_is_counted_as_conversion() -> None:
    payload = {
        "source": "unit",
        "input_reports": {"stage42_cd_verdict": "stage42_cd_source_diversity_acquisition_package_pass"},
        "summary": {
            "targets_checked": 5,
            "targets_with_local_path": 4,
            "targets_with_schema_possible": 3,
            "source_diversity_repair_ready_now": False,
        },
        "claim_boundary": {
            "preflight_counted_as_conversion": True,
            "converted_dataset_claim": False,
            "evaluated_dataset_claim": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "target_summaries": [{"legal_terms_blocked": True}],
        "user_action_required": [{"action": "verify"}],
    }
    gate = ce._gate(payload)
    assert gate["verdict"] == "stage42_ce_source_diversity_conversion_preflight_partial"
    assert not gate["gates"]["no_conversion_claim"]
