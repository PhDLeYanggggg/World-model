from src import stage42_source_diversity_acquisition_package as cd


def test_official_targets_do_not_auto_download() -> None:
    assert len(cd.OFFICIAL_TARGETS) >= 4
    assert all(not row["auto_download_allowed"] for row in cd.OFFICIAL_TARGETS)
    assert any(row["target_blocker"] == "UCY_students_t50_source_support" for row in cd.OFFICIAL_TARGETS)


def test_target_status_never_counts_conversion() -> None:
    cc = {"summary": {"unused_candidate_t50_sources": 0}}
    bv = {"blocker_matrix": [{"blocker_id": "UCY_students_t50_source_support", "status": "blocked"}]}
    row = cd._target_status(cd.OFFICIAL_TARGETS[0], cc, bv)
    assert row["can_claim_converted_now"] is False
    assert row["can_claim_source_diversity_repair_now"] is False
    assert "no unused independent" in row["reason"]


def test_gate_requires_no_overclaim() -> None:
    payload = {
        "source": "unit",
        "input_reports": {
            "stage42_bv_verdict": "stage42_bv_source_acquisition_status_pass_blockers_actionable",
            "stage42_cc_verdict": "stage42_cc_independent_t50_source_inventory_pass",
        },
        "summary": {
            "official_targets": 5,
            "critical_targets": 2,
            "manual_or_terms_targets": 4,
            "broad_source_generalization_claim_allowed": False,
        },
        "claim_boundary": {
            "auto_download_executed": False,
            "inventory_counted_as_converted": False,
            "registry_only_counted_as_converted": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "next_commands_after_user_data": ["convert"],
        "user_action_required": [{"action": "provide"}],
    }
    gate = cd._gate(payload)
    assert gate["verdict"] == "stage42_cd_source_diversity_acquisition_package_pass"
    assert gate["passed"] == gate["total"]


def test_gate_blocks_metric_overclaim() -> None:
    payload = {
        "source": "unit",
        "input_reports": {
            "stage42_bv_verdict": "stage42_bv_source_acquisition_status_pass_blockers_actionable",
            "stage42_cc_verdict": "stage42_cc_independent_t50_source_inventory_pass",
        },
        "summary": {
            "official_targets": 5,
            "critical_targets": 2,
            "manual_or_terms_targets": 4,
            "broad_source_generalization_claim_allowed": False,
        },
        "claim_boundary": {
            "auto_download_executed": False,
            "inventory_counted_as_converted": False,
            "registry_only_counted_as_converted": False,
            "metric_or_seconds_claim": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "next_commands_after_user_data": ["convert"],
        "user_action_required": [{"action": "provide"}],
    }
    gate = cd._gate(payload)
    assert gate["verdict"] == "stage42_cd_source_diversity_acquisition_package_partial"
    assert not gate["gates"]["no_metric_seconds_overclaim"]
