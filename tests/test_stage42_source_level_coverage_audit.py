from src import stage42_source_level_coverage_audit as s42al


def test_stage42al_domain_statuses_mark_partial_and_extra() -> None:
    split = {
        "proposed_source_level_split": {
            "train": {"domains": {"ETH_UCY": 10}},
            "val": {"domains": {"ETH_UCY": 2}},
            "test": {"domains": {"TrajNet": 100, "UCY": 20}},
        }
    }
    stress = {
        "by_domain": {
            "ETH_UCY": {"rows": 30},
            "TrajNet": {"rows": 50},
            "UCY": {"rows": 20},
        }
    }
    rows = {row["domain"]: row for row in s42al._domain_coverage(split, stress)}
    assert rows["ETH_UCY"]["status"] == "extra_available_not_in_proposed_source_test"
    assert rows["TrajNet"]["status"] == "partial_coverage"
    assert rows["UCY"]["status"] == "exact_row_count_match"


def test_stage42al_claim_table_rejects_full_source_eval() -> None:
    domain_rows = [
        {"domain": "UCY", "status": "exact_row_count_match"},
        {"domain": "TrajNet", "status": "partial_coverage"},
        {"domain": "ETH_UCY", "status": "extra_available_not_in_proposed_source_test"},
    ]
    horizon_rows = [{"horizon": 50, "status": "different_eval_pool"}]
    claims = {row["claim"]: row["status"] for row in s42al._claim_table(domain_rows, horizon_rows)}
    assert claims["Current locked-policy metrics can be described as full proposed source-level split evaluation."] == "rejected"
    assert claims["Current locked-policy metrics can be described as available row-level post-repair stress with explicit coverage gap."] == "supported"


def test_stage42al_gate_passes_when_gap_is_honestly_detected() -> None:
    payload = {
        "stage42ak_gate": {"verdict": "stage42_ak_post_repair_locked_policy_audit_pass"},
        "source_split": {
            "proposed_source_level_split": {"test": {"rows": 1}},
            "proposed_split_no_source_overlap": True,
        },
        "domain_coverage": [{"domain": "UCY", "status": "exact_row_count_match"}],
        "horizon_coverage": [{"horizon": 50, "status": "different_eval_pool"}],
        "claim_table": [
            {"claim": "Current locked-policy metrics can be described as full proposed source-level split evaluation.", "status": "rejected"},
            {"claim": "Current locked-policy metrics can be described as available row-level post-repair stress with explicit coverage gap.", "status": "supported"},
        ],
        "next_actions": ["rebuild cache"],
        "no_leakage": {"passed": True},
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = s42al._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_al_source_level_coverage_audit_pass_with_full_split_eval_gap"
