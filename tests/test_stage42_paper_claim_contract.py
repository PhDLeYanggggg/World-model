from src.stage42_paper_claim_contract import _contract_rows, _gate


def _closure():
    return {
        "supported_claims": [
            {
                "claim": "Stage26 SDD cost-aware selector remains the SDD deployable baseline",
                "status": "supported",
                "paper_use": "baseline",
                "evidence": "cached verified",
            },
            {
                "claim": "Stage37 external t50 safe selector is deployable",
                "status": "supported",
                "paper_use": "external safety floor",
                "evidence": "cached verified",
            },
        ],
        "blocked_claims": [
            {
                "claim": "true 3D or foundation world model",
                "status": "blocked",
                "reason": "not supported",
            }
        ],
    }


def _payload(**summary_overrides):
    closure = _closure()
    summary = {
        "closure_gate_passed": True,
        "supported_claim_count": 6,
        "blocked_claim_count": 7,
        "contract_row_count": 13,
        "paper_files_total": 8,
        "paper_files_existing": 8,
        "paper_files_with_claim_caveat": 8,
        "paper_files_with_stage5c_smc_boundary": 8,
        "metric_seconds_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
        "new_training_or_conversion": False,
    }
    summary.update(summary_overrides)
    contract = [
        {
            "claim": f"supported {idx}",
            "source_status": "supported",
            "paper_role": "main",
            "claim_boundary": "raw-frame 2.5D",
            "deployment_boundary": "protected",
        }
        for idx in range(summary["supported_claim_count"])
    ]
    contract.extend(
        {
            "claim": f"blocked {idx}",
            "source_status": "blocked",
            "paper_role": "blocked_or_limitation_only",
            "claim_boundary": "blocked",
            "deployment_boundary": "not deployable",
        }
        for idx in range(summary["blocked_claim_count"])
    )
    return {
        "summary": summary,
        "contract": contract,
        "paper_file_status": [
            {
                "exists": True,
                "bytes": 500,
                "has_raw_frame_or_dataset_local_caveat": True,
                "mentions_stage5c_or_smc_boundary": True,
            }
            for _ in range(summary["paper_files_total"])
        ],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_contract_rows_mark_blocked_claims_as_limitations():
    rows = _contract_rows(_closure())
    blocked = [row for row in rows if row["source_status"] == "blocked"]
    assert blocked
    assert all(row["paper_role"] == "blocked_or_limitation_only" for row in blocked)
    assert all(row["deployment_boundary"] == "not deployable" for row in blocked)


def test_gate_passes_for_complete_paper_contract():
    gate = _gate(_payload())
    assert gate["verdict"] == "stage42_id_paper_claim_contract_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_paper_files_missing():
    gate = _gate(_payload(paper_files_existing=7))
    assert gate["gates"]["paper_files_exist"] is False
    assert gate["verdict"] == "stage42_id_paper_claim_contract_partial"


def test_gate_fails_if_stage5c_or_smc_boundary_missing():
    gate = _gate(_payload(paper_files_with_stage5c_smc_boundary=3))
    assert gate["gates"]["paper_files_have_stage5c_smc_boundary"] is False


def test_gate_fails_on_metric_seconds_overclaim():
    payload = _payload()
    payload["claim_boundary"]["metric_or_seconds_claim"] = True
    gate = _gate(payload)
    assert gate["gates"]["no_metric_seconds_claim"] is False
