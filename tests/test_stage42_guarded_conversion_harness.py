from src.stage42_guarded_conversion_harness import _execution_plan, _gate, _summary


def test_no_ready_contract_builds_no_execution_plan():
    contract = {
        "source": "fresh_stage42_gl_source_conversion_contract",
        "contract_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "domain": "UCY",
                "contract_conversion_ready_now": False,
                "contract_status": "blocked_until_user_terms_path_source_confirmation",
            }
        ],
    }
    plans = _execution_plan(contract, execute=False)
    summary = _summary(contract, plans, execute=False)
    assert plans == []
    assert summary["contract_ready_now"] == 0
    assert summary["conversion_refused_reason"] == "contract_ready_now_is_zero"
    assert summary["conversion_executed"] is False


def test_ready_contract_is_dry_run_by_default():
    contract = {
        "source": "fresh_stage42_gl_source_conversion_contract",
        "contract_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "domain": "UCY",
                "contract_conversion_ready_now": True,
                "contract_status": "queued_for_future_guarded_conversion",
            }
        ],
    }
    plans = _execution_plan(contract, execute=False)
    assert len(plans) == 1
    assert plans[0]["execution_status"] == "dry_run_ready_target"
    assert plans[0]["conversion_executed"] is False
    assert "no-leakage audit" in plans[0]["required_pipeline_steps"]


def test_gate_passes_current_no_ready_refusal():
    payload = {
        "source": "fresh_stage42_gm_guarded_conversion_harness",
        "input_status": {"contract_exists": True},
        "contract_gate": {"passed": 16, "total": 16},
        "summary": {
            "execute_requested": False,
            "contract_ready_now": 0,
            "execution_plan_count": 0,
            "blocked_contract_rows": 5,
            "download_executed": False,
            "conversion_executed": False,
            "feature_store_built": False,
            "no_leakage_audit_executed": False,
            "source_cv_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "converted_dataset_claim_allowed": False,
            "restricted_subset_claim_allowed_now": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required_written": True,
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_gm_guarded_conversion_harness_pass"
