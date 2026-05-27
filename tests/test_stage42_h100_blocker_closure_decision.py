from __future__ import annotations

from src import stage42_h100_blocker_closure_decision as gw


def test_decision_marks_ucy_candidate_as_terms_blocked_not_ready() -> None:
    fp = {"audits": {"UCY|100": {"blockers": ["single_or_sparse_validation_source_support"]}}}
    fq = {
        "key_rows": {
            "UCY|100": {
                "candidate_count": 2,
                "top_candidates": [{"relative_path": "UCY/zara02/obsmat.txt"}],
                "repair_status": {"status": "candidate_support_exists_terms_unverified"},
            }
        }
    }
    bv = {"blocker_matrix": []}
    validations = {
        "ucy_crowd_original": {
            "dataset_id": "ucy_crowd_original",
            "conversion_ready": False,
            "terms_accepted_by_user": False,
            "confirmation_blockers": ["terms_not_accepted"],
            "cf_blockers": ["manual_terms_or_application_required"],
        }
    }

    row = gw._decision_for_key("UCY|100", fp, fq, bv, validations)

    assert row["technical_support_exists"] is True
    assert row["legal_conversion_ready"] is False
    assert row["can_run_repair_now"] is False
    assert row["closure_status"] == "blocked_by_terms_and_conversion_readiness"


def test_decision_preserves_trajnet_missing_long_source_hard_blocker() -> None:
    fp = {"audits": {"TrajNet|100": {"blockers": ["long_horizon_h100_context_still_insufficient"]}}}
    fq = {
        "key_rows": {
            "TrajNet|100": {
                "candidate_count": 0,
                "top_candidates": [],
                "repair_status": {"status": "hard_blocker_no_local_trajnet_h100_long_source"},
            }
        }
    }
    bv = {"blocker_matrix": [{"blocker_id": "TrajNet_raw_long_t100_source_support", "status": "blocked"}]}

    row = gw._decision_for_key("TrajNet|100", fp, fq, bv, {})

    assert row["technical_support_exists"] is False
    assert row["hard_blocker"] == "missing_official_long_raw_trajnet_source"
    assert row["can_run_repair_now"] is False
    assert row["closure_status"] == "hard_blocked_missing_source_support"


def test_gate_passes_for_closure_decision_payload() -> None:
    payload = {
        "input_status": {
            "fp": {"exists": True},
            "fq": {"exists": True},
            "bv": {"exists": True},
            "cg": {"exists": True},
        },
        "summary": {
            "fp_verdict": "stage42_fp_h100_source_support_audit_pass",
            "fq_verdict": "stage42_fq_h100_source_support_repair_queue_pass",
            "bv_verdict": "stage42_bv_source_acquisition_status_pass_blockers_actionable",
            "cg_verdict": "stage42_cg_source_terms_confirmation_validator_pass",
            "weak_keys": ["TrajNet|100", "UCY|100"],
            "can_run_repair_now_count": 0,
            "requires_user_action_count": 2,
        },
        "closure_decisions": [
            {
                "key": "TrajNet|100",
                "hard_blocker": "missing_official_long_raw_trajnet_source",
                "technical_support_exists": False,
                "legal_conversion_ready": False,
                "can_run_repair_now": False,
                "closure_status": "hard_blocked_missing_source_support",
            },
            {
                "key": "UCY|100",
                "hard_blocker": None,
                "technical_support_exists": True,
                "legal_conversion_ready": False,
                "can_run_repair_now": False,
                "closure_status": "blocked_by_terms_and_conversion_readiness",
            },
        ],
        "user_action_required_written": True,
        "user_action_required": [{}, {}],
        "claim_boundary": {
            "uniform_h100_or_t100_claim_allowed": False,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
    }

    gate = gw._gate(payload)

    assert gate["verdict"] == "stage42_gw_h100_blocker_closure_decision_pass"
    assert gate["passed"] == gate["total"]
