from __future__ import annotations

from src import stage42_horizon_retry_decision_map as fy


def test_decision_rows_stop_trajnet_and_ucy_retries() -> None:
    rows = fy._decision_rows(
        {
            "fq_repair_queue": {
                "weak_keys": ["TrajNet|100", "UCY|100"],
                "key_rows": {
                    "TrajNet|100": {
                        "repair_status": {"status": "hard_blocker_no_local_trajnet_h100_long_source"},
                        "candidate_count": 0,
                    },
                    "UCY|100": {
                        "repair_status": {"status": "candidate_support_exists_terms_unverified"},
                        "candidate_count": 6,
                    },
                },
            },
            "fw_source_action": {"consolidated_actions": []},
        }
    )

    by_key = {row["weak_key"]: row for row in rows}
    assert by_key["TrajNet|100"]["decision"] == "stop_model_retry_until_longer_legal_source"
    assert by_key["UCY|100"]["decision"] == "stop_model_retry_until_terms_and_guarded_conversion"
    assert "repeat gain/harm specialist on same source support" in by_key["TrajNet|100"]["blocked_retries"]


def test_summary_blocks_uniform_horizon_claim() -> None:
    decisions = [{"weak_key": "TrajNet|100"}, {"weak_key": "UCY|100"}]
    attempts = [{"policy_promoted": False}, {"policy_promoted": False}]

    summary = fy._summary(decisions, attempts)

    assert summary["stop_repeat_modeling_now"] is True
    assert summary["uniform_horizon_claim_allowed"] is False
    assert summary["promoted_policy_count"] == 0


def test_gate_passes_for_retry_decision_payload() -> None:
    decisions = [
        {
            "weak_key": "TrajNet|100",
            "decision": "stop_model_retry_until_longer_legal_source",
            "allowed_retry_after": ["new legal source support"],
            "next_action_ids": ["FW-H100-TrajNet|100"],
        },
        {
            "weak_key": "UCY|100",
            "decision": "stop_model_retry_until_terms_and_guarded_conversion",
            "allowed_retry_after": ["new guarded converted h100 rows"],
            "next_action_ids": ["FW-H100-UCY|100"],
        },
    ]
    attempts = [{"policy_promoted": False} for _ in range(5)]
    payload = {
        "source": fy.SOURCE,
        "horizon_retry_decisions": decisions,
        "summary": fy._summary(decisions, attempts),
        "claim_boundary": {
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
            "metric_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fy._gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_fy_horizon_retry_decision_pass"


def test_gate_rejects_promoted_retry_policy() -> None:
    decisions = [
        {
            "weak_key": "TrajNet|100",
            "decision": "stop_model_retry_until_longer_legal_source",
            "allowed_retry_after": ["new legal source support"],
            "next_action_ids": ["FW-H100-TrajNet|100"],
        },
        {
            "weak_key": "UCY|100",
            "decision": "stop_model_retry_until_terms_and_guarded_conversion",
            "allowed_retry_after": ["new guarded converted h100 rows"],
            "next_action_ids": ["FW-H100-UCY|100"],
        },
    ]
    attempts = [{"policy_promoted": True}, {"policy_promoted": False}, {"policy_promoted": False}, {"policy_promoted": False}]
    payload = {
        "source": fy.SOURCE,
        "horizon_retry_decisions": decisions,
        "summary": fy._summary(decisions, attempts),
        "claim_boundary": {
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
            "metric_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fy._gate(payload)

    assert gate["gates"]["no_policy_promoted_from_failed_retries"] is False
    assert gate["verdict"] == "stage42_fy_horizon_retry_decision_partial"
