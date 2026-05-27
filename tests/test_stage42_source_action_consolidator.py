from __future__ import annotations

from src import stage42_source_action_consolidator as fw


def test_h100_missing_trajnet_hard_blocker() -> None:
    missing = fw._h100_missing({"status": "hard_blocker_no_local_trajnet_h100_long_source"}, [])

    assert "official longer TrajNet-compatible raw source" in missing
    assert "terms confirmation" in missing


def test_dedupe_actions_keeps_highest_priority_first() -> None:
    actions = [
        {"action_id": "A", "priority": 10},
        {"action_id": "B", "priority": 30},
        {"action_id": "A", "priority": 5},
    ]

    out = fw._dedupe_actions(actions)  # type: ignore[arg-type]

    assert [row["action_id"] for row in out] == ["B", "A"]


def test_gate_requires_ucy_and_trajnet_actions() -> None:
    payload = {
        "input_status": {
            "source_terms_gap": {"exists": True},
            "source_closure": {"exists": True},
            "h100_queue": {"exists": True},
            "unified_queue": {"exists": True},
        },
        "summary": {"actions_total": 8, "conversion_ready_now": 0},
        "consolidated_actions": [
            {
                "action_id": "FW-TERMS-ucy_crowd_original",
                "status": "not_run_user_action_required",
                "claim_guard": "guard",
            },
            {
                "action_id": "FW-H100-UCY|100",
                "status": "not_run_user_action_required",
                "claim_guard": "guard",
            },
            {
                "action_id": "FW-H100-TrajNet|100",
                "status": "not_run_user_action_required",
                "claim_guard": "guard",
            },
            *[
                {
                    "action_id": f"X{i}",
                    "status": "not_run_user_action_required",
                    "claim_guard": "guard",
                }
                for i in range(5)
            ],
        ],
        "user_action_required_written": True,
        "claim_boundary": {
            "download_executed": False,
            "conversion_executed": False,
            "evaluation_executed": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fw._gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_fw_source_action_consolidator_pass"


def test_gate_fails_if_action_marked_complete() -> None:
    payload = {
        "input_status": {
            "source_terms_gap": {"exists": True},
            "source_closure": {"exists": True},
            "h100_queue": {"exists": True},
            "unified_queue": {"exists": True},
        },
        "summary": {"actions_total": 8, "conversion_ready_now": 0},
        "consolidated_actions": [
            {"action_id": "FW-TERMS-ucy_crowd_original", "status": "complete", "claim_guard": "guard"},
            {"action_id": "FW-H100-UCY|100", "status": "not_run", "claim_guard": "guard"},
            {"action_id": "FW-H100-TrajNet|100", "status": "not_run", "claim_guard": "guard"},
            *[{"action_id": f"X{i}", "status": "not_run", "claim_guard": "guard"} for i in range(5)],
        ],
        "user_action_required_written": True,
        "claim_boundary": {
            "download_executed": False,
            "conversion_executed": False,
            "evaluation_executed": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fw._gate(payload)

    assert gate["gates"]["no_action_marked_complete"] is False
    assert gate["verdict"] == "stage42_fw_source_action_consolidator_partial"
