from __future__ import annotations

from pathlib import Path

from src import stage42_t100_runtime_batch_replay_sufficiency as hu


def test_artifact_summary_detects_missing_row_arrays() -> None:
    summary = hu._artifact_summary(Path("fake.json"), {"metric": {"rows": 10}, "decision_table": {}})

    assert summary["has_per_row_rollout_arrays"] is False
    assert summary["row_like_top_level_keys"] == []


def test_artifact_summary_detects_nested_row_arrays() -> None:
    summary = hu._artifact_summary(Path("fake.json"), {"replay": {"selected_xy": [[0.0, 0.0]]}})

    assert summary["has_per_row_rollout_arrays"] is True
    assert "replay.selected_xy" in summary["row_like_nested_keys"]


def _payload(has_rows: bool = False) -> dict:
    artifacts = [
        {
            "path": "hr",
            "exists": True,
            "top_level_keys": [],
            "row_like_top_level_keys": ["candidate_xy"] if has_rows else [],
            "row_like_nested_keys": [],
            "has_per_row_rollout_arrays": has_rows,
        }
    ]
    payload = {
        "inputs": {
            "stage42_hr": {"exists": True, "stage42_hr_gate": {"passed": 23, "total": 23}},
            "stage42_hs": {"exists": True, "stage42_hs_gate": {"passed": 27, "total": 27}},
            "stage42_ht": {"exists": True, "stage42_ht_gate": {"passed": 19, "total": 19}},
        },
        "policy_artifact": {"exists": True},
        "artifact_summaries": artifacts,
        "required_row_cache_fields": hu.REQUIRED_ROW_CACHE_FIELDS,
        "user_action_required": {"written": True},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    payload["sufficiency"] = hu._sufficiency(payload)
    return payload


def test_sufficiency_marks_batch_replay_not_run_without_row_arrays() -> None:
    suff = hu._sufficiency(_payload(has_rows=False))

    assert suff["runtime_api_ready"] is True
    assert suff["frozen_policy_ready"] is True
    assert suff["row_level_batch_replay_ready"] is False
    assert suff["real_batch_replay_status"] == "not_run"
    assert suff["blocker"] == "missing_row_level_candidate_floor_selected_arrays"


def test_sufficiency_marks_ready_when_row_arrays_exist() -> None:
    suff = hu._sufficiency(_payload(has_rows=True))

    assert suff["row_level_batch_replay_ready"] is True
    assert suff["real_batch_replay_status"] == "ready"
    assert suff["blocker"] == ""


def test_gate_passes_for_honest_not_run_boundary() -> None:
    gate = hu._gate(_payload(has_rows=False))

    assert gate["verdict"] == "stage42_hu_t100_runtime_batch_replay_sufficiency_pass_with_blocker"
    assert gate["passed"] == gate["total"]
    assert gate["gates"]["row_level_replay_not_overclaimed"] is True


def test_gate_fails_if_batch_replay_is_overclaimed() -> None:
    payload = _payload(has_rows=False)
    payload["sufficiency"]["real_batch_replay_status"] = "ready"

    gate = hu._gate(payload)

    assert gate["gates"]["row_level_replay_not_overclaimed"] is False
    assert gate["passed"] == gate["total"] - 1
