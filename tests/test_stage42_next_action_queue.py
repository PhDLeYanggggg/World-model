from __future__ import annotations

from pathlib import Path

from src import stage42_next_action_queue as da


def _claim_boundary() -> dict[str, bool]:
    return {
        "true_3d": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def test_actions_are_prioritized_and_keep_not_run_status() -> None:
    actions = da._actions()
    priorities = [row["priority"] for row in actions]
    assert priorities == sorted(priorities, reverse=True)
    assert all(row["status"] == "not_run_next_action" for row in actions)
    assert any(row["requires_user_or_external_state"] for row in actions)
    assert all(row["success_gate"] for row in actions)


def test_gate_preserves_negative_context_boundaries() -> None:
    payload = {
        "evidence_files": {f"e{i}": True for i in range(10)},
        "current_evidence": {
            "paper_freeze_status": "candidate_clean",
            "goal_scene_rescue_success": False,
            "neighbor_interaction_rescue_success": False,
        },
        "next_actions": da._actions(),
        "claim_boundary": _claim_boundary(),
    }
    gate = da._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_da_next_action_queue_pass"


def test_gate_fails_if_not_run_action_marked_complete() -> None:
    actions = da._actions()
    actions[0] = {**actions[0], "status": "complete"}
    payload = {
        "evidence_files": {f"e{i}": True for i in range(10)},
        "current_evidence": {
            "paper_freeze_status": "candidate_clean",
            "goal_scene_rescue_success": False,
            "neighbor_interaction_rescue_success": False,
        },
        "next_actions": actions,
        "claim_boundary": _claim_boundary(),
    }
    gate = da._gate(payload)
    assert gate["passed"] < gate["total"]
    assert not gate["gates"]["no_not_run_marked_complete"]


def test_run_writes_isolated_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(da, "OUT_DIR", tmp_path)
    monkeypatch.setattr(da, "REPORT_JSON", tmp_path / "next_action_queue_stage42.json")
    monkeypatch.setattr(da, "REPORT_MD", tmp_path / "next_action_queue_stage42.md")
    monkeypatch.setattr(da, "GATE_MD", tmp_path / "stage42_stage_da_gate.md")
    monkeypatch.setattr(
        da,
        "EVIDENCE_FILES",
        {f"evidence_{i}": tmp_path / f"evidence_{i}.json" for i in range(12)},
    )
    for path in da.EVIDENCE_FILES.values():
        path.write_text("{}", encoding="utf-8")
    payload = da.run_stage42_next_action_queue(refresh_readmes=False)
    assert payload["stage42_da_gate"]["verdict"] == "stage42_da_next_action_queue_pass"
    assert da.REPORT_JSON.exists()
    assert da.REPORT_MD.exists()
    assert da.GATE_MD.exists()
