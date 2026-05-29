from pathlib import Path

from src import stage42_current_reviewer_replay_package as ju


def test_gate_passes_for_current_reviewer_replay_package() -> None:
    payload = ju.run_stage42_current_reviewer_replay_package(refresh_readmes=False)
    gate = payload["stage42_ju_gate"]
    assert gate["verdict"] == "stage42_ju_current_reviewer_replay_package_pass"
    assert gate["passed"] == gate["total"]
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False


def test_replay_commands_do_not_train_or_execute_forbidden_stages() -> None:
    payload = ju.run_stage42_current_reviewer_replay_package(refresh_readmes=False)
    lowered = "\n".join(payload["replay_commands"]).lower()
    assert "train_" not in lowered
    assert "stage5c" not in lowered
    assert "smc" not in lowered


def test_package_keeps_context_modules_blocked_as_independent_claims() -> None:
    payload = ju.run_stage42_current_reviewer_replay_package(refresh_readmes=False)
    blocked = set(payload["summary"]["blocked_independent_claims"])
    assert "scene_goal_independent_main_claim" in blocked
    assert "neighbor_interaction_independent_main_claim" in blocked
    assert "JEPA_downstream_main_claim" in blocked
    assert "Transformer_independent_main_claim" in blocked
    assert payload["summary"]["ao"]["positive_incremental_context_variants"] == []


def test_file_row_hashes_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"ok": true}\n', encoding="utf-8")
    row = ju._file_row(path)
    assert row["exists"] is True
    assert row["size_bytes"] > 0
    assert len(row["sha256"]) == 64
