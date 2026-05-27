from pathlib import Path

from src import stage42_floor_free_proximity_guard_robustness as he
from src.stage14_pipeline import read_json


def _payload():
    path = Path("outputs/stage42_long_research/floor_free_proximity_guard_robustness_stage42.json")
    if path.exists():
        return read_json(path, {})
    return he._build_payload()


def test_stage42_he_gate_passes() -> None:
    payload = _payload()
    gate = payload["stage42_he_gate"]
    assert gate["passed"] == gate["total"]
    assert payload["summary"]["causal_floor_fallback_used"] is True
    assert payload["summary"]["global_floor_removal_allowed"] is False


def test_stage42_he_bootstrap_positive_and_easy_safe() -> None:
    payload = _payload()
    boot = payload["bootstrap"]
    assert boot["all"]["bootstrap_n"] >= he.BOOTSTRAP_N
    assert boot["t50"]["low"] > 0.0
    assert boot["hard_failure"]["low"] > 0.0
    assert boot["easy_degradation"]["high"] <= he.EASY_LIMIT


def test_stage42_he_teacherless_but_not_floorless() -> None:
    payload = _payload()
    assert payload["summary"]["teacher_gate_used"] is False
    assert payload["claim_boundary"]["teacher_gate_removed_for_repaired_floor_free_candidate"] is True
    assert payload["claim_boundary"]["causal_floor_safety_fallback_still_required"] is True
    assert payload["claim_boundary"]["global_floor_removal_allowed"] is False
