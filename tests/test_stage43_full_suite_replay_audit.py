from __future__ import annotations

from src import stage43_full_suite_replay_audit as ar


def test_parse_pytest_summary_extracts_counts() -> None:
    output = "======================= 997 passed, 2 skipped in 42.50s ======================="
    parsed = ar._parse_pytest_summary(output)
    assert parsed["found"] is True
    assert parsed["summary_line"] == "997 passed, 2 skipped in 42.50s"
    assert parsed["duration_seconds"] == 42.5
    assert parsed["counts"]["passed"] == 997
    assert parsed["counts"]["skipped"] == 2


def test_parse_pytest_summary_accepts_parenthesized_duration() -> None:
    output = "====================== 1360 passed in 3750.05s (1:02:30) ======================"
    parsed = ar._parse_pytest_summary(output)
    assert parsed["found"] is True
    assert parsed["summary_line"] == "1360 passed in 3750.05s"
    assert parsed["duration_seconds"] == 3750.05
    assert parsed["counts"]["passed"] == 1360


def _payload(*, return_code: int = 0, failed: int = 0, found: bool = True) -> dict:
    return {
        "source": ar.SOURCE,
        "pytest_run": {
            "command": ["python", "-m", "pytest", "tests"],
            "return_code": return_code,
            "timed_out": False,
            "elapsed_wall_seconds": 1.0,
            "pytest_summary": {
                "found": found,
                "summary_line": "10 passed in 1.00s" if found else "",
                "counts": {"passed": 10, "failed": failed, "errors": 0},
            },
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "goal_complete": False,
    }


def test_gate_passes_for_clean_full_suite_replay() -> None:
    gate = ar._gate(_payload())
    assert gate["passed"] == gate["total"]
    assert gate["full_suite_replay_passed"] is True
    assert gate["goal_complete"] is False


def test_gate_fails_when_pytest_fails() -> None:
    gate = ar._gate(_payload(return_code=1, failed=1))
    assert gate["gates"]["pytest_exit_zero"] is False
    assert gate["gates"]["no_failed_or_error_tests"] is False
    assert gate["full_suite_replay_passed"] is False


def test_gate_fails_when_summary_missing() -> None:
    gate = ar._gate(_payload(return_code=0, found=False))
    assert gate["gates"]["pytest_summary_found"] is False
    assert gate["full_suite_replay_passed"] is False
