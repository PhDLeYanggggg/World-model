from __future__ import annotations

from src import stage43_t100_source_coverage_preflight as s


def test_family_summary_marks_family_feasible_with_four_sources() -> None:
    records = [
        {"split": "train", "source_family": "UCY", "source_short": "a", "source_file": "/x/a", "h100_rows": 100},
        {"split": "train", "source_family": "UCY", "source_short": "b", "source_file": "/x/b", "h100_rows": 100},
        {"split": "val", "source_family": "UCY", "source_short": "c", "source_file": "/x/c", "h100_rows": 100},
        {"split": "test", "source_family": "UCY", "source_short": "d", "source_file": "/x/d", "h100_rows": 100},
    ]
    summary = s._family_summary(records, min_val_source_count=2, min_source_rows=50)
    assert summary["UCY"]["feasible_source_stable_validation"] is True
    proposal = summary["UCY"]["source_level_split_proposal"]
    assert proposal["status"] == "feasible"
    assert len(proposal["val_sources"]) == 2


def test_family_summary_blocks_three_source_family_for_train_val_test_stability() -> None:
    records = [
        {"split": "train", "source_family": "ETH_UCY", "source_short": "a", "source_file": "/x/a", "h100_rows": 100},
        {"split": "val", "source_family": "ETH_UCY", "source_short": "b", "source_file": "/x/b", "h100_rows": 100},
        {"split": "test", "source_family": "ETH_UCY", "source_short": "c", "source_file": "/x/c", "h100_rows": 100},
    ]
    summary = s._family_summary(records, min_val_source_count=2, min_source_rows=50)
    assert summary["ETH_UCY"]["feasible_source_stable_validation"] is False
    assert summary["ETH_UCY"]["reason"] == "blocked_cannot_hold_train_val_test_with_source_stable_validation"


def test_gate_accepts_audit_with_feasible_and_blocked_families() -> None:
    payload = {
        "source": s.SOURCE,
        "stage43_r_precondition": {
            "verdict": "stage43_r_source_stable_h100_guard_blocks_t100_false_positive"
        },
        "cache_inputs": {
            "train": {"exists": True},
            "val": {"exists": True},
            "test": {"exists": True},
        },
        "result_source": "fresh_h100_source_coverage_preflight",
        "protocol": {
            "audit_only": True,
            "rewrites_cache": False,
            "test_threshold_tuning": False,
        },
        "preflight_summary": {
            "h100_source_count": 4,
            "family_count": 2,
            "feasible_family_count": 1,
            "blocked_family_count": 1,
            "can_rebuild_source_stable_h100_validation": True,
            "needs_more_h100_sources_for_uniform_t100": True,
        },
        "no_leakage": {
            "test_threshold_tuning": False,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["rebuild_source_stable_h100_split_recommended"] is True
    assert gate["uniform_t100_blocker"] is True
