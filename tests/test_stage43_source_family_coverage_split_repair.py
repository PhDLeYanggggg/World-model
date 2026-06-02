from __future__ import annotations

from src import stage43_source_family_coverage_split_repair as m


def _toy_records() -> dict[str, dict[str, object]]:
    return {
        "a1": {"source_file": "a1", "source_id": "a1", "domain": "A", "family": "zara", "scene_ids": ["s1"], "rows": 10, "horizon_counts": {"50": 10}, "hard_rows": 1, "failure_rows": 1, "easy_rows": 5, "basename": "a1"},
        "a2": {"source_file": "a2", "source_id": "a2", "domain": "A", "family": "zara", "scene_ids": ["s2"], "rows": 8, "horizon_counts": {"50": 8}, "hard_rows": 1, "failure_rows": 1, "easy_rows": 4, "basename": "a2"},
        "a3": {"source_file": "a3", "source_id": "a3", "domain": "A", "family": "hotel", "scene_ids": ["s3"], "rows": 6, "horizon_counts": {"25": 6}, "hard_rows": 0, "failure_rows": 0, "easy_rows": 3, "basename": "a3"},
        "b1": {"source_file": "b1", "source_id": "b1", "domain": "B", "family": "biwi", "scene_ids": ["s4"], "rows": 9, "horizon_counts": {"50": 9}, "hard_rows": 1, "failure_rows": 1, "easy_rows": 4, "basename": "b1"},
        "b2": {"source_file": "b2", "source_id": "b2", "domain": "B", "family": "biwi", "scene_ids": ["s5"], "rows": 7, "horizon_counts": {"50": 7}, "hard_rows": 1, "failure_rows": 1, "easy_rows": 4, "basename": "b2"},
        "c1": {"source_file": "c1", "source_id": "c1", "domain": "C", "family": "students", "scene_ids": ["s6"], "rows": 11, "horizon_counts": {"50": 11}, "hard_rows": 2, "failure_rows": 1, "easy_rows": 4, "basename": "c1"},
        "c2": {"source_file": "c2", "source_id": "c2", "domain": "C", "family": "students", "scene_ids": ["s7"], "rows": 5, "horizon_counts": {"50": 5}, "hard_rows": 0, "failure_rows": 0, "easy_rows": 3, "basename": "c2"},
    }


def test_coverage_assignment_covers_test_families(monkeypatch) -> None:
    monkeypatch.setattr(m, "DOMAINS", ["A", "B", "C"])
    records = _toy_records()
    payload = m._build_coverage_assignments(records)
    coverage = m._coverage_summary(records, payload["assignments"])
    assert coverage["global_family_coverage_pass"] is True
    assert coverage["domain_family_coverage_pass"] is True
    assert set(m._split_summary(records, payload["assignments"])["test"]["domains"]) == {"A", "B", "C"}


def test_leakage_reports_source_disjoint_and_basename_overlap() -> None:
    records = _toy_records()
    assignments = {source: "train" for source in records}
    assignments["a1"] = "val"
    assignments["a2"] = "test"
    leakage = m._leakage_summary(records, assignments)
    assert leakage["source_file_disjoint"] is True
    assert "basename_overlap_counts" in leakage


def test_gate_requires_coverage() -> None:
    payload = {
        "stage43_f_precondition": {"verdict": "stage43_f_source_level_split_ready"},
        "coverage_split": {
            "assignment_hash": "abc",
            "blockers": {"singleton_domain_families": ["A|hotel"]},
        },
        "split_summary": {
            "train": {"rows": 1, "domains": ["ETH_UCY"]},
            "val": {"rows": 1, "domains": ["ETH_UCY", "TrajNet", "UCY"]},
            "test": {"rows": 1, "domains": ["ETH_UCY", "TrajNet", "UCY"]},
        },
        "coverage_summary": {
            "global_family_coverage_pass": True,
            "domain_family_coverage_pass": True,
            "domain_families_by_split": {"test": ["A|zara"]},
        },
        "no_leakage": {
            "source_file_disjoint": True,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "basename_overlap_counts": {},
        },
        "claim_boundary": {
            "new_training_or_evaluation_not_run": True,
            "requires_cache_rebuild_before_training": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    gate = m._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_ce_source_family_coverage_split_repair_ready"
