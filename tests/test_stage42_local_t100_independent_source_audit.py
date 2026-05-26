from src import stage42_local_t100_independent_source_audit as s42bh


def test_independent_key_deduplicates_scene_formats() -> None:
    a = {"relative_path": "UCY/students03/obsmat.txt", "suggested_domain": "UCY_or_ETH_UCY"}
    b = {"relative_path": "UCY/students03/obsmat_px.txt", "suggested_domain": "UCY_or_ETH_UCY"}
    assert s42bh._independent_key(a) == s42bh._independent_key(b)
    assert s42bh._canonical_domain(a) == "UCY"


def test_choose_canonical_prefers_more_t100_windows() -> None:
    rows = [
        {"relative_path": "UCY/a/obsmat.txt", "estimated_t100_windows": 5},
        {"relative_path": "UCY/a/obsmat_px.txt", "estimated_t100_windows": 7},
    ]
    assert s42bh._choose_canonical(rows)["relative_path"] == "UCY/a/obsmat_px.txt"


def test_build_independent_sources_reports_duplicates() -> None:
    inventory = [
        {
            "relative_path": "UCY/a/obsmat.txt",
            "path": "/tmp/a",
            "suggested_domain": "UCY_or_ETH_UCY",
            "t100_capable": True,
            "synthetic_or_diagnostic": False,
            "estimated_t100_windows": 5,
        },
        {
            "relative_path": "UCY/a/obsmat_px.txt",
            "path": "/tmp/b",
            "suggested_domain": "UCY_or_ETH_UCY",
            "t100_capable": True,
            "synthetic_or_diagnostic": False,
            "estimated_t100_windows": 7,
        },
    ]
    sources, audit = s42bh._build_independent_sources(inventory)
    assert len(sources) == 1
    assert audit["raw_t100_capable_files"] == 2
    assert audit["duplicate_or_alternate_format_group_count"] == 1


def test_folds_require_three_independent_sources() -> None:
    sources = [
        {"source_id": "a", "domain": "UCY", "estimated_t100_windows": 10},
        {"source_id": "b", "domain": "UCY", "estimated_t100_windows": 9},
    ]
    assert s42bh._folds_for_domain(sources, "UCY") == []
    sources.append({"source_id": "c", "domain": "UCY", "estimated_t100_windows": 8})
    folds = s42bh._folds_for_domain(sources, "UCY")
    assert len(folds) == 3
    assert all(fold["validation_source"] != fold["holdout_source"] for fold in folds)


def test_gate_requires_global_t100_blocked() -> None:
    payload = {
        "source": "fresh_local_independent_source_audit",
        "bd_verdict": "stage42_bd_local_t100_source_inventory_pass",
        "bg_verdict": "stage42_bg_local_t100_protected_policy_pass_with_global_t100_blocker",
        "duplicate_audit": {},
        "support_matrix": {"ETH_UCY": {}, "UCY": {}, "TrajNet": {}},
        "domain_summary": {"UCY": {}},
        "summary": {
            "raw_t100_capable_files": 4,
            "independent_t100_sources": 3,
            "ucy_t100_source_cv_supported": True,
            "blocked_domains": ["ETH_UCY", "TrajNet"],
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "holdout_used_for_threshold": False,
            "duplicate_scene_versions_treated_as_independent": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "global_t100_positive_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42bh._gate(payload)
    assert gate["passed"] == gate["total"]
