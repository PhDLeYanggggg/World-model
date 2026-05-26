from src import stage42_local_t100_conversion_readiness as s42be


def test_horizon_counts_use_track_lengths() -> None:
    counts = s42be._horizon_counts([101, 120])
    assert counts["10"] == 201
    assert counts["50"] == 121
    assert counts["100"] == 21


def test_history_counts_require_past_and_future() -> None:
    counts = s42be._history_counts([164])
    assert counts["k64_h100"] == 1
    assert counts["k64_h50"] == 51


def test_source_cv_plan_requires_three_sources() -> None:
    readiness = [
        {"domain": "UCY", "source_id": "a", "schema_conversion_ready": True, "horizon_counts": {"100": 10}},
        {"domain": "UCY", "source_id": "b", "schema_conversion_ready": True, "horizon_counts": {"100": 20}},
        {"domain": "UCY", "source_id": "c", "schema_conversion_ready": True, "horizon_counts": {"100": 30}},
        {"domain": "ETH_UCY", "source_id": "d", "schema_conversion_ready": True, "horizon_counts": {"100": 1}},
    ]
    plan = s42be._source_cv_plan(readiness)
    assert plan["domains"]["UCY"]["source_cv_feasible_after_conversion"] is True
    assert len(plan["domains"]["UCY"]["folds"]) == 3
    assert plan["domains"]["ETH_UCY"]["source_cv_feasible_after_conversion"] is False


def test_choose_xy_columns_handles_obsmat_px_layout() -> None:
    rows = [
        [1.0, 1.0, 323.0, 0.0, 430.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 266.0, 0.0, 539.0, 0.0, 0.0, 0.0],
    ]
    assert s42be._choose_xy_columns(rows) == (2, 4)


def test_gate_passes_for_readiness_payload() -> None:
    payload = {
        "source": "fresh_local_conversion_readiness",
        "bd_verdict": "stage42_bd_local_t100_source_inventory_pass",
        "source_cv_plan": {
            "domains": {
                "UCY": {
                    "source_cv_feasible_after_conversion": True,
                }
            }
        },
        "summary": {
            "candidate_files": 3,
            "schema_conversion_ready_files": 3,
            "estimated_t50_windows": 100,
            "estimated_t100_windows": 10,
            "domains_with_source_cv_after_conversion": ["UCY"],
            "full_feature_store_written": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42be._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_be_local_t100_conversion_readiness_pass"
