from __future__ import annotations

from src import stage42_source_specific_conversion_dry_run as dw


def test_candidate_source_ids_excludes_non_source_specific_rows() -> None:
    payload = {
        "candidate_rows": [
            {
                "candidate_class": "source_specific_metric_time_candidate_after_terms",
                "bn_source_specific_candidates": ["UCY_zara01", "UCY_students03"],
            },
            {
                "candidate_class": "time_stride_candidate_dataset_local_only",
                "bn_source_specific_candidates": ["TrajNet_fake"],
            },
        ]
    }

    assert dw._candidate_source_ids(payload) == ["UCY_students03", "UCY_zara01"]


def test_source_cv_plan_requires_three_sources_in_domain() -> None:
    rows = [
        {
            "domain": "UCY",
            "source_id": "a",
            "technical_conversion_ready_after_terms": True,
            "horizon_counts": {"50": 10, "100": 1},
        },
        {
            "domain": "UCY",
            "source_id": "b",
            "technical_conversion_ready_after_terms": True,
            "horizon_counts": {"50": 20, "100": 2},
        },
        {
            "domain": "UCY",
            "source_id": "c",
            "technical_conversion_ready_after_terms": True,
            "horizon_counts": {"50": 30, "100": 3},
        },
        {
            "domain": "ETH_UCY",
            "source_id": "d",
            "technical_conversion_ready_after_terms": True,
            "horizon_counts": {"50": 40, "100": 4},
        },
    ]

    plan = dw._source_cv_plan(rows)

    assert plan["domains"]["UCY"]["source_cv_feasible_after_terms"] is True
    assert plan["domains"]["ETH_UCY"]["source_cv_feasible_after_terms"] is False
    assert plan["domains_with_source_cv_after_terms"] == ["UCY"]


def test_gate_blocks_conversion_rows_and_metric_claims() -> None:
    payload = {
        "stage42_dv_verdict": "stage42_dv_calibration_candidate_manifest_pass",
        "stage42_bn_verdict": "stage42_bn_source_time_geometry_calibration_pass_with_global_claim_blocked",
            "summary": {
                "source_specific_sources_checked": 6,
            "technical_conversion_ready_after_terms_sources": 5,
            "technical_not_ready_sources": ["UCY_zara03"],
            "estimated_t50_windows": 100,
            "estimated_t100_windows": 10,
            "domains_with_source_cv_after_terms": ["UCY"],
            "conversion_allowed_now_sources": 0,
            "full_world_state_rows_written": 0,
            "evaluation_rows_written": 0,
        },
        "no_leakage_preflight": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "future_labels_loss_eval_only": True,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = dw._gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_dw_source_specific_conversion_dry_run_pass"
