from src.stage42_conversion_capability_intake_bridge import (
    _capability_for_dataset,
    _gate,
    _merge_intake_with_capability,
)


def _dw_payload():
    return {
        "stage42_dw_gate": {"verdict": "stage42_dw_source_specific_conversion_dry_run_pass"},
        "source_rows": [
            {
                "source_id": "UCY_zara01",
                "domain": "UCY",
                "dataset": "UCY",
                "trajectory_file": "zara01.txt",
                "path_exists": True,
                "rows": 100,
                "agents": 8,
                "common_frame_step": 10,
                "horizon_counts": {"50": 40, "100": 20},
                "history_horizon_counts": {"50": 35, "100": 15},
                "t50_capable": True,
                "t100_capable": True,
                "causal_velocity_possible": True,
                "central_velocity_used": False,
                "technical_conversion_ready_after_terms": True,
                "blocked_by": ["terms/source_identity/path_version_not_confirmed"],
            },
            {
                "source_id": "UCY_zara02",
                "domain": "UCY",
                "dataset": "UCY",
                "trajectory_file": "zara02.txt",
                "path_exists": True,
                "rows": 120,
                "agents": 7,
                "common_frame_step": 10,
                "horizon_counts": {"50": 60, "100": 25},
                "history_horizon_counts": {"50": 55, "100": 18},
                "t50_capable": True,
                "t100_capable": True,
                "causal_velocity_possible": True,
                "central_velocity_used": False,
                "technical_conversion_ready_after_terms": True,
                "blocked_by": [],
            },
            {
                "source_id": "ETH_seq_eth",
                "domain": "ETH_UCY",
                "dataset": "ETH",
                "trajectory_file": "seq_eth.txt",
                "path_exists": True,
                "rows": 80,
                "agents": 5,
                "horizon_counts": {"50": 20, "100": 10},
                "t50_capable": True,
                "t100_capable": True,
                "causal_velocity_possible": True,
                "central_velocity_used": False,
                "technical_conversion_ready_after_terms": True,
            },
        ],
        "source_cv_plan": {
            "domains": {
                "UCY": {"source_cv_feasible_after_terms": True, "sources": ["UCY_zara01", "UCY_zara02", "UCY_zara03"]},
                "ETH_UCY": {"source_cv_feasible_after_terms": False, "sources": ["ETH_seq_eth"]},
            }
        },
        "no_leakage_preflight": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
    }


def test_capability_prefill_is_dry_run_only():
    cap = _capability_for_dataset("ucy_crowd_original", _dw_payload())

    assert cap["domain"] == "UCY"
    assert cap["source_specific_dry_run_available"] is True
    assert cap["source_cv_feasible_after_terms"] is True
    assert cap["technical_ready_after_terms_sources"] == 2
    assert cap["t50_windows_after_terms"] == 100
    assert cap["t100_windows_after_terms"] == 45
    assert cap["conversion_allowed_now"] is False
    assert cap["converted_now"] is False
    assert cap["evaluated_now"] is False


def test_merge_adds_capability_without_user_confirmation_or_conversion_ready():
    intake = {
        "datasets": [
            {
                "dataset_id": "ucy_crowd_original",
                "user_confirmation": {
                    "terms_accepted_by_user": False,
                    "terms_acceptance_date": "",
                    "allowed_use": "",
                    "local_path": "",
                    "source_identity": "",
                    "confirmed_by_user": "",
                },
                "conversion_ready_now": False,
            }
        ]
    }

    merged = _merge_intake_with_capability(intake, _dw_payload())
    row = merged["datasets"][0]

    assert row["conversion_capability_prefill"]["source_count"] == 2
    assert row["conversion_capability_prefill"]["conversion_allowed_now"] is False
    assert row["conversion_ready_now"] is False
    assert row["converted_now"] is False
    assert row["evaluated_now"] is False
    assert row["user_confirmation"]["local_path"] == ""


def test_ge_gate_passes_for_safe_bridge_payload():
    payload = {
        "source": "fresh_stage42_ge_conversion_capability_intake_bridge",
        "input_status": {
            "dw_exists": True,
            "intake_exists": True,
            "dw_verdict": "stage42_dw_source_specific_conversion_dry_run_pass",
        },
        "summary": {
            "intake_rows": 5,
            "rows_with_capability_prefill": 5,
            "rows_with_source_specific_dry_run": 2,
            "rows_with_source_cv_feasible_after_terms": 1,
            "technical_ready_after_terms_sources": 5,
            "t50_windows_after_terms": 100,
            "t100_windows_after_terms": 10,
            "rows_with_user_confirmation": 0,
            "conversion_ready_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
        },
        "snapshot_written": True,
        "intake_template_updated": True,
        "user_action_required_written": True,
        "claim_boundary": {
            "download_executed": False,
            "conversion_executed": False,
            "evaluation_executed": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ge_conversion_capability_intake_bridge_pass"
