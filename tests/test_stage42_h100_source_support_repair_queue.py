from __future__ import annotations

from src import stage42_h100_source_support_repair_queue as fq


def test_family_bucket_maps_known_external_families() -> None:
    assert fq._family_bucket("UCY/zara02/obsmat.txt") == "zara"
    assert fq._family_bucket("TrajNet/Train/crowds/crowds_zara03.txt") == "zara"
    assert fq._family_bucket("TrajNet/Train/crowds/students003.txt") == "students"
    assert fq._family_bucket("ETH/seq_eth/obsmat.txt") == "eth_seq"


def test_candidate_rows_for_key_filters_ucy_t100_candidates() -> None:
    rows = [
        {
            "domain": "UCY",
            "t100_capable": True,
            "relative_path": "UCY/zara02/obsmat.txt",
            "independent_key": "UCY::UCY/zara02",
            "file_format": "txt",
            "max_track_points": 150,
            "estimated_t100_windows": 300,
        },
        {
            "domain": "ETH_UCY",
            "t100_capable": True,
            "relative_path": "ETH/seq_eth/obsmat.txt",
            "independent_key": "ETH_UCY::ETH/seq_eth",
            "file_format": "txt",
            "max_track_points": 200,
            "estimated_t100_windows": 500,
        },
        {
            "domain": "UCY",
            "t100_capable": False,
            "relative_path": "UCY/students01/obsmat.txt",
            "independent_key": "UCY::UCY/students01",
            "file_format": "txt",
            "max_track_points": 80,
            "estimated_t100_windows": 0,
        },
    ]
    fp_payload = {
        "audits": {
            "UCY|100": {
                "support": {
                    "test_source_rows": [
                        {"name": "/tmp/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt"}
                    ]
                }
            }
        }
    }

    candidates = fq._candidate_rows_for_key("UCY|100", rows, fp_payload)

    assert len(candidates) == 1
    assert candidates[0]["relative_path"] == "UCY/zara02/obsmat.txt"
    assert candidates[0]["target_bucket_match"] is True
    assert candidates[0]["conversion_status"] == "not_converted_not_evaluated"


def test_repair_status_records_trajnet_hard_blocker_without_candidates() -> None:
    status = fq._repair_status("TrajNet|100", [])

    assert status["status"] == "hard_blocker_no_local_trajnet_h100_long_source"
    assert status["can_repair_now"] is False
    assert status["requires_user_action"] is True


def test_gate_passes_for_repair_queue_payload() -> None:
    payload = {
        "source": fq.SOURCE,
        "summary": {
            "input_fp_verdict": "stage42_fp_h100_source_support_audit_pass",
            "weak_key_count": 2,
            "local_files_scanned": 4,
        },
        "key_rows": {
            "TrajNet|100": {
                "repair_status": {
                    "status": "hard_blocker_no_local_trajnet_h100_long_source",
                }
            },
            "UCY|100": {
                "repair_status": {
                    "status": "candidate_support_exists_terms_unverified",
                }
            },
        },
        "user_action_required": [{}, {}],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "local_inventory_only": True,
            "auto_download_executed": False,
        },
        "claim_boundary": {
            "converted_dataset_claim_allowed": False,
            "uniform_horizon_claim": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fq._gate(payload)

    assert gate["verdict"] == "stage42_fq_h100_source_support_repair_queue_pass"
    assert gate["passed"] == gate["total"]
