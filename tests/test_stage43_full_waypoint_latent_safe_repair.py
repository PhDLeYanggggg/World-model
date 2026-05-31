from __future__ import annotations

import numpy as np

from src import stage43_full_waypoint_latent_safe_repair as o


def test_source_family_maps_known_external_files() -> None:
    assert o._source_family("/tmp/TrajNet/Train/mot/PETS09-S2L1.txt") == "TrajNet_mot"
    assert o._source_family("/tmp/TrajNet/Train/biwi/biwi_hotel.txt") == "TrajNet_biwi"
    assert o._source_family("/tmp/TrajNet/Train/crowds/crowds_zara03.txt") == "TrajNet_crowds"
    assert o._source_family("/tmp/UCY/students03/obsmat.txt") == "UCY"


def test_validation_support_blocks_low_support_and_easy_harm() -> None:
    class Fake:
        source_file = np.asarray(["/tmp/UCY/a.txt"] * 3 + ["/tmp/TrajNet/Train/biwi/b.txt"] * 2)
        horizon = np.asarray([50, 50, 50, 10, 10])
        floor_ade = np.ones(5, dtype=np.float32)
        floor_fde = np.ones(5, dtype=np.float32)
        easy = np.asarray([False, False, False, True, True])

    selected_ade = np.asarray([0.8, 0.7, 0.9, 1.3, 1.4], dtype=np.float32)
    selected_fde = selected_ade.copy()
    switched = np.asarray([True, True, True, True, True])
    table, allowed = o._validation_support_table(
        Fake(),
        selected_ade,
        selected_fde,
        switched,
        min_support_rows=3,
        min_improvement=0.0,
        max_easy_degradation=0.02,
    )
    assert ("UCY", 50) in allowed
    assert ("TrajNet_biwi", 10) not in allowed
    assert table["TrajNet_biwi|10"]["reason"] == "blocked_insufficient_validation_support"


def test_gate_marks_t100_fallback_repair_not_positive() -> None:
    payload = {
        "source": o.SOURCE,
        "stage43_n_precondition": {"negative_source_count": 1, "t100_improvement": -0.1},
        "repair_policy": {"selection_data": "validation_only", "test_threshold_tuning": False},
        "no_leakage": {"test_threshold_tuning": False, "future_waypoint_input": False},
        "overall_full_test_metrics": {
            "full_waypoint_ade_improvement_vs_floor": 0.1,
            "t50_full_waypoint_ade_improvement_vs_floor": 0.1,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.1,
            "easy_degradation_vs_floor": 0.0,
        },
        "source_domain_caveats": {"negative_source_count": 0, "domain_easy_harm_count": 0},
        "t100_repair": {
            "improvement": 0.0,
            "easy_degradation": 0.0,
            "status": "t100_harm_repaired_by_fallback_not_positive",
        },
        "claim_boundary": {
            "t100_positive_success": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = o._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["deploy_safe_repair_policy"] is True
    assert gate["t100_positive_success"] is False
