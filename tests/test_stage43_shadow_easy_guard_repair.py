from __future__ import annotations

import numpy as np

from src import stage43_shadow_easy_guard_repair as m


class DummySplit:
    def __init__(self) -> None:
        self.x = np.zeros((4, 2), dtype=np.float32)
        self.floor_ade = np.ones(4, dtype=np.float32)
        self.floor_fde = np.ones(4, dtype=np.float32)
        self.hard = np.asarray([True, False, False, True])
        self.failure = np.asarray([False, False, False, True])
        self.easy = np.asarray([False, True, True, False])
        self.horizon = np.asarray([10, 10, 50, 50])
        self.domain = np.asarray(["TrajNet", "TrajNet", "UCY", "UCY"])
        self.source_file = np.asarray(["/tmp/PETS09-S2L1.txt", "/tmp/zara.txt", "/tmp/students.txt", "/tmp/biwi_hotel.txt"])


def test_source_family_maps_common_external_sources() -> None:
    assert m._source_family("/x/PETS09-S2L1.txt") == "pets"
    assert m._source_family("/x/crowds_zara03.txt") == "zara"
    assert m._source_family("/x/biwi_hotel.txt") == "biwi"
    assert m._source_family("/x/students003.txt") == "students"


def test_apply_guard_blocks_domain_and_unsupported_family() -> None:
    ds = DummySplit()
    pred = {
        "gain": np.ones(4, dtype=np.float32),
        "harm": np.zeros(4, dtype=np.float32),
        "failure": np.ones(4, dtype=np.float32),
    }
    cand_ade = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    cand_fde = cand_ade.copy()
    disagreement = {
        "model_floor_mean_disagreement": np.zeros(4, dtype=np.float32),
        "model_floor_endpoint_disagreement": np.zeros(4, dtype=np.float32),
    }
    families = np.asarray([m._source_family(x) for x in ds.source_file])
    policy = {
        "base_policy": {
            "gain_threshold": 0.1,
            "harm_threshold": 0.1,
            "failure_threshold": 0.1,
            "disagreement_threshold": 1.0,
            "endpoint_disagreement_threshold": 1.0,
        },
        "blocked_domains": ["TrajNet"],
        "blocked_horizons": [],
        "blocked_domain_horizons": [],
        "source_family_support_mode": "global",
        "supported_global_families": ["students", "biwi"],
    }
    selected_ade, _, switched = m._apply_guarded_policy(ds, pred, cand_ade, cand_fde, disagreement, families, policy)
    assert switched.tolist() == [False, False, True, True]
    assert selected_ade.tolist() == [1.0, 1.0, 0.5, 0.5]


def test_gate_passes_for_shadow_safe_test_safe_lift() -> None:
    payload = {
        "stage43_cb_precondition": {"verdict": "stage43_cb_downstream_easy_guard_val_safe_test_easy_mismatch"},
        "result_source": "fresh_shadow_validation_easy_guard_repair",
        "protocol": {"train_only_heads_refit": True},
        "shadow_validation": {"plan": {"calibration_rows": 10, "shadow_holdout_rows": 5, "support": {"global_families": ["biwi"]}}},
        "no_leakage": {
            "test_threshold_tuning": False,
            "guard_uses_future_labels": False,
            "guard_uses_test_endpoints": False,
        },
        "shadow_policy": {
            "selected_shadow_policy": {
                "shadow_holdout_metrics": {"easy_degradation_vs_floor": 0.0},
            }
        },
        "test_once": {
            "metrics": {
                "easy_degradation_vs_floor": 0.0,
                "full_waypoint_ade_improvement_vs_floor": 0.03,
                "t50_full_waypoint_ade_improvement_vs_floor": 0.0,
                "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.05,
            },
            "slice_tables": {"domain": {"UCY": {}}, "horizon": {"50": {}}, "source_family": {"biwi": {}}},
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "long_objective_complete": False,
    }
    gate = m._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_cc_shadow_easy_guard_repair_pass"
