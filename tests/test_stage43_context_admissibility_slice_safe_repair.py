from __future__ import annotations

import numpy as np

from src import stage43_context_admissibility_slice_safe_repair as bv
from src import stage43_full_waypoint_latent_dynamics as m


def _split() -> m.WaypointSplit:
    n = 8
    return m.WaypointSplit(
        split="val",
        x=np.zeros((n, 2), dtype=np.float32),
        waypoint_delta=np.zeros((n, 4, 2), dtype=np.float32),
        waypoint_valid=np.ones((n, 4), dtype=bool),
        floor_waypoint_delta=np.zeros((n, 4, 2), dtype=np.float32),
        floor_ade=np.ones(n, dtype=np.float32),
        floor_fde=np.ones(n, dtype=np.float32),
        y_failure=np.zeros(n, dtype=np.float32),
        y_gain=np.zeros(n, dtype=np.float32),
        y_harm=np.zeros(n, dtype=np.float32),
        y_density=np.zeros(n, dtype=np.float32),
        horizon=np.asarray([10, 10, 50, 50, 100, 100, 50, 100], dtype=np.int64),
        domain=np.asarray(["A", "A", "A", "B", "B", "B", "C", "C"]),
        source_file=np.asarray(["s1", "s1", "s1", "s2", "s2", "s2", "s3", "s3"]),
        scene_id=np.asarray(["scene"] * n),
        hard=np.asarray([False, True, True, False, True, False, True, False]),
        failure=np.asarray([False, False, True, False, False, False, True, False]),
        easy=np.asarray([True, False, False, True, False, True, False, True]),
        scale=np.ones(n, dtype=np.float32),
        feature_names=[],
    )


def test_slice_table_marks_safe_and_unsafe_keys() -> None:
    ds = _split()
    graph = np.asarray([0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 0.8, 0.7], dtype=np.float32)
    selected = np.asarray([0.7, 0.7, 0.7, 0.9, 0.8, 0.9, 0.7, 0.65], dtype=np.float32)
    used = np.asarray(["scene_proxy_only"] * len(graph), dtype=object)
    table = bv._slice_table(ds, selected, graph, used, min_rows=1, min_delta=0.0, easy_limit=0.02)
    assert table["summary"]["horizon"]["safe_keys"] >= 1
    assert table["summary"]["horizon"]["unsafe_keys"] >= 1
    assert any(row["unsafe"] for row in table["families"]["source_horizon"].values())


def test_block_mask_supports_conservative_modes() -> None:
    ds = _split()
    graph = np.asarray([0.8] * 8, dtype=np.float32)
    selected = np.asarray([0.7, 0.7, 0.7, 0.7, 0.9, 0.9, 0.7, 0.9], dtype=np.float32)
    used = np.asarray(["scene_proxy_only"] * 8, dtype=object)
    table = bv._slice_table(ds, selected, graph, used, min_rows=1, min_delta=0.0, easy_limit=0.02)
    block_t100 = bv._block_mask(ds, used, table, "block_t100")
    strict = bv._block_mask(ds, used, table, "strict_safe_no_t100")
    assert int(block_t100.sum()) == 3
    assert int(strict.sum()) >= int(block_t100.sum())


def _payload(*, easy_safe: bool, slice_safe: bool, core_lift: bool, t100_low: float) -> dict:
    delta = {
        "all": 0.01 if core_lift else 0.0,
        "t50": 0.0,
        "t100_raw_frame_diagnostic": t100_low,
        "hard_failure": 0.0,
        "easy_degradation": 0.0,
    }
    return {
        "precondition": {
            "bt_verdict": "stage43_bt_context_admissibility_pass_safe_lift_diagnostic",
            "bu_verdict": "stage43_bu_context_admissibility_partial_robust_lift_pass",
            "bu_gate": {"passed": 12, "total": 12},
        },
        "validation_slice_table": {"summary": {"horizon": {"keys": 1}}},
        "validation_selection": {"test_tuned": False},
        "test_metrics": {"rows": 8, "easy_degradation_vs_floor": 0.0 if easy_safe else 0.05},
        "bootstrap": {
            "n": 1000,
            "metrics": {
                "all_delta_vs_graph": {"low": delta["all"], "mean": delta["all"], "high": delta["all"], "rows": 8},
                "hard_failure_delta_vs_graph": {"low": 0.0, "mean": 0.0, "high": 0.0, "rows": 4},
                "t50_delta_vs_graph": {"low": 0.0, "mean": 0.0, "high": 0.0, "rows": 3},
                "t100_raw_frame_delta_vs_graph": {"low": t100_low, "mean": t100_low, "high": max(t100_low, 0.01), "rows": 3},
                "easy_degradation_delta_vs_graph": {"low": 0.0, "mean": 0.0, "high": 0.0 if easy_safe else 0.05, "rows": 3},
            },
        },
        "slice_audit": {"slice_count": 1, "easy_hazard_slice_count": 0 if slice_safe else 1},
        "delta_vs_graph_history_only": delta,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_variant_error_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_train_only": True,
            "graph_inputs_past_or_current_only": True,
            "test_threshold_selection": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "raw_scene_or_verified_sdf_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }


def test_gate_verdicts_are_safety_aware() -> None:
    full = bv._gate(_payload(easy_safe=True, slice_safe=True, core_lift=True, t100_low=0.01))
    assert full["verdict"] == "stage43_bv_context_admissibility_slice_safe_repair_pass"

    partial = bv._gate(_payload(easy_safe=True, slice_safe=True, core_lift=True, t100_low=-0.01))
    assert partial["verdict"] == "stage43_bv_context_admissibility_slice_safe_partial_lift_pass"

    risky = bv._gate(_payload(easy_safe=True, slice_safe=False, core_lift=True, t100_low=0.01))
    assert risky["verdict"] == "stage43_bv_context_admissibility_slice_repair_diagnostic_remaining_risk"
