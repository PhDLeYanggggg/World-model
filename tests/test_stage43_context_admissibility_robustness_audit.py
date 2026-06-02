from __future__ import annotations

import numpy as np

from src import stage43_context_admissibility_robustness_audit as bu
from src import stage43_full_waypoint_latent_dynamics as m


def _split() -> m.WaypointSplit:
    n = 6
    return m.WaypointSplit(
        split="test",
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
        horizon=np.asarray([10, 10, 50, 50, 100, 100], dtype=np.int64),
        domain=np.asarray(["UCY", "UCY", "UCY", "ETH_UCY", "TrajNet", "TrajNet"]),
        source_file=np.asarray(["a", "a", "a", "b", "c", "c"]),
        scene_id=np.asarray(["s"] * n),
        hard=np.asarray([False, True, True, False, True, False]),
        failure=np.asarray([False, False, True, False, False, False]),
        easy=np.asarray([True, False, False, True, False, True]),
        scale=np.ones(n, dtype=np.float32),
        feature_names=[],
    )


def test_bootstrap_summary_reports_positive_delta() -> None:
    ds = _split()
    selected = np.asarray([0.7, 0.6, 0.7, 0.65, 0.8, 0.75], dtype=np.float32)
    graph = np.asarray([0.8, 0.7, 0.8, 0.75, 0.85, 0.8], dtype=np.float32)
    boot = bu._bootstrap_summary(ds, selected, graph, n=100, seed=1)
    assert boot["metrics"]["all_delta_vs_graph"]["mean"] > 0
    assert boot["metrics"]["hard_failure_delta_vs_graph"]["rows"] == 3
    assert boot["metrics"]["t50_delta_vs_graph"]["rows"] == 2


def test_slice_audit_finds_positive_and_negative_slices() -> None:
    ds = _split()
    selected = np.asarray([0.7, 0.6, 0.7, 0.65, 0.9, 0.9], dtype=np.float32)
    graph = np.asarray([0.8, 0.7, 0.8, 0.75, 0.85, 0.8], dtype=np.float32)
    used = np.asarray(["scene_proxy_only", "scene_proxy_only", "graph_history_only", "scene_graph_full", "scene_proxy_only", "graph_history_only"])
    audit = bu._slice_audit(ds, selected, graph, used, min_rows=1)
    assert audit["positive_slice_count"] > 0
    assert audit["negative_slice_count"] > 0
    assert any(row["slice"] == "horizon_100" for row in audit["core_weak_slices"])


def _payload(
    *,
    all_low: float,
    hard_low: float,
    t50_low: float,
    t100_low: float,
    t100_high: float,
    easy_high: float,
    easy_hazards: int = 0,
) -> dict:
    return {
        "precondition": {
            "bt_verdict": "stage43_bt_context_admissibility_pass_safe_lift_diagnostic",
            "bt_gate": {"passed": 14, "total": 14},
        },
        "checkpoint": {"path": "README.md", "committed": False},
        "replay_diff_vs_stage43_bt_report": {
            "full_waypoint_ade_improvement_vs_floor": 0.0,
            "t50_full_waypoint_ade_improvement_vs_floor": 0.0,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.0,
            "easy_degradation_vs_floor": 0.0,
        },
        "bootstrap": {
            "n": 1000,
            "metrics": {
                "all_delta_vs_graph": {"low": all_low, "mean": all_low, "high": all_low, "rows": 6},
                "hard_failure_delta_vs_graph": {"low": hard_low, "mean": hard_low, "high": hard_low, "rows": 3},
                "t50_delta_vs_graph": {"low": t50_low, "mean": t50_low, "high": t50_low, "rows": 2},
                "t100_raw_frame_delta_vs_graph": {"low": t100_low, "mean": (t100_low + t100_high) / 2.0, "high": t100_high, "rows": 2},
                "easy_degradation_delta_vs_graph": {"low": 0.0, "mean": 0.0, "high": easy_high, "rows": 3},
            },
        },
        "slice_audit": {"slice_count": 1, "easy_hazard_slice_count": easy_hazards},
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


def test_gate_distinguishes_robust_partial_and_fragile() -> None:
    robust = bu._gate(_payload(all_low=0.01, hard_low=0.01, t50_low=0.01, t100_low=0.01, t100_high=0.02, easy_high=0.0))
    assert robust["verdict"] == "stage43_bu_context_admissibility_robust_lift_pass"

    partial = bu._gate(
        _payload(all_low=0.01, hard_low=0.01, t50_low=0.01, t100_low=-0.01, t100_high=0.01, easy_high=0.0, easy_hazards=2)
    )
    assert partial["verdict"] == "stage43_bu_context_admissibility_partial_robust_lift_pass"

    fragile = bu._gate(_payload(all_low=-0.01, hard_low=0.01, t50_low=-0.01, t100_low=-0.01, t100_high=0.01, easy_high=0.0))
    assert fragile["verdict"] == "stage43_bu_context_admissibility_fragile_lift_diagnostic_pass"
