from __future__ import annotations

import numpy as np

from src import stage43_context_hazard_attribution_guard as bw
from src import stage43_context_admissibility_slice_safe_repair as bv
from src import stage43_full_waypoint_latent_dynamics as m


def _split() -> m.WaypointSplit:
    n = 10
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
        horizon=np.asarray([10, 10, 10, 10, 50, 50, 50, 100, 100, 100], dtype=np.int64),
        domain=np.asarray(["A", "A", "A", "A", "B", "B", "B", "C", "C", "C"]),
        source_file=np.asarray(["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s3", "s3", "s3"]),
        scene_id=np.asarray(["scene"] * n),
        hard=np.asarray([False, False, True, False, True, False, True, False, True, False]),
        failure=np.asarray([False, False, True, False, False, False, True, False, False, False]),
        easy=np.asarray([True, True, False, True, False, True, False, True, False, True]),
        scale=np.ones(n, dtype=np.float32),
        feature_names=[],
    )


def test_context_easy_hazard_audit_attributes_context_harm() -> None:
    ds = _split()
    graph = np.asarray([0.8] * 10, dtype=np.float32)
    selected = graph.copy()
    selected[[0, 1, 3]] = np.asarray([0.9, 0.91, 0.88], dtype=np.float32)
    used = np.asarray(["scene_proxy_only"] * 4 + [bv.DEFAULT_VARIANT] * 6, dtype=object)
    audit = bw._context_easy_hazard_audit(
        ds,
        selected,
        graph,
        used,
        min_rows=1,
        min_context_easy_rows=2,
        rate_threshold=0.5,
        mean_harm_threshold=0.01,
    )
    assert audit["context_hazard_slice_count"] >= 1
    assert audit["top_context_hazard_slices"][0]["mean_context_easy_harm_vs_graph"] > 0.0


def test_hazard_key_guard_blocks_context_for_matching_family() -> None:
    ds = _split()
    graph = np.asarray([0.8] * 10, dtype=np.float32)
    selected = graph.copy()
    selected[[0, 1, 3]] = np.asarray([0.9, 0.91, 0.88], dtype=np.float32)
    used = np.asarray(["scene_proxy_only"] * 4 + [bv.DEFAULT_VARIANT] * 6, dtype=object)
    keys = bw._hazard_keys(
        ds,
        selected,
        graph,
        used,
        family="source_horizon",
        min_context_easy_rows=2,
        rate_threshold=0.5,
        mean_harm_threshold=0.01,
    )
    assert keys
    assert any("s1" in key for key in keys)


def _payload(*, graph_abs: int, selected_abs: int, selected_context_hazards: int, bt_context_hazards: int) -> dict:
    return {
        "precondition": {
            "bt_verdict": "stage43_bt_context_admissibility_pass_safe_lift_diagnostic",
            "bv_verdict": "stage43_bv_context_admissibility_slice_repair_diagnostic_remaining_risk",
        },
        "validation_selection": {"test_tuned": False},
        "source_overlap": {"held_out_source_level": True},
        "absolute_slice_audit": {
            "graph_history_floor": {"easy_hazard_slice_count": graph_abs},
            "selected_guard": {"easy_hazard_slice_count": selected_abs},
        },
        "context_induced_hazard_audit": {
            "selected_guard": {"context_hazard_slice_count": selected_context_hazards},
            "bt_unrepaired": {"context_hazard_slice_count": bt_context_hazards},
        },
        "test_metrics": {"rows": 10, "easy_degradation_vs_floor": 0.0},
        "delta_vs_graph_history_only": {
            "all": 0.01,
            "t50": 0.0,
            "t100_raw_frame_diagnostic": 0.0,
            "hard_failure": 0.0,
            "easy_degradation": 0.0,
        },
        "bootstrap": {
            "n": 1000,
            "metrics": {
                "easy_degradation_delta_vs_graph": {"high": 0.0},
            },
        },
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


def test_gate_distinguishes_context_safe_lift_from_floor_inherent_risk() -> None:
    safe = bw._gate(_payload(graph_abs=3, selected_abs=2, selected_context_hazards=0, bt_context_hazards=2))
    assert safe["verdict"] == "stage43_bw_context_hazard_guard_pass_context_safe_lift_diagnostic"

    floor_inherent = bw._gate(_payload(graph_abs=3, selected_abs=2, selected_context_hazards=1, bt_context_hazards=2))
    assert floor_inherent["verdict"] == "stage43_bw_context_hazard_attribution_pass_floor_inherent_risk"
