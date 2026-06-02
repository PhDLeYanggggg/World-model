import numpy as np

from src import stage43_scene_graph_context_router as bs
from src import stage43_full_waypoint_latent_dynamics as m


def _split() -> m.WaypointSplit:
    n = 6
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
        horizon=np.asarray([10, 10, 10, 50, 50, 50], dtype=np.int64),
        domain=np.asarray(["A", "A", "A", "A", "B", "B"]),
        source_file=np.asarray(["s1", "s1", "s2", "s2", "s3", "s3"]),
        scene_id=np.asarray(["scene"] * n),
        hard=np.asarray([True, True, False, False, True, False]),
        failure=np.asarray([False] * n),
        easy=np.asarray([False, False, True, True, False, True]),
        scale=np.ones(n, dtype=np.float32),
        feature_names=["a", "b"],
    )


def _arrays() -> dict[str, dict[str, np.ndarray]]:
    # scene wins on A/t10, full wins on B/t50, graph is default elsewhere.
    return {
        "no_context": {
            "selected_ade": np.asarray([0.80, 0.80, 0.80, 0.90, 0.90, 0.90], dtype=np.float32),
            "selected_fde": np.ones(6, dtype=np.float32),
            "switched": np.zeros(6, dtype=bool),
        },
        "scene_proxy_only": {
            "selected_ade": np.asarray([0.50, 0.50, 0.50, 0.90, 0.85, 0.85], dtype=np.float32),
            "selected_fde": np.ones(6, dtype=np.float32),
            "switched": np.ones(6, dtype=bool),
        },
        "graph_history_only": {
            "selected_ade": np.asarray([0.70, 0.70, 0.70, 0.70, 0.70, 0.70], dtype=np.float32),
            "selected_fde": np.ones(6, dtype=np.float32),
            "switched": np.zeros(6, dtype=bool),
        },
        "scene_graph_full": {
            "selected_ade": np.asarray([0.65, 0.65, 0.65, 0.65, 0.40, 0.40], dtype=np.float32),
            "selected_fde": np.ones(6, dtype=np.float32),
            "switched": np.ones(6, dtype=bool),
        },
    }


def test_select_routes_uses_validation_slice_gain() -> None:
    routes, rows = bs._select_routes(_split(), _arrays(), min_rows=2, min_gain=0.01, allowed_variants=set(bs.VARIANTS))
    assert routes["domain_horizon::A::10"] == "scene_proxy_only"
    assert routes["domain_horizon::B::50"] == "scene_graph_full"
    assert any(row["accepted"] for row in rows)


def test_apply_router_falls_back_to_graph_history() -> None:
    ds = _split()
    arrays = _arrays()
    routes = {"domain_horizon::A::10": "scene_proxy_only"}
    selected_ade, _, _, used = bs._apply_router(ds, arrays, routes)
    assert used.tolist()[:3] == ["scene_proxy_only", "scene_proxy_only", "scene_proxy_only"]
    assert used.tolist()[3:] == ["graph_history_only", "graph_history_only", "graph_history_only"]
    assert float(selected_ade[0]) == float(arrays["scene_proxy_only"]["selected_ade"][0])
    assert float(selected_ade[4]) == float(arrays["graph_history_only"]["selected_ade"][4])


def _gate_payload(*, all_delta: float, t50_delta: float, hard_delta: float, easy: float) -> dict:
    return {
        "precondition": {
            "bp_verdict": "stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic",
            "bq_verdict": "stage43_bq_gated_scene_graph_fusion_pass_safe_no_lift_diagnostic",
            "br_verdict": "stage43_br_scene_graph_slice_forensics_pass_targeted_scene_signal",
        },
        "validation_candidates": [{}],
        "selected_candidate": {"test_tuned": False, "route_count": 1},
        "rows": {"test": 6},
        "test_reference_metrics": {"graph_history_only": {}},
        "test_metrics": {
            "easy_degradation_vs_floor": easy,
        },
        "delta_vs_graph_history_only": {
            "all": all_delta,
            "t50": t50_delta,
            "hard_failure": hard_delta,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "raw_scene_or_verified_sdf_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_train_only": True,
            "graph_inputs_past_or_current_only": True,
            "test_route_selection": False,
        },
    }


def test_gate_distinguishes_safe_lift_safe_no_lift_and_unsafe() -> None:
    safe_lift = bs._gate(_gate_payload(all_delta=0.01, t50_delta=-0.01, hard_delta=0.0, easy=0.01))
    assert safe_lift["verdict"] == "stage43_bs_scene_graph_context_router_pass_safe_lift_diagnostic"

    safe_no_lift = bs._gate(_gate_payload(all_delta=-0.01, t50_delta=-0.01, hard_delta=0.0, easy=0.01))
    assert safe_no_lift["verdict"] == "stage43_bs_scene_graph_context_router_pass_safe_no_lift_diagnostic"

    unsafe = bs._gate(_gate_payload(all_delta=0.01, t50_delta=0.01, hard_delta=0.0, easy=0.05))
    assert unsafe["verdict"] == "stage43_bs_scene_graph_context_router_pass_unsafe_diagnostic"


def test_candidate_grid_blocks_full_context_by_default() -> None:
    blocked = bs._candidate_grid(include_full=False)
    assert blocked
    assert all(row["allow_full"] is False for row in blocked)
    assert all("scene_graph_full" not in row["allowed"] for row in blocked)

    diagnostic = bs._candidate_grid(include_full=True)
    assert any(row["allow_full"] is True and "scene_graph_full" in row["allowed"] for row in diagnostic)
