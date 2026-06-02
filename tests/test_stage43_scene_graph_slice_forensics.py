from __future__ import annotations

import numpy as np

from src import stage43_scene_graph_slice_forensics as br
from src import stage43_full_waypoint_latent_dynamics as m


def _arrays(selected: list[float], floor: list[float] | None = None) -> dict:
    floor_arr = np.asarray(floor if floor is not None else [1.0] * len(selected), dtype=np.float32)
    selected_arr = np.asarray(selected, dtype=np.float32)
    return {
        "selected_ade": selected_arr,
        "selected_fde": selected_arr,
        "switched": selected_arr < floor_arr,
        "floor_ade": floor_arr,
        "floor_fde": floor_arr,
    }


def test_slice_row_reports_scene_graph_deltas() -> None:
    ds = m.WaypointSplit(
        split="test",
        x=np.zeros((4, 2), dtype=np.float32),
        waypoint_delta=np.zeros((4, 4, 2), dtype=np.float32),
        waypoint_valid=np.ones((4, 4), dtype=bool),
        floor_waypoint_delta=np.zeros((4, 4, 2), dtype=np.float32),
        floor_ade=np.ones(4, dtype=np.float32),
        floor_fde=np.ones(4, dtype=np.float32),
        y_failure=np.zeros(4, dtype=np.float32),
        y_gain=np.zeros(4, dtype=np.float32),
        y_harm=np.zeros(4, dtype=np.float32),
        y_density=np.zeros(4, dtype=np.float32),
        horizon=np.asarray([10, 50, 50, 100]),
        domain=np.asarray(["UCY", "UCY", "ETH_UCY", "TrajNet"]),
        source_file=np.asarray(["a", "a", "b", "c"]),
        scene_id=np.asarray(["s", "s", "s", "s"]),
        hard=np.asarray([False, True, False, False]),
        failure=np.asarray([False, False, True, False]),
        easy=np.asarray([True, False, False, True]),
        scale=np.ones(4, dtype=np.float32),
        feature_names=[],
    )
    arrays = {
        "no_context": _arrays([0.9, 0.9, 0.9, 0.9]),
        "scene_proxy_only": _arrays([0.8, 0.8, 0.8, 0.8]),
        "graph_history_only": _arrays([0.85, 0.85, 0.85, 0.85]),
        "scene_graph_full": _arrays([0.83, 0.83, 0.83, 0.83]),
    }
    row = br._slice_row("all", np.ones(4, dtype=bool), ds, arrays)
    assert row["best_variant"] == "scene_proxy_only"
    assert row["scene_minus_graph"] > 0.0
    assert row["full_minus_graph"] > 0.0


def _payload(*, scene_over_graph: int, scene_over_no_context: int) -> dict:
    slice_rows = []
    for i in range(max(scene_over_graph, scene_over_no_context, 5)):
        slice_rows.append(
            {
                "slice": f"domain_UCY_horizon_{i}",
                "rows": 200,
                "best_variant": "scene_proxy_only" if i < scene_over_graph else "graph_history_only",
                "scene_minus_graph": 0.01 if i < scene_over_graph else -0.01,
                "scene_minus_no_context": 0.01 if i < scene_over_no_context else -0.01,
                "full_minus_graph": -0.01,
                "improvements": {
                    "no_context": 0.1,
                    "scene_proxy_only": 0.12,
                    "graph_history_only": 0.11,
                    "scene_graph_full": 0.1,
                },
            }
        )
    slice_rows.extend(
        [
            {"slice": "hard_failure", "rows": 200, "best_variant": "graph_history_only", "scene_minus_graph": -0.01, "scene_minus_no_context": 0.0, "full_minus_graph": -0.01, "improvements": {}},
            {"slice": "easy", "rows": 200, "best_variant": "graph_history_only", "scene_minus_graph": -0.01, "scene_minus_no_context": 0.0, "full_minus_graph": -0.01, "improvements": {}},
            {"slice": "horizon_50", "rows": 200, "best_variant": "graph_history_only", "scene_minus_graph": -0.01, "scene_minus_no_context": 0.0, "full_minus_graph": -0.01, "improvements": {}},
        ]
    )
    return {
        "summary": {
            "slice_count": len(slice_rows),
            "eligible_slice_count": len(slice_rows),
            "scene_over_graph_slice_count": scene_over_graph,
            "scene_over_no_context_slice_count": scene_over_no_context,
            "full_over_graph_slice_count": 0,
            "best_variant_counts": {"no_context": 0, "scene_proxy_only": scene_over_graph, "graph_history_only": 5, "scene_graph_full": 0},
        },
        "slice_rows": slice_rows,
        "bp_precondition": {"verdict": "stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic"},
        "bq_precondition": {"verdict": "stage43_bq_gated_scene_graph_fusion_pass_safe_no_lift_diagnostic"},
        "variant_replay": {variant: {"recomputed_metrics": {}} for variant in br.VARIANTS},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_train_only": True,
            "graph_inputs_past_or_current_only": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "raw_scene_or_verified_sdf_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }


def test_gate_marks_targeted_scene_signal() -> None:
    gate = br._gate(_payload(scene_over_graph=2, scene_over_no_context=2))
    assert gate["passed"] == gate["total"]
    assert gate["targeted_scene_signal"] is True
    assert gate["verdict"] == "stage43_br_scene_graph_slice_forensics_pass_targeted_scene_signal"


def test_gate_marks_weak_scene_signal() -> None:
    gate = br._gate(_payload(scene_over_graph=0, scene_over_no_context=2))
    assert gate["passed"] == gate["total"]
    assert gate["targeted_scene_signal"] is False
    assert gate["weak_scene_signal"] is True
    assert gate["verdict"] == "stage43_br_scene_graph_slice_forensics_pass_weak_scene_signal_diagnostic"


def test_gate_marks_no_scene_signal() -> None:
    gate = br._gate(_payload(scene_over_graph=0, scene_over_no_context=0))
    assert gate["passed"] == gate["total"]
    assert gate["targeted_scene_signal"] is False
    assert gate["weak_scene_signal"] is False
    assert gate["verdict"] == "stage43_br_scene_graph_slice_forensics_pass_no_scene_signal_diagnostic"
