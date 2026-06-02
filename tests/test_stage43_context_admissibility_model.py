from __future__ import annotations

import numpy as np

from src import stage43_context_admissibility_model as bt
from src import stage43_full_waypoint_latent_dynamics as m


def _split() -> m.WaypointSplit:
    n = 4
    return m.WaypointSplit(
        split="val",
        x=np.zeros((n, 3), dtype=np.float32),
        waypoint_delta=np.zeros((n, 4, 2), dtype=np.float32),
        waypoint_valid=np.ones((n, 4), dtype=bool),
        floor_waypoint_delta=np.zeros((n, 4, 2), dtype=np.float32),
        floor_ade=np.ones(n, dtype=np.float32),
        floor_fde=np.ones(n, dtype=np.float32),
        y_failure=np.zeros(n, dtype=np.float32),
        y_gain=np.zeros(n, dtype=np.float32),
        y_harm=np.zeros(n, dtype=np.float32),
        y_density=np.zeros(n, dtype=np.float32),
        horizon=np.asarray([10, 50, 50, 100], dtype=np.int64),
        domain=np.asarray(["UCY", "UCY", "ETH_UCY", "TrajNet"]),
        source_file=np.asarray(["a", "a", "b", "c"]),
        scene_id=np.asarray(["s", "s", "s", "s"]),
        hard=np.asarray([False, True, False, True]),
        failure=np.asarray([False, False, True, False]),
        easy=np.asarray([True, False, False, True]),
        scale=np.ones(n, dtype=np.float32),
        feature_names=["a", "b", "c"],
    )


def _batch() -> bt.ContextBatch:
    arrays = {
        "no_context": {
            "selected_ade": np.asarray([0.9, 0.9, 0.9, 0.9], dtype=np.float32),
            "selected_fde": np.ones(4, dtype=np.float32),
            "switched": np.zeros(4, dtype=bool),
        },
        "scene_proxy_only": {
            "selected_ade": np.asarray([0.8, 0.5, 0.8, 1.2], dtype=np.float32),
            "selected_fde": np.ones(4, dtype=np.float32),
            "switched": np.ones(4, dtype=bool),
        },
        "graph_history_only": {
            "selected_ade": np.asarray([0.7, 0.7, 0.7, 0.7], dtype=np.float32),
            "selected_fde": np.ones(4, dtype=np.float32),
            "switched": np.zeros(4, dtype=bool),
        },
        "scene_graph_full": {
            "selected_ade": np.asarray([0.6, 0.6, 0.9, 0.9], dtype=np.float32),
            "selected_fde": np.ones(4, dtype=np.float32),
            "switched": np.ones(4, dtype=bool),
        },
    }
    return bt.ContextBatch(ds=_split(), arrays=arrays, info={})


def test_supervision_marks_gain_and_harm_per_context_variant() -> None:
    sup = bt._supervision(_batch(), gain_margin=0.01, harm_margin=0.01)
    # scene wins on row 1 but harms row 3; full wins rows 0/1 but harms rows 2/3.
    assert sup["gain_label"][:, 0].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert sup["harm_label"][:, 0].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert sup["gain_label"][:, 1].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert sup["harm_label"][:, 1].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_apply_policy_uses_predicted_gain_and_harm_gate() -> None:
    batch = _batch()
    pred = {
        "gain_prob": np.asarray([[0.9, 0.95], [0.9, 0.9], [0.95, 0.9], [0.9, 0.9]], dtype=np.float32),
        "harm_prob": np.asarray([[0.9, 0.05], [0.05, 0.10], [0.9, 0.9], [0.9, 0.9]], dtype=np.float32),
        "gain_reg": np.asarray([[-0.1, 0.10], [0.20, 0.10], [-0.1, -0.2], [-0.2, -0.3]], dtype=np.float32),
    }
    selected_ade, _, _, used = bt._apply_policy(
        batch,
        pred,
        {"gain_threshold": 0.8, "harm_threshold": 0.2, "predicted_gain_threshold": 0.01},
    )
    assert used.tolist() == ["scene_graph_full", "scene_proxy_only", "graph_history_only", "graph_history_only"]
    assert np.allclose(selected_ade, [0.6, 0.5, 0.7, 0.7])


def _gate_payload(*, all_delta: float, t50_delta: float, hard_delta: float, easy: float) -> dict:
    return {
        "result_source": "fresh_row_level_harm_aware_context_admissibility",
        "precondition": {
            "bp_verdict": "stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic",
            "bq_verdict": "stage43_bq_gated_scene_graph_fusion_pass_safe_no_lift_diagnostic",
            "br_verdict": "stage43_br_scene_graph_slice_forensics_pass_targeted_scene_signal",
            "bs_verdict": "stage43_bs_scene_graph_context_router_pass_safe_no_lift_diagnostic",
        },
        "rows": {"train": 4, "val": 4, "test": 4},
        "model": {"checkpoint": "README.md", "checkpoint_committed": False},
        "validation_selection": {"test_tuned": False},
        "test_metrics": {"rows": 4, "easy_degradation_vs_floor": easy},
        "test_reference_metrics": {"graph_history_only": {}},
        "delta_vs_graph_history_only": {"all": all_delta, "t50": t50_delta, "hard_failure": hard_delta},
        "admissibility_diagnostics": {"test": {"scene_proxy_only": {}}},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_variant_error_label_only": True,
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


def test_gate_distinguishes_safe_lift_safe_no_lift_and_unsafe() -> None:
    safe_lift = bt._gate(_gate_payload(all_delta=0.01, t50_delta=-0.01, hard_delta=0.0, easy=0.01))
    assert safe_lift["verdict"] == "stage43_bt_context_admissibility_pass_safe_lift_diagnostic"

    safe_no_lift = bt._gate(_gate_payload(all_delta=-0.01, t50_delta=-0.01, hard_delta=0.0, easy=0.01))
    assert safe_no_lift["verdict"] == "stage43_bt_context_admissibility_pass_safe_no_lift_diagnostic"

    unsafe = bt._gate(_gate_payload(all_delta=0.01, t50_delta=0.01, hard_delta=0.0, easy=0.04))
    assert unsafe["verdict"] == "stage43_bt_context_admissibility_pass_unsafe_diagnostic"
