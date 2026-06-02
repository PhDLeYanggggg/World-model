from __future__ import annotations

import numpy as np

from src import stage43_downstream_easy_guard_audit as m


class DummySplit:
    def __init__(self) -> None:
        self.x = np.zeros((4, 2), dtype=np.float32)
        self.floor_ade = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.floor_fde = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.floor_waypoint_delta = np.zeros((4, 4, 2), dtype=np.float32)
        self.waypoint_delta = np.zeros((4, 4, 2), dtype=np.float32)
        self.waypoint_valid = np.ones((4, 4), dtype=bool)
        self.hard = np.asarray([True, True, False, False])
        self.failure = np.asarray([True, False, False, False])
        self.easy = np.asarray([False, False, True, True])
        self.horizon = np.asarray([50, 50, 10, 25], dtype=np.int64)
        self.domain = np.asarray(["UCY", "UCY", "ETH_UCY", "TrajNet"])
        self.source_file = np.asarray(["a", "a", "b", "c"])


def _pred() -> dict[str, np.ndarray]:
    waypoint = np.zeros((4, 4, 2), dtype=np.float32)
    waypoint[0, :, 0] = 0.1
    waypoint[1, :, 0] = 0.2
    waypoint[2, :, 0] = 3.0
    waypoint[3, :, 0] = 4.0
    return {
        "waypoint": waypoint,
        "gain": np.asarray([0.9, 0.8, 0.9, 0.9], dtype=np.float32),
        "harm": np.asarray([0.02, 0.04, 0.20, 0.20], dtype=np.float32),
        "failure": np.asarray([0.9, 0.7, 0.9, 0.9], dtype=np.float32),
    }


def test_disagreement_features_are_inference_side_quantities() -> None:
    ds = DummySplit()
    out = m._disagreement_features(ds, _pred())
    assert out["model_floor_mean_disagreement"].shape == (4,)
    assert out["model_floor_endpoint_disagreement"].shape == (4,)
    assert out["model_floor_mean_disagreement"][0] < out["model_floor_mean_disagreement"][2]


def test_select_with_easy_guard_blocks_high_harm_rows() -> None:
    ds = DummySplit()
    pred = _pred()
    candidate_ade = np.asarray([0.5, 0.6, 3.0, 4.0], dtype=np.float32)
    candidate_fde = candidate_ade.copy()
    dis = m._disagreement_features(ds, pred)
    policy = {
        "gain_threshold": 0.5,
        "harm_threshold": 0.10,
        "failure_threshold": 0.5,
        "disagreement_threshold": 1.0,
        "endpoint_disagreement_threshold": 1.0,
    }
    selected_ade, _, switched = m._select_with_easy_guard(ds, pred, candidate_ade, candidate_fde, dis, policy)
    assert switched.tolist() == [True, True, False, False]
    assert np.allclose(selected_ade[:2], [0.5, 0.6])
    assert np.allclose(selected_ade[2:], [1.0, 1.0])


def test_gate_reports_validation_safe_test_easy_mismatch() -> None:
    payload = {
        "stage43_ca_precondition": {"verdict": "stage43_ca_latent_adapter_downstream_heads_partial_lift"},
        "result_source": "fresh_validation_only_easy_guard_replay",
        "protocol": {"train_only_heads_refit": True},
        "no_leakage": {
            "future_labels_as_inputs": False,
            "future_labels_train_eval_only": True,
            "test_threshold_tuning": False,
            "guard_uses_future_labels": False,
            "guard_uses_test_endpoints": False,
        },
        "validation_easy_guard": {"metrics": {"easy_degradation_vs_floor": 0.0, "switch_rate": 0.1}},
        "test_once": {
            "metrics": {
                "easy_degradation_vs_floor": 0.05,
                "full_waypoint_ade_improvement_vs_floor": 0.03,
                "t50_full_waypoint_ade_improvement_vs_floor": -0.01,
                "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.06,
            },
            "slice_summary": {"domain": {"UCY": {}}, "horizon": {"50": {}}, "source_file": {"a": {}}},
        },
        "validation_test_gap": {"easy_degradation_vs_floor": 0.05},
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "long_objective_complete": False,
    }
    gate = m._gate(payload)
    assert gate["gates"]["validation_easy_safe_policy_found"] is True
    assert gate["gates"]["test_easy_preserved"] is False
    assert gate["verdict"] == "stage43_cb_downstream_easy_guard_val_safe_test_easy_mismatch"
