from __future__ import annotations

import numpy as np

from src import stage43_scene_proxy_guarded_latent_policy as ac


class _MiniSplit:
    def __init__(self) -> None:
        self.x = np.zeros((5, 2), dtype=np.float32)
        self.horizon = np.asarray([10, 50, 100, 50, 100], dtype=np.int64)
        self.hard = np.asarray([False, True, True, False, False])
        self.failure = np.asarray([False, False, True, False, False])
        self.easy = np.asarray([True, False, False, True, True])
        self.floor_ade = np.ones(5, dtype=np.float32)
        self.floor_fde = np.ones(5, dtype=np.float32)
        self.domain = np.asarray(["UCY"] * 5)
        self.source_file = np.asarray(["s"] * 5)
        self.scene_id = np.asarray(["scene"] * 5)
        self.waypoint_delta = np.zeros((5, 4, 2), dtype=np.float32)
        self.waypoint_valid = np.ones((5, 4), dtype=bool)


def test_family_masks_block_h100_when_requested() -> None:
    ds = _MiniSplit()
    mask = ac._family_mask(ds, "ab_non_h100")
    assert mask.tolist() == [True, True, False, True, False]
    hard = ac._family_mask(ds, "ab_hard_failure_non_h100")
    assert hard.tolist() == [False, True, False, False, False]


def test_guarded_policy_falls_back_to_stage43_m_on_h100() -> None:
    ds = _MiniSplit()
    pack = {
        "ds": ds,
        "pred_ab": {
            "gain": np.ones(5, dtype=np.float32),
            "harm": np.zeros(5, dtype=np.float32),
            "failure": np.ones(5, dtype=np.float32),
        },
        "ab_ade": np.asarray([0.5, 0.5, 2.0, 0.5, 2.0], dtype=np.float32),
        "ab_fde": np.asarray([0.5, 0.5, 2.0, 0.5, 2.0], dtype=np.float32),
        "m_ade": np.asarray([0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float32),
        "m_fde": np.asarray([0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float32),
        "m_switched": np.asarray([True, True, True, True, True]),
    }
    policy = {
        "family": "ab_non_h100",
        "gain_threshold": 0.0,
        "harm_threshold": 1.0,
        "failure_threshold": 0.0,
    }
    selected_ade, _, _, ab_allowed = ac._select_guarded(pack, policy)
    assert ab_allowed.tolist() == [True, True, False, True, False]
    assert selected_ade.tolist() == [0.5, 0.5, 0.800000011920929, 0.5, 0.800000011920929]


def test_delta_metrics_reports_t100_regression() -> None:
    current = {"t100_raw_frame_full_waypoint_diagnostic_vs_floor": -0.2, "full_waypoint_ade_improvement_vs_floor": 0.3}
    baseline = {"t100_raw_frame_full_waypoint_diagnostic_vs_floor": -0.1, "full_waypoint_ade_improvement_vs_floor": 0.2}
    delta = ac._delta_metrics(current, baseline)
    assert delta["full_waypoint_ade_improvement_vs_floor"] > 0
    assert delta["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] < 0


def test_gate_requires_t100_not_worse() -> None:
    payload = {
        "source": ac.SOURCE,
        "result_source": "fresh_replay_guarded_scene_proxy_policy",
        "stage43_m_report": {"stage43_m_gate": {"verdict": "stage43_m_protected_full_waypoint_latent_candidate_pass"}},
        "stage43_ab_report": {"stage43_ab_gate": {"verdict": "stage43_ab_scene_proxy_augmented_latent_lift_candidate"}},
        "validation_selected_policy": {"policy": {"selected_on": "validation_only", "test_threshold_tuning": False}},
        "row_alignment_passed": True,
        "test_metrics_guarded": {"easy_degradation_vs_floor": 0.0, "t100_scene_proxy_override_rate": 0.0},
        "delta_vs_stage43_m": {
            "full_waypoint_ade_improvement_vs_floor": 0.01,
            "t50_full_waypoint_ade_improvement_vs_floor": 0.0,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.0,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": -0.05,
        },
        "no_leakage": {
            "future_waypoint_input": False,
            "future_endpoint_input": False,
            "test_endpoint_goal_construction": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = ac._gate(payload)
    assert gate["gates"]["core_lift_over_stage43_m"] is True
    assert gate["gates"]["t100_not_worse_than_stage43_m"] is False
    assert gate["deploy_guarded_scene_proxy_latent"] is False
