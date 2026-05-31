from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import stage43_integrated_tail_h100_policy as u


def test_standardized_copy_does_not_mutate_original() -> None:
    @dataclass
    class Fake:
        x: np.ndarray
        split: str = "test"

    ds = Fake(np.asarray([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))
    copied = u._standardized_copy(ds, np.asarray([2.0, 2.0], dtype=np.float32), np.asarray([2.0, 3.0], dtype=np.float32))
    assert copied is not ds
    assert np.allclose(ds.x, [[2.0, 4.0], [6.0, 8.0]])
    assert np.allclose(copied.x, [[0.0, 2.0 / 3.0], [2.0, 2.0]])


def test_gate_accepts_family_limited_h100_when_safe() -> None:
    payload = {
        "source": u.SOURCE,
        "preconditions": {
            "stage43_p_verdict": "stage43_p_tail_horizon_adapter_pass_t100_still_fallback",
            "stage43_t_verdict": "stage43_t_source_stable_h100_specialist_deployable",
            "stage43_t_deployed": True,
            "stage43_t_validation_source_safe": True,
        },
        "replay_hashes": {
            "stage43_p_model_hash_replay": "p",
            "stage43_p_model_hash_reported": "p",
            "stage43_t_model_hash_replay": "t",
            "stage43_t_model_hash_reported": "t",
        },
        "integrated_policy": {"specialist_rows_in_full_test": 10},
        "training_protocol": {"test_threshold_tuning": False, "max_easy_degradation": 0.02},
        "no_leakage": {
            "test_threshold_tuning": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
        },
        "delta_vs_stage43_p": {
            "full_waypoint_ade_improvement_delta": 0.001,
            "t50_delta": 0.0,
        },
        "integrated_full_test_metrics": {
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.001,
            "easy_degradation_vs_floor": 0.0,
        },
        "h100_specialist_slice": {
            "integrated_slice": {
                "full_waypoint_ade_improvement_vs_floor": 0.02,
                "easy_degradation_vs_floor": 0.0,
            },
            "negative_source_count": 0,
        },
        "bootstrap_ci": {
            "metrics": {
                "full_waypoint_ade_improvement_vs_floor": {
                    "low": 0.4,
                },
            },
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = u._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["deploy_integrated_policy"] is True
    assert gate["uniform_t100_success"] is False


def test_gate_rejects_h100_easy_harm() -> None:
    payload = {
        "source": u.SOURCE,
        "preconditions": {
            "stage43_p_verdict": "stage43_p_tail_horizon_adapter_pass_t100_still_fallback",
            "stage43_t_verdict": "stage43_t_source_stable_h100_specialist_deployable",
            "stage43_t_deployed": True,
            "stage43_t_validation_source_safe": True,
        },
        "replay_hashes": {
            "stage43_p_model_hash_replay": "p",
            "stage43_p_model_hash_reported": "p",
            "stage43_t_model_hash_replay": "t",
            "stage43_t_model_hash_reported": "t",
        },
        "integrated_policy": {"specialist_rows_in_full_test": 10},
        "training_protocol": {"test_threshold_tuning": False, "max_easy_degradation": 0.02},
        "no_leakage": {
            "test_threshold_tuning": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
        },
        "delta_vs_stage43_p": {
            "full_waypoint_ade_improvement_delta": 0.001,
            "t50_delta": 0.0,
        },
        "integrated_full_test_metrics": {
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.001,
            "easy_degradation_vs_floor": 0.0,
        },
        "h100_specialist_slice": {
            "integrated_slice": {
                "full_waypoint_ade_improvement_vs_floor": 0.02,
                "easy_degradation_vs_floor": 0.05,
            },
            "negative_source_count": 0,
        },
        "bootstrap_ci": {
            "metrics": {
                "full_waypoint_ade_improvement_vs_floor": {
                    "low": 0.4,
                },
            },
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = u._gate(payload)
    assert gate["passed"] < gate["total"]
    assert gate["deploy_integrated_policy"] is False
