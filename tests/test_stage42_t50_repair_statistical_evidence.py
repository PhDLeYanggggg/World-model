from __future__ import annotations

import numpy as np

from src import stage42_t50_repair_statistical_evidence as bz


def test_ci_positive_requires_bootstrap_n_and_bound() -> None:
    assert bz._ci_positive({"low": 0.01, "high": 0.03, "bootstrap_n": bz.BOOTSTRAP_N})
    assert not bz._ci_positive({"low": -0.01, "high": 0.03, "bootstrap_n": bz.BOOTSTRAP_N})
    assert not bz._ci_positive({"low": 0.01, "high": 0.03, "bootstrap_n": bz.BOOTSTRAP_N - 1})
    assert bz._ci_positive({"low": -0.1, "high": 0.01, "bootstrap_n": bz.BOOTSTRAP_N}, easy=True)
    assert not bz._ci_positive({"low": -0.1, "high": 0.03, "bootstrap_n": bz.BOOTSTRAP_N}, easy=True)


def test_slice_mask_matches_domain_and_horizon_only_on_test() -> None:
    data = {
        "dataset": np.array(["UCY", "UCY", "TrajNet", "UCY"], dtype=object),
        "horizon": np.array([50, 25, 50, 50]),
    }
    split = np.array(["test", "test", "test", "train"], dtype=object)
    mask = bz._slice_mask(data, split, "UCY|50")
    assert mask.tolist() == [True, False, False, False]


def test_gate_passes_for_bootstrap_backed_repair() -> None:
    payload = {
        "source": "unit",
        "input_reports": {"stage42_by_verdict": "stage42_by_t50_floor_relaxability_repair_pass"},
        "summary": {
            "target_union_ci_positive_and_easy_safe": True,
            "bootstrap_n": bz.BOOTSTRAP_N,
            "teacher_floor_context_required": True,
            "floor_free_neural_deployable": False,
        },
        "slice_evidence": {
            "TrajNet|50": {"ci_positive_and_easy_safe": True},
            "UCY|50": {"ci_positive_and_easy_safe": True},
        },
        "no_leakage": {
            "internal_val_from_train_only": True,
            "test_threshold_tuning": False,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "claim_boundary": {
            "floor_free_neural_deployable": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = bz._gate(payload)
    assert gate["verdict"] == "stage42_bz_t50_repair_statistical_evidence_pass"
    assert gate["passed"] == gate["total"]


def test_gate_blocks_missing_slice_ci() -> None:
    payload = {
        "source": "unit",
        "input_reports": {"stage42_by_verdict": "stage42_by_t50_floor_relaxability_repair_pass"},
        "summary": {
            "target_union_ci_positive_and_easy_safe": True,
            "bootstrap_n": bz.BOOTSTRAP_N,
            "teacher_floor_context_required": True,
            "floor_free_neural_deployable": False,
        },
        "slice_evidence": {
            "TrajNet|50": {"ci_positive_and_easy_safe": True},
            "UCY|50": {"ci_positive_and_easy_safe": False},
        },
        "no_leakage": {
            "internal_val_from_train_only": True,
            "test_threshold_tuning": False,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "claim_boundary": {
            "floor_free_neural_deployable": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = bz._gate(payload)
    assert gate["verdict"] == "stage42_bz_t50_repair_statistical_evidence_partial"
    assert not gate["gates"]["ucy_t50_bootstrap_positive"]
