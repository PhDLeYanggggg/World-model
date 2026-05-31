from __future__ import annotations

import numpy as np

from src import stage43_world_state_head_audit as v


def test_binary_metrics_perfect_ranking() -> None:
    y = np.asarray([0, 0, 1, 1], dtype=np.float32)
    score = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    metrics = v._binary_metrics(y, score)
    assert metrics["defined"] is True
    assert metrics["auroc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["brier_improvement_vs_prevalence"] > 0.0


def test_binary_metrics_handles_single_class() -> None:
    y = np.zeros(5, dtype=np.float32)
    score = np.linspace(0.1, 0.9, 5, dtype=np.float32)
    metrics = v._binary_metrics(y, score)
    assert metrics["defined"] is False
    assert metrics["auroc"] is None
    assert metrics["auprc"] is None


def test_regression_metrics_detects_mean_baseline_improvement() -> None:
    y = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    pred = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    metrics = v._regression_metrics(y, pred)
    assert metrics["r2"] == 1.0
    assert metrics["rmse"] == 0.0


def test_gate_marks_physical_validity_non_deployable() -> None:
    payload = {
        "source": v.SOURCE,
        "stage43_m_precondition": {"checkpoint_sha256_matches_stage43_m": True},
        "head_metrics": {
            "failure": {"auroc": 0.7},
            "gain": {"auroc": 0.55},
            "harm": {"auroc": 0.58},
            "density": {"rows": 10},
            "physical_validity_proxy": {"trained_with_explicit_loss": False, "deployment_allowed": False},
        },
        "latent_stats": {"mean_variance": 0.03, "noncollapse_threshold": 0.01},
        "evaluation_protocol": {"test_threshold_tuning": False},
        "no_leakage": {
            "test_threshold_tuning": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = v._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["informative_binary_heads"] == ["failure"]
    assert gate["deploy_physical_validity_head"] is False
