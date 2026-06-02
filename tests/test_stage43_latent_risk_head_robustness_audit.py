from __future__ import annotations

import numpy as np

from src import stage43_latent_risk_head_robustness_audit as bx


def test_bootstrap_head_returns_defined_ci_for_two_class_scores() -> None:
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float32)
    score = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=np.float32)
    out = bx._bootstrap_head(y, score, n=20, sample_rows=6, seed=1)
    assert out["defined_replicates"] > 0
    assert out["auroc"]["mean"] > 0.8
    assert out["sample_rows"] == 6


def test_breakdown_skips_small_slices_and_reports_defined_metrics() -> None:
    values = np.asarray(["A", "A", "A", "B"])
    y = np.asarray([0, 1, 1, 0], dtype=np.float32)
    score = np.asarray([0.2, 0.8, 0.7, 0.1], dtype=np.float32)
    out = bx._breakdown(values, y, score, min_rows=2)
    assert "A" in out
    assert "B" not in out
    assert out["A"]["defined"] is True


def _payload(*, weak: bool) -> dict:
    good_head = {
        "rows": 100,
        "positives": 50,
        "negatives": 50,
        "positive_rate": 0.5,
        "auroc": 0.9,
        "auprc": 0.9,
        "brier": 0.1,
        "ece": 0.02,
        "defined": True,
    }
    horizon = {
        "10": {**good_head, "auroc": 0.9},
        "100": {**good_head, "auroc": 0.65 if weak else 0.8},
    }
    boot = {
        head: {
            "defined_replicates": 500,
            "auroc": {"low": 0.7, "mean": 0.8, "high": 0.9},
            "auprc": {"low": 0.7, "mean": 0.8, "high": 0.9},
            "brier": {"low": 0.1, "mean": 0.1, "high": 0.2},
        }
        for head in bx.HEADS
    }
    return {
        "stage43_m_precondition": {"checkpoint_sha256_matches_stage43_m": True},
        "stage43_y_precondition": {"verdict": "stage43_y_protected_multimodal_latent_head_suite_candidate"},
        "evaluation_protocol": {"rows": 100},
        "latent_stats": {"min_variance": 0.1, "noncollapse_threshold": 0.01},
        "head_metrics": {head: dict(good_head) for head in bx.HEADS},
        "by_domain": {head: {"A": dict(good_head)} for head in bx.HEADS},
        "by_horizon": {head: dict(horizon) for head in bx.HEADS},
        "bootstrap": boot,
        "weak_horizon_slices": ([{"head": "failure", "horizon": "100"}] if weak else []),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }


def test_gate_reports_horizon_caveat_when_weak_horizon_slices_exist() -> None:
    weak_gate = bx._gate(_payload(weak=True))
    assert weak_gate["verdict"] == "stage43_bx_latent_risk_head_robustness_pass_horizon_caveat"
    assert weak_gate["weak_horizon_slice_count"] == 1

    clean_gate = bx._gate(_payload(weak=False))
    assert clean_gate["verdict"] == "stage43_bx_latent_risk_head_robustness_pass"
