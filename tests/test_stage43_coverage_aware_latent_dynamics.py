from __future__ import annotations

from pathlib import Path

from src import stage43_coverage_aware_latent_dynamics as cg
from src import stage43_full_waypoint_latent_dynamics as base


def _metrics(*, all_: float, t50: float, hard: float, easy: float) -> dict:
    return {
        "rows": 100,
        "full_waypoint_ade_improvement_vs_floor": all_,
        "endpoint_fde_improvement_vs_floor": all_,
        "t50_full_waypoint_ade_improvement_vs_floor": t50,
        "t50_endpoint_fde_improvement_vs_floor": t50,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": hard,
        "easy_degradation_vs_floor": easy,
        "switch_rate": 0.2,
        "harm_over_floor_ade": -0.01,
        "mean_floor_ade": 1.0,
        "mean_selected_ade": 0.95,
    }


def _payload(metrics: dict, *, deploy: bool) -> dict:
    return {
        "source": cg.SOURCE,
        "result_source": "fresh_run",
        "checkpoint": __file__,
        "checkpoint_committed": False,
        "stage43_l_precondition": {
            "verdict": "stage43_cf_coverage_aware_full_waypoint_cache_ready",
            "full_waypoint_supervised_training_ready": True,
        },
        "data_rows": {"train": 10, "val": 5, "test": 5},
        "latent_variance": 0.03,
        "validation_selected_policy": {"policy": {"gain_threshold": 0.5}},
        "test_metrics_with_floor": metrics,
        "test_metrics_neural_without_floor": dict(metrics),
        "deploy_neural": deploy,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_configure_base_points_to_coverage_aware_cache() -> None:
    old = {
        "cache_dir": base.CACHE_DIR,
        "cache_prefix": base.CACHE_FILE_PREFIX,
        "checkpoint": base.CHECKPOINT_NAME,
        "precondition": base.PRECONDITION_JSON,
    }
    try:
        cg._configure_base()
        assert base.CACHE_DIR == cg.CACHE_DIR
        assert base.CACHE_FILE_PREFIX == "stage43_ce_full_waypoint_supervision"
        assert base.CHECKPOINT_NAME == "stage43_coverage_aware_latent_dynamics.pt"
        assert base.PRECONDITION_JSON == cg.CF_JSON
        assert base._cache_path("train") == Path("data/stage43_ce_full_waypoint_supervision_cache/stage43_ce_full_waypoint_supervision_train.npz")
    finally:
        base.CACHE_DIR = old["cache_dir"]
        base.CACHE_FILE_PREFIX = old["cache_prefix"]
        base.CHECKPOINT_NAME = old["checkpoint"]
        base.PRECONDITION_JSON = old["precondition"]


def test_gate_passes_candidate_when_safe_lift_deploys() -> None:
    payload = _payload(_metrics(all_=0.02, t50=0.03, hard=0.04, easy=0.01), deploy=True)
    gate = cg._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["deploy_coverage_aware_latent_dynamics"] is True
    assert gate["verdict"] == "stage43_cg_coverage_aware_latent_dynamics_candidate_pass"


def test_gate_keeps_floor_for_honest_no_lift_diagnostic() -> None:
    payload = _payload(_metrics(all_=0.0, t50=0.0, hard=0.0, easy=0.0), deploy=False)
    gate = cg._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["deploy_coverage_aware_latent_dynamics"] is False
    assert gate["verdict"] == "stage43_cg_coverage_aware_latent_dynamics_diagnostic_keep_floor"


def test_gate_blocks_easy_harm() -> None:
    payload = _payload(_metrics(all_=0.05, t50=0.05, hard=0.05, easy=0.03), deploy=True)
    gate = cg._gate(payload)
    assert gate["gates"]["easy_preserved"] is False
    assert gate["deploy_coverage_aware_latent_dynamics"] is False
