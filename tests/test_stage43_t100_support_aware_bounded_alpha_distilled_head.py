from __future__ import annotations

import argparse

from src import stage43_t100_support_aware_bounded_alpha_distilled_head as di
from src.stage14_pipeline import read_json


def test_stage43_di_support_aware_head_training_runs_with_safe_boundaries() -> None:
    payload = di.train_t100_support_aware_bounded_alpha_distilled_head(
        argparse.Namespace(
            quick=False,
            small=True,
            seeds="4323,4331,4337",
            max_train=None,
            max_val=None,
            max_test=None,
            epochs=5,
            batch_size=2048,
            hidden_dim=96,
            lr=1.1e-3,
            bootstrap=2000,
            alpha_cap=0.75,
            top_k=8,
            min_label_rows=80,
            min_val_improvement=0.0002,
        )
    )
    gate = payload["stage43_di_gate"]
    assert gate["passed"] == gate["total"]
    assert payload["training_protocol"]["teacher"] == "stage43_dh_support_aware_policy"
    assert payload["selection_protocol"]["test_threshold_tuning"] is False
    assert all(run["checkpoint_committed"] is False for run in payload["seed_runs"])
    assert payload["deploy_on_current_heldout"] is False


def test_stage43_di_report_keeps_no_leakage_and_claim_boundary() -> None:
    payload = read_json(di.REPORT_JSON, {})
    assert payload["no_leakage"]["future_endpoint_input"] is False
    assert payload["no_leakage"]["future_waypoint_input"] is False
    assert payload["no_leakage"]["future_waypoint_label_eval_only"] is True
    assert payload["no_leakage"]["test_statistics_normalization"] is False
    assert payload["claim_boundary"]["metric_or_seconds_claim"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
