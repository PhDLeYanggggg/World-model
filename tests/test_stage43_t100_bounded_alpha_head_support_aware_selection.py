from __future__ import annotations

import argparse

from src import stage43_t100_bounded_alpha_head_support_aware_selection as dh
from src.stage14_pipeline import read_json


def test_stage43_dh_support_aware_selection_repairs_df_group_fragility() -> None:
    payload = dh.run_t100_bounded_alpha_head_support_aware_selection(
        argparse.Namespace(
            quick=False,
            small=True,
            max_train=None,
            max_val=None,
            max_test=None,
            batch_size=2048,
            top_k=5,
            seeds="4323,4331,4337",
            epochs=5,
            hidden_dim=96,
            lr=1.3e-3,
            bootstrap=2000,
            alpha_cap=0.75,
            min_label_rows=80,
            min_val_improvement=0.0002,
        )
    )
    gate = payload["stage43_dh_gate"]
    aggregate = payload["aggregate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_dh_t100_support_aware_selection_repairs_df_group_fragility_diagnostic"
    assert aggregate["all_t100_positive"] is True
    assert aggregate["all_min_without_group_positive"] is True
    assert aggregate["easy_safe"] is True
    assert payload["deploy_on_current_heldout"] is False


def test_stage43_dh_report_records_claim_boundaries() -> None:
    payload = read_json(dh.REPORT_JSON, {})
    assert payload["selection_protocol"]["test_threshold_tuning"] is False
    assert payload["selection_protocol"]["test_oracle_used_for_selection"] is False
    assert payload["no_leakage"]["future_waypoint_input"] is False
    assert payload["no_leakage"]["future_waypoint_label_eval_only"] is True
    assert payload["claim_boundary"]["metric_or_seconds_claim"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
