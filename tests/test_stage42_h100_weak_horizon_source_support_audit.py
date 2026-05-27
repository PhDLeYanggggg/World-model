from __future__ import annotations

import numpy as np

from src import stage42_h100_weak_horizon_source_support_audit as fp


def test_family_name_groups_parent_source() -> None:
    assert fp._family_name("TrajNet/Test/crowds/students002.txt") == "Test/crowds"
    assert fp._family_name("UCY/zara03/crowds_zara03.txt") == "UCY/zara03"


def test_support_summary_reports_shared_family() -> None:
    data = {
        "dataset": np.asarray(["UCY", "UCY", "UCY", "UCY"], dtype=object),
        "horizon": np.asarray([100, 100, 100, 100]),
        "source_file": np.asarray(
            [
                "UCY/zara03/train.txt",
                "UCY/zara03/val.txt",
                "UCY/zara03/test.txt",
                "UCY/zara02/test.txt",
            ],
            dtype=object,
        ),
        "scene_id": np.asarray(["zara03", "zara03", "zara03", "zara02"], dtype=object),
    }
    val_ids = np.asarray([0, 1])
    test_ids = np.asarray([2, 3])

    row = fp._support_summary(data, val_ids, test_ids, "UCY|100")

    assert row["val_rows"] == 2
    assert row["test_rows"] == 2
    assert row["shared_family_count"] >= 1
    assert row["test_source_count"] == 2


def test_classify_blocker_detects_h100_support_and_margin() -> None:
    support = {
        "val_rows": 50,
        "val_source_count": 1,
        "shared_family_count": 0,
    }
    oracle = {
        "oracle_improvement_vs_fh": 0.01,
        "low_margin_share": {"0.05": 0.95},
    }
    source_rows = [
        {
            "robust_positive": False,
            "weak_reasons": ["easy_ci_exceeds_2pct"],
        }
    ]
    reasons = fp._classify_blocker("UCY|100", support, oracle, source_rows, {"switch_rows": 0})

    assert "oracle_low_margin_ambiguous" in reasons
    assert "low_material_headroom" in reasons
    assert "validation_rows_insufficient" in reasons
    assert "validation_to_test_source_family_shift" in reasons
    assert "source_specific_easy_safety_ci_failure" in reasons
    assert "gain_harm_policy_abstained_due_to_validation_safety" in reasons
    assert "long_horizon_h100_context_still_insufficient" in reasons


def test_gate_passes_diagnostic_with_h100_blocker() -> None:
    payload = {
        "source": fp.SOURCE,
        "summary": {
            "input_fo_verdict": "stage42_fo_gain_harm_specialist_pass_with_horizon_limit",
            "h100_weak_horizon_count": 2,
            "blocker_counts": {"oracle_low_margin_ambiguous": 2},
        },
        "audits": {
            "TrajNet|100": {
                "support": {},
                "source_rows": [{"name": "s"}],
                "scene_rows": [],
                "test_oracle": {"low_margin_share": {"0.05": 0.99}},
                "next_action": "add_support",
            }
        },
        "selection_rule": {
            "diagnostic_only": True,
            "does_not_train_new_policy": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "diagnostic_oracle_not_deployed": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "uniform_horizon_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fp._gate(payload)

    assert gate["verdict"] == "stage42_fp_h100_source_support_audit_pass"
