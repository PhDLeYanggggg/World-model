from __future__ import annotations

import numpy as np

from src import stage43_t100_policy_distilled_admissibility_head as dc
from src import stage43_t100_residual_admissibility_head as ct


class _TinySplit:
    def __init__(self) -> None:
        self.x = np.zeros((3, 2), dtype=np.float32)
        self.easy = np.asarray([False, True, False])


def test_teacher_switch_labels_only_selected_alpha_and_easy_guard() -> None:
    n = 3
    alpha_count = len(ct.ALPHAS)
    gain = np.zeros(n * alpha_count, dtype=np.float32)
    harm = np.ones(n * alpha_count, dtype=np.float32)
    delta = np.ones(n * alpha_count, dtype=np.float32)
    ai = 2
    sl = slice(ai * n, (ai + 1) * n)
    gain[sl] = np.asarray([0.9, 0.9, 0.2])
    harm[sl] = np.asarray([0.1, 0.1, 0.1])
    delta[sl] = np.asarray([-0.01, -0.01, -0.01])
    labels = dc._teacher_switch_labels(
        _TinySplit(),
        {"gain": gain, "harm": harm, "delta": delta},
        {
            "alpha_index": ai,
            "gain_threshold": 0.8,
            "harm_threshold": 0.2,
            "delta_threshold": 0.0,
            "force_easy_floor": True,
        },
    )
    assert labels.sum() == 1.0
    assert labels[ai * n] == 1.0
    assert labels[ai * n + 1] == 0.0
    assert labels[: ai * n].sum() == 0.0


def test_aggregate_compares_against_da_and_cz_references() -> None:
    seed_runs = [
        {
            "test_metrics_with_floor": {
                "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.002,
                "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.002,
                "easy_degradation_vs_floor": 0.0,
                "switch_rate": 0.1,
            },
            "test_group_summary": {"min_without_any_group_t100": 0.001},
            "teacher_switch_rate_train": 0.08,
            "bootstrap_ci": {
                "metrics": {
                    "t100_raw_frame_full_waypoint_diagnostic_vs_floor": {"low": 0.001}
                }
            },
        }
        for _ in range(3)
    ]
    cz_report = {"aggregate": {"robust_t100": {"mean": 0.003}, "robust_min_without_group_t100": {"mean": 0.002}}}
    da_report = {"aggregate": {"t100": {"mean": 0.001}, "min_without_group_t100": {"mean": -0.001}}}
    agg = dc._aggregate(seed_runs, cz_report, da_report)
    assert agg["beats_da_t100_mean"] is True
    assert agg["beats_cz_t100_mean"] is False
    assert agg["beats_da_min_without_group_mean"] is True


def test_gate_reports_positive_but_not_cz_when_only_da_improves() -> None:
    payload = {
        "stage43_db_precondition": {"verdict": "stage43_db_t100_head_failure_forensics_complete_policy_distill_next"},
        "result_source": "fresh_torch_policy_distilled_t100_admissibility_head",
        "seed_runs": [{"checkpoint": __file__, "checkpoint_committed": False, "test_metrics_with_floor": {"rows": 1}} for _ in range(3)],
        "training_protocol": {"teacher": "stage43_cz_leave_group_out_policy"},
        "selection_protocol": {"test_threshold_tuning": False},
        "feature_contract": {"denied_feature_name_hits": []},
        "aggregate": {
            "all_seed_easy_safe": True,
            "all_seed_t100_positive": True,
            "beats_da_t100_mean": True,
            "beats_cz_t100_mean": False,
        },
        "deploy_on_current_heldout": False,
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
        "long_objective_complete": False,
    }
    gate = dc._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_dc_t100_policy_distilled_head_improves_da_not_cz"
