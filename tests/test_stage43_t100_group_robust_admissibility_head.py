from __future__ import annotations

import numpy as np

from src import stage43_t100_group_robust_admissibility_head as da


def test_balanced_group_weights_normalized() -> None:
    source = np.asarray(["a", "a", "a", "b"])
    scene = np.asarray(["s1", "s1", "s2", "s3"])
    domain = np.asarray(["d", "d", "d", "e"])
    weights = da._balanced_group_weights(source, scene, domain)
    assert weights.shape == (4,)
    assert 0.8 <= float(weights.mean()) <= 1.2
    assert weights[-1] > weights[0]


def test_repeat_for_alpha_order() -> None:
    values = np.asarray(["x", "y"])
    repeated = da._repeat_for_alpha(values, alpha_count=3)
    assert repeated.tolist() == ["x", "y", "x", "y", "x", "y"]


def test_gate_reports_positive_but_not_policy_best() -> None:
    payload = {
        "stage43_cz_precondition": {"verdict": "stage43_cz_t100_leave_group_out_policy_reduces_fragility_diagnostic"},
        "result_source": "fresh_torch_group_robust_t100_admissibility_head",
        "seed_runs": [{"checkpoint": __file__, "checkpoint_committed": False, "test_metrics_with_floor": {"rows": 1}} for _ in range(3)],
        "feature_contract": {"denied_feature_name_hits": []},
        "training_protocol": {"group_weighting": True},
        "selection_protocol": {"objective": "leave_group_out_min_t100_plus_flip_penalty"},
        "aggregate": {
            "all_seed_t100_positive": True,
            "all_seed_easy_safe": True,
            "beats_cz_t100_mean": False,
            "min_without_group_t100": {"mean": -0.001},
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
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "long_objective_complete": False,
    }
    gate = da._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_da_t100_group_robust_head_positive_but_not_policy_best"
