import numpy as np

from src import stage42_source_level_residual_context as s42ap


def test_residual_variant_masks_include_context_and_controls():
    names = [
        "history_scalar_0",
        "history_scalar_1",
        "history_tail_0",
        "prototype_0",
        "prototype_entropy",
        "goal_ambiguity",
        "safe_baseline_rel_0",
        "family_baseline_rel_0",
        "floor_rel_x",
        "domain_UCY",
        "horizon_50",
    ]
    masks = s42ap._residual_variant_masks(names)
    assert masks["residual_history"][0]
    assert masks["residual_history"][9]
    assert masks["residual_history"][10]
    assert masks["residual_goal"][3]
    assert masks["residual_neighbor"][1]
    assert masks["residual_history_goal_neighbor"][0]
    assert masks["residual_history_goal_neighbor"][3]
    assert masks["residual_history_goal_neighbor"][1]


def test_baseline_mask_uses_rollout_and_controls():
    names = [
        "history_scalar_0",
        "safe_baseline_rel_0",
        "family_baseline_rel_0",
        "floor_rel_x",
        "domain_UCY",
        "horizon_50",
    ]
    mask = s42ap._baseline_mask(names)
    assert not mask[0]
    assert mask[1]
    assert mask[2]
    assert mask[3]
    assert mask[4]
    assert mask[5]


def test_predict_residual_xy_adds_scaled_delta():
    base = np.zeros((2, 4, 2), dtype=np.float32)
    x = np.ones((2, 3), dtype=np.float32)
    coef = np.zeros((3, 8), dtype=np.float32)
    coef[:, 0] = 1.0
    data = {"scale": np.asarray([2.0, 3.0], dtype=np.float32)}
    out = s42ap._predict_residual_xy(base, x, coef, data, residual_alpha=0.5)
    assert out.shape == base.shape
    assert np.isclose(out[0, 0, 0], 3.0)
    assert np.isclose(out[1, 0, 0], 4.5)


def test_gate_partial_when_no_residual_increment():
    metric = {"all_improvement": 0.2, "t50_improvement": 0.1, "hard_failure_improvement": 0.15, "easy_degradation": 0.0}
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "baseline_family_only": {
            "protected": metric,
            "bootstrap": {"all": {"bootstrap_n": 1000}, "t50": {"bootstrap_n": 1000}},
        },
        "residual_variants": {f"v{i}": {} for i in range(7)},
        "positive_residual_context_variants": [],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42ap._gate(result)
    assert gate["passed"] == gate["total"] - 1
    assert gate["gates"]["residual_context_increment_found"] is False
