import numpy as np

from src import stage42_source_level_neural_context as s42aq


def test_context_variant_masks_cover_neural_context_groups():
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
    masks = s42aq._context_variant_masks(names)
    assert masks["neural_history"][0]
    assert masks["neural_history"][9]
    assert masks["neural_goal_neighbor"][3]
    assert masks["neural_goal_neighbor"][1]
    assert masks["neural_history_goal_neighbor"][0]
    assert masks["neural_history_goal_neighbor"][3]
    assert masks["neural_history_goal_neighbor"][1]
    assert not masks["neural_history"][6]


def test_standardize_context_uses_train_only_stats():
    x = np.asarray([[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]], dtype=np.float32)
    train = np.asarray([True, True, False])
    z, mean, std = s42aq._standardize_context(x, train)
    assert np.allclose(mean, [2.0, 3.0])
    assert np.allclose(std, [1.0, 1.0])
    assert np.allclose(z[0], [-1.0, -1.0])


def test_positive_neural_delta_threshold():
    d = {
        "all_improvement": 0.0,
        "t50_improvement": 0.011,
        "hard_failure_improvement": 0.0,
    }
    assert s42aq._positive_neural_delta(d)
    d["t50_improvement"] = 0.005
    assert not s42aq._positive_neural_delta(d)


def test_gate_requires_neural_increment():
    metric = {"all_improvement": 0.2, "t50_improvement": 0.1, "hard_failure_improvement": 0.15, "easy_degradation": 0.0}
    result = {
        "runtime": {"machine": "arm64", "num_workers": 0},
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "baseline_family_only": {
            "protected": metric,
            "bootstrap": {"all": {"bootstrap_n": 1000}, "t50": {"bootstrap_n": 1000}},
        },
        "neural_variants": {"a": {}, "b": {}, "c": {}},
        "positive_neural_context_variants": [],
        "training": {
            "a": {"checkpoint": "README_RESULTS.md"},
            "b": {"checkpoint": "README_RESULTS.md"},
            "c": {"checkpoint": "README_RESULTS.md"},
        },
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
    gate = s42aq._gate(result)
    assert gate["passed"] == gate["total"] - 1
    assert gate["gates"]["neural_context_increment_found"] is False
