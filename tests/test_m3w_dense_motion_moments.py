import numpy as np
import torch

from scripts.audit_m3w_dense_predicted_feasibility import feasibility
from scripts.run_m3w_dense_safe_union import choose_union
from src.world_model.m3w_dense_motion_moments import (
    DenseMoments, choose, features, matched_hindcast, targets,
)


def test_translation_leaves_past_features_unchanged_and_costs_bounded():
    h = np.arange(16, dtype=float).reshape(1, 8, 2)
    x, scale, bound, _ = features(h)
    assert x.shape == (1, 22)
    np.testing.assert_array_equal(x, features(h+100)[0])
    r, m = DenseMoments()(torch.from_numpy(x), torch.tensor(bound,dtype=torch.float32))
    assert r[0, 0] >= r[0, 1] >= 0
    assert not m.any()
    assert scale[0] > 0


def test_unobserved_targets_charge_the_same_disagreement_as_risk():
    cv = np.zeros((1, 12, 2))
    ols = cv+np.array([4., 0.])
    r, m = targets(cv, ols, np.full_like(cv, np.nan), np.zeros((1, 12), bool), np.ones(1), 10.)
    assert not r.any()
    np.testing.assert_array_equal(m[0], [[-1, 1, 1], [-2, 2, 2], [-3, 3, 3], [-4, 4, 4]])


def test_joint_ladder_checks_both_predicted_denominators():
    x = np.ones((2, 4, 3))
    x[..., 1:] = 0
    legal = np.array([True, True])
    offsets = np.array([0, 2])
    independent, joint, price = choose(np.array([[1, 0], [1, 0]]), x, offsets, legal)
    assert independent.tolist() == [1, 1] and not joint.any() and price is None
    _, joint, price = choose(np.ones((2, 2)), x, offsets, legal)
    assert joint.tolist() == [1, 1] and price == 0.


def test_matched_control_preserves_each_query_amplitude_multiset():
    actions = np.array([1, 0, 4, 2, 0], dtype=np.uint8)
    offsets = np.array([0, 3, 5])
    match = matched_hindcast(actions, np.array([0, 2, 1, 0, 2]), offsets, np.ones(5, bool))
    np.testing.assert_array_equal(match, [0, 4, 1, 0, 2])


def test_union_recovers_feasible_positive_actions_without_searching_prices():
    ref = np.array([[100., 10.], [20., 5.]])
    m = np.zeros((2, 4, 3))
    m[:, 0] = [[3., 1., .1], [2., .3, .05]]
    m[:, 3] = [[7., 5., 1.], [9., 8., 4.]]
    legal = np.ones(2, bool)
    assert feasibility(ref, m, legal)["positive_gain_singleton_actions"] == 2
    acts, ratios = choose_union(ref, m, legal)
    np.testing.assert_array_equal(acts, [1, 1])
    np.testing.assert_allclose(ratios, [1.3/120, .15/15])
