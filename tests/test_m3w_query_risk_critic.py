import numpy as np

from src.world_model import m3w_query_risk_critic as model


def test_unknown_harm_contributes_to_both_caps_without_creating_support():
    floor = np.zeros((3, 12, 2))
    target = floor.copy()
    target[:, :, 0] = [np.full(12, 2), np.full(12, 8), np.full(12, 1)]
    pred = target.copy()
    valid = np.ones((3, 12), dtype=bool)
    valid[-1] = False
    a = model.action_targets(floor, pred, target, valid, 3.)
    np.testing.assert_allclose(model.targets(a, 2.), [4.5, .4, .48, 1.])
    np.testing.assert_array_equal(model.targets(a[1:], 2.)[-1], 0.)


def test_proposal_descriptor_invariant_to_row_order_and_fixed_for_single_switch():
    rng = np.random.default_rng(11)
    h, floor, bank = rng.normal(size=(5, 8, 2)), np.zeros((5, 12, 2)), rng.normal(size=(5, 3, 12, 2))
    choice = np.array([0, -1, 2, -1, -1])
    costs = tuple(rng.normal(size=(5, 3)) for _ in range(4))
    scale = np.arange(1., 6.)
    order = [2, 0, 3, 1, 4]
    compact, full, _ = model.proposal_features(h, floor, bank, choice, costs[:3], scale, costs[3])
    a, b, _ = model.proposal_features(h[order], floor[order], bank[order], choice[order],
        tuple(c[order] for c in costs[:3]), scale[order], costs[3][order])
    np.testing.assert_allclose(a, compact)
    np.testing.assert_allclose(b, full)
    choice[0] = -1
    assert model.proposal_features(h, floor, bank, choice, costs[:3], scale, costs[3])[1].shape == full.shape


def test_fit_weights_sources_equally_and_accept_requires_actual_positive_support():
    x = np.ones((4, 2))
    y = np.array([[2., -1., -1., 1.], [2., -1., -1., 1.], [2., -1., -1., 1.],
                  [4., -1., -1., 0.]])
    p = model.predict(x, model.fit(x, y, ['a', 'a', 'a', 'b']))
    np.testing.assert_allclose(p[0], [3., -1., -1., .5])
    np.testing.assert_array_equal(model.accept(y), [True, True, True, False])
