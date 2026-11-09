import itertools

import numpy as np
import torch

from src.evaluation.m3w_cv_anchor_budget import candidate_terms
from src.world_model import m3w_cv_selected_harm as model
from src.world_model.m3w_prefix_expert_mixture import full_masses


def test_fixed_paths_preserve_nominations_and_match_query_counts():
    gain = np.repeat(np.array([[1., 2.], [2., 1.], [3., 4.], [4., 3.], [5., 0.]])[..., None], 2, axis=-1)
    risk = np.zeros((5, 2, 3))
    legal = np.array([[True, True], [True, True], [True, True], [True, True], [False, False]])
    anchor = np.array([0, -1, 0, -1, -1])
    choice = model.proposals(gain, risk, legal, anchor)
    np.testing.assert_array_equal(choice, [0, 0, 0, 0, -1])
    groups = [np.array([0, 1]), np.array([2, 3, 4])]
    score = np.array([1., 2., -2., -1., 9.])
    np.testing.assert_array_equal(model.select(choice, anchor, score, groups), [-1, 0, -1, 0, -1])
    np.testing.assert_array_equal(model.select(choice, anchor, score, groups, margin=0.), [-1, 0, -1, -1, -1])


def test_upper_matches_exhaustive_paired_missing_step_score():
    rng = np.random.default_rng(42)
    cv, ols, path, y = rng.normal(size=(4, 6, 12, 2))
    valid = rng.random((6, 12)) > .4
    valid[0] = False
    y[~valid] = np.nan
    groups = [np.arange(3), np.arange(3, 6)]
    anchor = np.array([-1, 0, -1, 0, 0, -1])
    choice = np.array([-1, 0, 0, 0, 0, 0])
    w = np.arange(1, 7) / 10
    _, offset, g, _ = candidate_terms(cv, ols, path[:, None], y, valid)
    upper = model.upper_at_count(g[:, 0], choice, anchor, groups, w) + [0., w @ offset]
    scores = []
    for a, b in itertools.product((1, 2), itertools.combinations(range(3, 6), 2)):
        selected = [a, *b]
        out = cv.copy()
        out[selected] = path[selected]
        pair = [full_masses(p, out, cv, y, valid, w) for p in (cv, ols)]
        scores.append([x['known_delta'] - x['unknown_radius'] for x in pair])
    np.testing.assert_allclose(upper, np.max(scores, axis=0))


def test_wholly_unknown_uses_peak_debit_but_no_signed_easy_label():
    cv = np.zeros((3, 12, 2))
    y = np.ones_like(cv)
    v = np.ones((3, 12), bool)
    v[0] = False
    y[0] = np.nan
    p = cv.copy()
    p[0, 0, 0] = 5.
    p[1] = 1.
    p[2] = -1.
    out = model.targets(cv, p, y, v, np.ones(3), 2.)
    np.testing.assert_allclose(out, [[5., 5., 0.], [0., 0., -np.sqrt(2)],
        [np.sqrt(2), np.sqrt(2), np.sqrt(2)]], atol=1e-6)


def test_tail_loss_penalizes_underestimate_more_than_median():
    net = torch.nn.Identity()
    x = torch.zeros((1, 3))
    y = model.decode(x)
    assert model.loss(net, x, y + 1., .85) > model.loss(net, x, y + 1., .5)
    assert model.loss(net, x, y - 1., .85) < model.loss(net, x, y - 1., .5)


def test_expanded_menu_and_free_counts_have_nested_exact_bounds():
    from scripts.audit_m3w_cv_selected_harm_capacity import maxima

    cv = np.zeros((3, 12, 2))
    ols = np.ones_like(cv)
    y = 2. * ols
    valid = np.ones((3, 12), bool)
    q = np.zeros(3, int)
    one = .05 * ols
    bank = np.stack((ols, y), axis=1)
    upper = maxima(cv, ols, bank, one, np.zeros(3, int), np.array([0, -1, -1]), y, valid, q, 1)
    np.testing.assert_allclose(upper[:, 1], [-49.1666666667, -33.3333333333, -16.6666666667,
        -16.6666666667, 50.])
