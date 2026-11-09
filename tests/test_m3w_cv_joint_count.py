import numpy as np
import torch

from scripts import train_m3w_cv_joint_count as study
from src.world_model import m3w_cv_joint_count as model
from src.world_model.m3w_prefix_expert_mixture import full_masses


def test_menu_contains_all_levels_and_preserves_anchor_and_legal_paths():
    rng = np.random.default_rng(42)
    legal = np.ones((9, 4), bool)
    legal[0] = False
    legal[1, 1:] = False
    a = np.array([-1, 0, 2, -1, 0, -1, -1, -1, -1])
    z, independent = model.menu(rng.normal(size=(9, 4, 2)), rng.normal(size=(9, 4, 3)),
        legal, rng.normal(size=(9, 4)), a)
    assert set(independent) == {0, 2, 3, 4, 6, 8}
    assert (z == a).all(1).any()
    assert len({p.tobytes() for p in z}) == len(z)
    plan, row = np.nonzero(z >= 0)
    assert legal[row, z[plan, row]].all()
    for k, i in independent.items():
        assert (z[i] >= 0).sum() == k


def test_no_legal_action_and_exact_count_selection():
    g, r = np.zeros((2, 3, 2)), np.zeros((2, 3, 3))
    z, ind = model.menu(g, r, np.zeros((2, 3), bool), g[..., 0], np.full(2, -1))
    assert z.tolist() == [[-1, -1]]
    assert ind == {0: 0}
    p = np.zeros((3, 6))
    p[:, :2] = [[2, 2], [3, 3], [5, 5]]
    n = np.array([0, 1, 2])
    assert model.select(p, n, 2., .25, 0.) == 2
    assert model.select(p, n, 2., .25, 4.) == 0
    assert model.select(p, n, 2., .25, 4., match=1) == 1


def test_query_labels_agree_with_lower_gain_and_geometry():
    rng = np.random.default_rng(12)
    cv, ols, y, edited = rng.normal(size=(4, 3, 12, 2))
    v = rng.random((3, 12)) > .4
    y[~v] = np.nan
    h = rng.normal(size=(3, 8, 2))
    edges = study.prior.geometry.local_edges(h)
    fore = np.stack((cv, edited))
    score = study.truth(cv, ols, fore, y, v, 1., 2., edges)
    np.testing.assert_allclose(score[0], 0., atol=1e-7)
    w = np.full(len(y), 1. / len(y))
    a = full_masses(ols, cv, cv, y, v, w)
    b = full_masses(ols, edited, cv, y, v, w)
    expected = ((b['known_delta'] - b['unknown_radius']) - (a['known_delta'] - a['unknown_radius'])) / 12.
    np.testing.assert_allclose(score[1, 1], expected, atol=1e-7)
    _, u, _, _ = study.prior.geometry.query(h, fore, y, v, ((0, 1),))
    np.testing.assert_allclose(score[1, 5], u[0, 1], atol=1e-7)


def test_masked_scorer_ignores_neighbor_inputs():
    torch.manual_seed(13)
    x = torch.randn(4, 2, 9)
    y = torch.randn(4, 2, 6)
    net = torch.nn.Linear(9, 6)
    z = x.clone()
    z[..., -model.PAIR_FEATURES:] += 3.
    assert model.loss(net, x, y, False) == model.loss(net, z, y, False)
    assert model.loss(net, x, y, True) != model.loss(net, z, y, True)


def test_recording_relaxation_requires_support_and_both_comparators():
    from scripts.audit_m3w_cv_joint_count_capacity import search

    cv = [0., 0., 0., 0., 0., 0.]
    legal = [2., 1., -.1, -.1, -.1, 1.]
    block = np.array([cv, legal])
    s, selected = search([block], 100., 0., 0.)
    assert selected == [1]
    np.testing.assert_allclose(s['upper'], 1.)
    s, selected = search([block], 100., -1.5, 0.)
    assert s['status'] == 'excluded'
    assert selected is None
    s, selected = search([block[:1]], 100., 0., 0.)
    assert s['status'] == 'unsupported'
    assert selected is None


def test_original_harm_caps_exclude_positive_unconstrained_gain():
    from scripts.audit_m3w_cv_joint_count_capacity import search

    z = np.array([[0., 0., 0., 0., 0., 0.], [2., 2., .4, .4, .4, 1.]])
    s, selected = search([z], 100., 0., 0.)
    assert s['unconstrained'] > 0.
    assert s['status'] == 'lp_infeasible'
    assert selected is None
