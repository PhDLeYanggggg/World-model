import numpy as np

from scripts import audit_m3w_ols_anchor_menu as study
from scripts.audit_m3w_ols_anchor_budget_location import upper
from src.world_model.m3w_ols_anchor_paths import NAMES, candidates


def test_paths_are_equivariant_and_exact_on_linear_history():
    t = np.arange(8.)
    h = np.stack((t, -2*t), axis=1)[None]
    forecast = (np.arange(1., 13.)[:, None]*[1., -2.])[None, None]
    np.testing.assert_allclose(candidates(h), np.broadcast_to(forecast, (1, len(NAMES), 12, 2)))
    rng = np.random.default_rng(7)
    noisy = rng.normal(size=(5, 8, 2))
    rot = np.array([[0., -1.], [1., 0.]])
    np.testing.assert_allclose(candidates(noisy@rot+13.), candidates(noisy)@rot, atol=1e-12)


def test_full_recording_ols_debit_and_exact_query_count():
    cv = np.ones((4, 12, 2))
    cv[..., 1] = 0.
    ols = .9*cv
    y = np.zeros_like(cv)
    valid = np.ones((4, 12), bool)
    q = np.array([1, 1, 2, 2])
    active, anchor = np.array([0, 1]), np.array([0, -1, -1, -1])
    args = (y, valid, q, active, anchor, (1., 1.1))
    base = .9*cv[active, None]
    r = study.search(cv, ols, base, *args)
    assert r['status'] == 'nonpositive_bound'
    menu = np.concatenate((base, .4*cv[active, None]), axis=1)
    r = study.search(cv, ols, menu, *args)
    assert r['status'] == 'verified'
    assert len(r['actions']) == 1
    np.testing.assert_allclose(r['receipt']['gain_pp'], [15., 5.])


def test_reallocating_fixed_record_count_does_not_relax_the_count():
    g = np.array([[1., -1.], [2., -2.], [8., 10.], [7., 12.]])
    q = np.array([1, 1, 2, 2])
    selected = np.array([True, False, False, False])
    np.testing.assert_array_equal(upper(g, q, selected, 'query'), [2., -1.])
    np.testing.assert_array_equal(upper(g, q, selected, 'record'), [8., 12.])
