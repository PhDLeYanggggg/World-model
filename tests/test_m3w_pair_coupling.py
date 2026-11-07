from itertools import combinations

import numpy as np

from src.world_model import m3w_pair_coupling as m


def setup():
    rng = np.random.default_rng(17)
    h = rng.normal(size=(6, 8, 2))
    floor = rng.normal(size=(6, 12, 2))
    candidate = rng.normal(size=(6, 12, 2))
    ids = np.arange(6)[::-1]
    return m.geometry(h, floor, candidate, ids), ids


def test_mixed_difference_is_causal_even_with_missing_labels():
    g, ids = setup()
    rng = np.random.default_rng(31)
    i, j, _, _, di, dj, _ = g
    for _ in range(3):
        y = rng.normal(size=(len(ids), 12, 2)); valid = rng.random(y.shape[:2]) > .4
        r, mask = m.observed_residual(g, y, valid)
        sq = lambda x: (x**2).sum(-1)
        mixed = sq(r+di-dj)-sq(r+di)-sq(r-dj)+sq(r)
        np.testing.assert_allclose(mixed[mask], (-2*(di*dj).sum(-1))[mask], atol=1e-13)


def test_exact_same_count_decoder_matches_enumeration():
    g, ids = setup(); i, j = g[:2]
    u, interaction = m.costs(g, np.ones((12, 9)), len(ids))
    a, b = m.choose(u, i, j, interaction, ids)
    assert a.sum() == b.sum() == 2
    def loss(selected): return u[selected].sum()+interaction[selected[i]&selected[j]].sum()
    assert np.isclose(loss(b), min(loss(np.isin(np.arange(len(ids)), s)) for s in combinations(range(len(ids)), 2)))


def test_unobserved_coordinates_are_not_supervision_or_scored():
    g, ids = setup()
    y = np.zeros((len(ids), 12, 2)); valid = np.zeros(y.shape[:2], bool); valid[:, :6] = True
    other = y.copy(); other[~valid] = np.nan
    for a, b in zip(m.moments(g, y, valid), m.moments(g, other, valid)):
        np.testing.assert_array_equal(a, b)
    choice = np.arange(len(ids)) < 2
    assert m.error(g, y, valid, choice) == m.error(g, other, valid, choice)


def test_singleton_and_two_agent_counts():
    for n in (1, 2):
        a, b = m.choose(np.zeros(n), np.array([], int), np.array([], int), np.array([]), np.arange(n))
        assert a.sum() == b.sum() == n-1
        np.testing.assert_array_equal(a, b)


def test_held_source_targets_do_not_affect_its_fitted_model_or_choices():
    from scripts.run_m3w_pair_coupling import fit_predict

    rng = np.random.default_rng(44)
    n = 8; ids = np.arange(n); history = rng.normal(size=(n, 8, 2))
    paths = rng.normal(size=(n, 12, 2)); candidate = rng.normal(size=paths.shape)
    d = ids, np.repeat(['A', 'B'], 4), history, ids, paths, candidate, [ids[:4], ids[4:]]
    target = rng.normal(size=paths.shape); valid = np.ones(paths.shape[:2], bool)
    first = fit_predict(d, dict(target_eval=target, valid=valid))
    target[:4] += rng.normal(size=target[:4].shape)*100
    valid[:4, 3:] = False
    second = fit_predict(d, dict(target_eval=target, valid=valid))
    np.testing.assert_array_equal(first['coefficients'][0], second['coefficients'][0])
    np.testing.assert_array_equal(first['choices'][:4], second['choices'][:4])
    assert not np.array_equal(first['coefficients'][1], second['coefficients'][1])
