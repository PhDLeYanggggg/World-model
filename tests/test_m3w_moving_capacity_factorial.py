import itertools

import numpy as np

from src.world_model import m3w_moving_capacity_factorial as m


def test_exact_budget_matches_exhaustive_and_requires_easy_anchor():
    g = np.array([8., 1., 2., 6., 5.])
    easy = np.array([0, 1, 1, 0, 0], bool)
    queries = [np.arange(3), np.arange(3, 5)]
    for cap in (1, 2, None):
        got = m.plan(g, easy, queries, cap)
        legal = (np.asarray(x, int) for n in range(1, 6) for x in itertools.combinations(range(5), n))
        best = max(g[x].sum() for x in legal if easy[x].any()
                   and (cap is None or all(np.isin(x, q).sum() <= cap for q in queries)))
        assert g[got].sum() == best
    assert not len(m.plan(g, np.zeros(5, bool), queries, 2))


def test_causal_extra_bank_respects_per_row_displacement_cap():
    rng = np.random.default_rng(12)
    h = rng.normal(size=(4, 8, 2))
    cv = rng.normal(size=(4, 12, 2))
    old = np.stack((cv, cv+.2), axis=1)
    p = m.extra_bank(h, cv, old, old[:, 0], old[:, 1])
    assert p.shape == (4, len(m.NAMES), 12, 2)
    assert (np.linalg.norm(p-cv[:, None], axis=-1) <= np.sqrt(.08)+1e-12).all()
