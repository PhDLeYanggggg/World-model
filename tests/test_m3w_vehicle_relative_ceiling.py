import numpy as np
import pytest

from src.evaluation.m3w_interaction_frontier import pair_reduction
from src.evaluation.m3w_vehicle_relative_ceiling import relative_lower


@pytest.mark.parametrize('case', ['complete', 'mixed', 'all_missing'])
def test_fast_relative_lower_equals_brute_force_with_fixed_edits(case):
    rng = np.random.default_rng(13)
    cv = rng.normal(size=(6, 12, 2))
    ref = cv.copy()
    ref[[2, 4]] += rng.normal(size=(2, 12, 2))
    y = rng.normal(size=cv.shape)
    valid = np.ones(cv.shape[:2], bool)
    if case == 'mixed':
        valid[2, 4:] = False
        valid[3] = False
    elif case == 'all_missing':
        valid[:] = False
    y[~valid] = np.nan
    rows = np.repeat(np.arange(len(cv)), 3)
    paths = cv[rows]+rng.normal(size=(len(rows), 12, 2))
    a, b = np.triu_indices(len(rows), 1)
    i = rows[a] != rows[b]
    a, b = a[i], b[i]
    out = relative_lower(cv, ref, y, valid, rows, paths, a, b, chunk=7)
    p = np.broadcast_to(cv, (len(a), *cv.shape)).copy()
    p[np.arange(len(a)), rows[a]] = paths[a]
    p[np.arange(len(a)), rows[b]] = paths[b]
    np.testing.assert_allclose(out, pair_reduction(p, ref, y, valid)[:, 1], atol=1e-12)


def test_one_actor_cannot_be_edited_twice():
    cv = np.zeros((2, 12, 2))
    with pytest.raises(ValueError, match='different actors'):
        relative_lower(cv, cv, cv, np.ones((2, 12), bool), np.array([0, 0]), cv,
                       np.array([0]), np.array([1]))
