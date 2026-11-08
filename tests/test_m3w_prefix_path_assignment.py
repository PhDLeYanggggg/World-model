from itertools import permutations
import numpy as np

from scripts.audit_m3w_prefix_path_assignment import prior, shuffled


def test_exact_expectation_matches_exhaustive_fixed_action_permutation():
    rng = np.random.default_rng(17)
    cv, y = rng.normal(size=(4, 2, 2)), rng.normal(size=(4, 2, 2))
    y[0, 1] = np.nan
    valid = np.isfinite(y).all(2)
    bank = rng.normal(size=(4, 3, 2, 2))
    choice = [0, 1, 1, 2]
    p = bank[np.arange(4), choice]
    expected = []
    for ix in permutations(choice):
        other = bank[np.arange(4), ix]
        known, lower = prior.paired(other, p, y, valid)
        expected.append([prior.paired(cv, p, y, valid)[1].sum(), prior.paired(cv, other, y, valid)[1].sum(),
                         known.sum(), lower.sum()])
    np.testing.assert_allclose(shuffled(cv, bank, choice, y, valid), np.mean(expected, axis=0))
