from itertools import product

import numpy as np
import pytest

from src.evaluation.m3w_fixed_spend_uniform import prefix_space
from src.evaluation.m3w_streaming_menu_capacity import respects_tokens


@pytest.mark.parametrize("total", range(5))
def test_exact_prefix_ranks_match_exhaustive_plans(total):
    legal = np.ones((12, 3), dtype=bool)
    legal[[1, 6], 1:] = False
    legal[[0, 7], 2] = False
    fits = np.ones_like(legal)
    fits[[3, 8], 1:] = False
    fits[9, 2] = False
    space = prefix_space(legal, total, fits)
    expected = {n for n in product(*[np.flatnonzero(row) for row in legal & fits])
                if sum(n) == total and respects_tokens(n, legal[:, 1])}
    found = set()
    for rank in range(space.size):
        q, k = space.unrank(rank)
        n = np.zeros(len(legal), dtype=int)
        n[q] = k
        found.add(tuple(n))
    assert expected == found and len(found) == space.size
    with pytest.raises(ValueError):
        space.unrank(space.size)


def test_unpayable_early_edits_still_earn_credits():
    legal = np.ones((10, 3), dtype=bool)
    fits = legal.copy()
    fits[:-1, 1:] = False
    space = prefix_space(legal, 2, fits)
    assert space.size == 1
    assert tuple(x.tolist() for x in space.unrank(0)) == ([9], [2])
