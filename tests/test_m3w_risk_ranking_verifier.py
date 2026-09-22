import numpy as np
from scripts.verify_m3w_risk_ranking import manual_rank
from src.evaluation.m3w_risk_ranking import fixed_count


def test_separate_direct_ratio_rank_matches_fraction_and_gain():
    rng = np.random.default_rng(801)
    p = rng.uniform(0, 100, (1000, 2)); p[:4] = [[0, 0], [0, 2], [3, 0], [3, 0]]
    ids = rng.permutation(1000); e = rng.uniform(size=1000) > .15
    pool = list(np.flatnonzero(e))
    for n in (0, 100, len(pool)):
        for rule in ('ratio', 'gain'):
            np.testing.assert_array_equal(manual_rank(p, pool, ids, n, rule == 'ratio'), fixed_count(p, e, ids, n, rule))
