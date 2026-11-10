import numpy as np

from src.evaluation.m3w_two_factor_bounds import contrast


def test_crossed_matching_removes_false_uncertainty_on_identical_pairs():
    a = np.zeros((1, 2, 2))
    b = a+1
    truth = np.full_like(a, np.nan)
    out = contrast(a, b, b, a, truth, np.zeros((1, 2), dtype=bool), np.ones(1), 100.)
    for v in out.values():
        assert v['lower_pp'] == v['upper_pp'] == 0.
    assert out['path']['separate_pair_radius_pp'] > 0.
    assert out['recipient']['separate_pair_radius_pp'] > 0.


def test_masked_factorial_contains_every_tested_common_completion():
    rng = np.random.default_rng(43)
    p = rng.normal(size=(4, 5, 12, 2))
    y = rng.normal(size=p.shape[1:])
    m = rng.random(y.shape[:2]) > .5
    w = rng.random(len(y))+.1
    bound = contrast(*p, np.where(m[..., None], y, np.nan), m, w, 17.)
    for _ in range(30):
        truth = np.where(m[..., None], y, rng.normal(size=y.shape)*10.)
        v = contrast(*p, truth, np.ones_like(m), w, 17.)
        np.testing.assert_allclose(v['total']['observed_pp'], v['path']['observed_pp']+v['recipient']['observed_pp'])
        for k, x in v.items():
            assert bound[k]['lower_pp'] <= x['observed_pp'] <= bound[k]['upper_pp']
            assert bound[k]['missing_radius_pp'] <= bound[k]['separate_pair_radius_pp']
