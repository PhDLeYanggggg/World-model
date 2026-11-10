import numpy as np

from src.evaluation.m3w_coherent_pair_bounds import compare
from src.evaluation.m3w_sparse_pair_bounds import compare as pairwise


def test_a_shared_missing_position_cancels_reordered_anchors():
    # Against the missing actor, the two possible anchor points swap.
    y = np.array([[[np.nan, np.nan]], [[0., 0.]], [[1., 0.]]])
    valid = np.array([[False], [True], [True]])
    ref = np.zeros((3, 1, 2))
    pred = ref.copy()
    pred[1, 0, 0], pred[2, 0, 0] = -1., 1.
    d = compare(pred, ref, y, valid)
    assert d['pairwise_debit'] > 0. and d['coherent_debit'] == 0.
    assert d['contracting_stars'] == 1
    assert d['witness_lower'] == d['witness_upper'] == d['observed']


def test_bound_contains_actual_masked_completions_and_matches_old_debit():
    rng = np.random.default_rng(100)
    for _ in range(20):
        r, p, y = rng.normal(size=(3, 12, 5, 2))
        p[3:] = r[3:]
        v = rng.uniform(size=y.shape[:2]) > .4
        masked = np.where(v[..., None], y, np.nan)
        q = compare(p, r, masked, v)
        base = pairwise(p, r, masked, v)
        all_points = pairwise(p, r, y, np.ones_like(v))['lower']
        np.testing.assert_allclose(base['lower'], q['observed']-q['pairwise_debit'], rtol=0., atol=1e-12)
        assert q['coherent_debit'] <= q['pairwise_debit']
        assert q['observed']-q['coherent_debit']-1e-12 <= all_points <= q['observed']+q['coherent_debit']+1e-12


def test_no_observed_anchor_keeps_the_old_bound():
    r = np.zeros((4, 2, 2))
    p = r.copy()
    p[0, :, 0] = 1.
    v = np.zeros(r.shape[:2], dtype=bool)
    q = compare(p, r, r, v)
    assert q['coherent_debit'] == q['pairwise_debit'] == q['two_missing_debit']
    assert q['witness_lower'] == -q['coherent_debit'] and q['witness_upper'] == q['coherent_debit']


def test_mixed_float_precision_keeps_star_and_completion_algebra_aligned():
    rng = np.random.default_rng(509)
    r, p, y = rng.normal(size=(3, 44, 12, 2))
    p[2:] = r[2:]
    v = rng.random(y.shape[:2]) > .08
    yy = y.astype(np.float32)
    assert compare(p, r, yy, v) == compare(p, r, yy.astype(np.float64), v)
