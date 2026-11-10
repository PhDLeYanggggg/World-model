import numpy as np

from src.evaluation.m3w_interaction_frontier import pair_reduction
from src.evaluation.m3w_sparse_pair_bounds import compare


def test_sparse_matches_both_directions_of_dense_bound():
    rng = np.random.default_rng(173)
    ref, pred, target = rng.normal(size=(3, 101, 12, 2))
    pred[np.arange(len(pred)) % 17 != 3] = ref[np.arange(len(ref)) % 17 != 3]
    valid = rng.random(size=target.shape[:2]) > .12
    target[~valid] = np.nan
    value = compare(pred, ref, target, valid)
    both = pair_reduction(pred[None], ref, target, valid)[0]
    reverse = pair_reduction(ref[None], pred, target, valid)[0]
    np.testing.assert_allclose([value['available'], value['lower'], value['upper']],
                               [both[0], both[1], -reverse[1]], atol=1e-12, rtol=0.)
    assert value['affected_pairs'] < 101*100//2


def test_identity_with_missing_targets_does_not_create_available_evidence():
    x = np.zeros((5, 12, 2))
    v = np.zeros(x.shape[:2], dtype=bool)
    result = compare(x, x, x, v)
    assert result == dict(lower=0., upper=0., available=None, pair_steps=120,
                          observed_pair_steps=0, affected_pairs=0)
