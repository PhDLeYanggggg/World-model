import numpy as np
import pytest

from scripts.check_m3w_external_three_seed_bank import existing_batch, fit_exposure


def test_probe_keeps_original_full_batch_boundary():
    take, loc = existing_batch(np.arange(10), np.array([0, 2, 3, 4, 6, 7, 8, 9]), batch=2, count=2)
    np.testing.assert_array_equal(take, [2,3,6,7])
    np.testing.assert_array_equal(loc, [1,2,4,5])
    with pytest.raises(ValueError, match='shared A-only'):
        existing_batch(np.arange(10), np.array([0, 2]), batch=2, count=1)


def test_actual_fit_roster_uses_past_support_and_keeps_unscreened_clips():
    meta = {k:{'clip':k, 'exclusion_group':v} for k,v in [('a','fit'), ('b','fit'), ('c','held'), ('d','fit')]}
    x = fit_exposure(meta, dict(a=2,b=3,c=4,d=0), ['fit'], {'a','c'})
    assert x == dict(recordings=['a','b'], rows=5, groups=['fit'], unscreened=['b'])
