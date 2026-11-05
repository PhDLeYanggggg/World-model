import copy

import numpy as np
import pytest

from scripts import run_m3w_moving_nonlinear_readout as run


def test_frozen_parent_preprocessing_exclusion_is_not_silently_skipped():
    state = dict(fit_ids=np.array([1, 2]), source_sites=['fit'],
        frozen=dict(fit_ids=np.array([1, 2]), source_sites=['fit'], preprocess=dict(source_names=['fit'])))
    before = copy.deepcopy(state)
    run.assert_readout_excluded(state, dict(ids=np.array([3, 4])), ['held'])
    run.api.base.exact(state, before)
    assert 'preprocess' not in state
    with pytest.raises(ValueError, match='must be disjoint'):
        run.assert_readout_excluded(state, dict(ids=np.array([2, 4])), ['held'])
    with pytest.raises(ValueError, match='must be disjoint'):
        run.assert_readout_excluded(state, dict(ids=np.array([3, 4])), ['fit'])
    altered = copy.deepcopy(state); altered['frozen']['preprocess']['source_names'] = ['held']
    with pytest.raises(ValueError, match='must be disjoint'):
        run.assert_readout_excluded(altered, dict(ids=np.array([3, 4])), ['held'])
    altered = copy.deepcopy(state); altered['fit_ids'] = np.array([999])
    with pytest.raises(AssertionError):
        run.assert_readout_excluded(altered, dict(ids=np.array([3, 4])), ['held'])
