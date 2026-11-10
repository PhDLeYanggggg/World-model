from io import BytesIO

import numpy as np
import pytest

from scripts.rehearse_m3w_pneuma_confirmation import DATES, fixture
from scripts.run_m3w_pneuma_single_readout import FIELDS, budget, frozen, restore, same_scores
from src.evaluation import m3w_pneuma_matched_readout as primary


@pytest.mark.parametrize('tamper', [None, 'profile', 'queries', 'schema'])
def test_restore_requires_identical_roster_and_profiles(tamper):
    cv, ols, median, y, valid, offsets, proposal = fixture(next(iter(DATES)))
    arrays = {}
    for name in frozen.CONTROLS:
        p = budget.build_pair(cv, offsets, np.arange(len(offsets)-1), proposal, name)
        arrays.update({f'{name}_queries': p['queries'], f'{name}_profile': p['profiles']})
    if tamper == 'profile':
        arrays['additive_profile'][0, 0] *= .9
    elif tamper == 'queries':
        arrays['independent_queries'] = np.array([1, 0])
    elif tamper == 'schema':
        arrays.pop('same_family_profile')
    z = BytesIO()
    np.savez(z, **arrays)
    z.seek(0)
    with np.load(z, allow_pickle=False) as profiles:
        if tamper:
            with pytest.raises(ValueError, match='Stepwise'):
                restore(cv, offsets, proposal, profiles)
            return
        restored = restore(cv, offsets, proposal, profiles)
    scores = primary.score(cv, ols, median, y, valid, offsets, proposal, easy_cut=.16, hard_cut=.28)
    for k, pair in restored.items():
        value = budget.score_pair(cv, offsets, pair, y, valid)['original']
        same_scores(value, scores['matched_contrasts'][k], FIELDS)


def test_missing_support_does_not_match_zero():
    with pytest.raises(ValueError, match='different support'):
        same_scores({'lower_pp': None}, {'lower_pp': 0.}, ('lower_pp',))
