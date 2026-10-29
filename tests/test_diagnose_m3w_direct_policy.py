import copy

import pytest

from scripts.diagnose_m3w_direct_policy import describe
from tests.test_m3w_direct_policy_readout import result


def test_mass_decomposition_retains_missing_ratio_and_scopes():
    data = result()
    rows = [dict(scope=s, source='one', result=copy.deepcopy(data)) for s in ('fit', 'inner_held')]
    out = describe(rows)
    for scope in out['scopes'].values():
        assert scope['unique_source_recordings'] == 2
        assert scope['primary_known_benefit_retained'] is None
        assert scope['primary_known_harm_retained'] is None
        assert scope['primary_known_net_change_pp'] == pytest.approx(10.)
    assert not out['thresholds_changed'] and not out['new_fit']


def test_cannot_drop_scope_or_zero_reference():
    rows = [dict(scope=s, source='one', result=result()) for s in ('fit', 'inner_held')]
    with pytest.raises(ValueError, match='Both'):
        describe(rows[:1])
    rows[0]['result']['full_known_reference_mass'] = 0.
    with pytest.raises(ValueError, match='undefined'):
        describe(rows)
