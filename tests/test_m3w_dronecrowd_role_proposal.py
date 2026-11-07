import pytest

from scripts.propose_m3w_dronecrowd_roles import proposed_groups


def test_proposals_preserve_groups_and_keep_excluded_recordings_out():
    groups = dict(a='g1', b='g1', c='g2', d='g3', e='g4', f='g4', g='g5')
    out = proposed_groups(groups, ['g4'])
    assert out['excluded']['recordings'] == ['e', 'f']
    cal, test = (out['split_calibration_confirmation'][k] for k in ('calibration', 'confirmation'))
    assert set(cal['groups']).isdisjoint(test['groups'])
    assert set(cal['recordings']) | set(test['recordings']) == {'a', 'b', 'c', 'd', 'g'}
    assert ('a' in cal['recordings']) == ('b' in cal['recordings'])
    assert out['frozen_zero_shot']['calibration']['groups'] == []
    assert out == proposed_groups(dict(reversed(list(groups.items()))), ['g4'])


def test_insufficient_groups_are_not_declared_as_separate_roles():
    out = proposed_groups({'a':'g1', 'b':'g2'}, ['g2'])
    assert out['split_calibration_confirmation'] is None
    with pytest.raises(ValueError, match='Unknown quarantined group'):
        proposed_groups({'a':'g1'}, ['not-a-group'])
