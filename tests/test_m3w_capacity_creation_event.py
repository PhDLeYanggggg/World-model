import pytest
from src.evaluation.m3w_capacity_creation_event import preserve_creation_event


def fixture():
    return dict(parent_policy_readout_complete_at_registration=False,
        bindings={'code.py': 'fixed'}, risk_budget=.02, groups=72,
        independent_roles_opened=False)


def test_historical_completion_flag_is_preserved_not_recomputed():
    recorded = fixture(); proposed = dict(recorded, parent_policy_readout_complete_at_registration=True)
    assert preserve_creation_event(proposed, recorded) == recorded
    assert proposed['parent_policy_readout_complete_at_registration'] is True


@pytest.mark.parametrize('key,value', [('bindings', {'code.py': 'different'}),
    ('risk_budget', .03), ('groups', 71), ('independent_roles_opened', True)])
def test_every_scientific_identity_or_role_change_still_refuses(key, value):
    recorded = fixture(); proposed = dict(recorded, **{key: value})
    with pytest.raises(ValueError): preserve_creation_event(proposed, recorded)


def test_missing_or_nonboolean_event_is_not_an_allowed_repair():
    recorded = fixture(); proposed = dict(recorded)
    del proposed['parent_policy_readout_complete_at_registration']
    with pytest.raises(ValueError): preserve_creation_event(proposed, recorded)
    with pytest.raises(ValueError):
        preserve_creation_event(dict(recorded, parent_policy_readout_complete_at_registration=1), recorded)
