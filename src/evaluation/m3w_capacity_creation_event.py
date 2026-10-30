"""Preserve one registration-time event without weakening scientific identity."""
from copy import deepcopy

FIELD = 'parent_policy_readout_complete_at_registration'


def preserve_creation_event(proposed, recorded):
    if type(proposed.get(FIELD)) is not bool or type(recorded.get(FIELD)) is not bool:
        raise ValueError('Both historical event flags must be explicit booleans')
    value = deepcopy(proposed)
    value[FIELD] = recorded[FIELD]
    if value != recorded:
        raise ValueError('Only the historical creation event may differ; scientific identity changed')
    return value
