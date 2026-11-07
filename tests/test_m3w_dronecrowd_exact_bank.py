import copy

import pytest

from scripts.audit_m3w_dronecrowd_exact_bank import role_compatibility


def inputs():
    groups = {'00001': 'a', '00002': 'a', '00003': 'b'}
    registry = {'records': {'dronecrowd_'+k: dict(dataset='DroneCrowd',
        exclusion_group=g, reserved_role='confirmation', predictive_admission=False)
        for k, g in groups.items()}}
    empty = dict(groups=[], recordings=[])
    a = dict(groups=['a'], recordings=['00001', '00002'])
    b = dict(groups=['b'], recordings=['00003'])
    whole = dict(groups=['a', 'b'], recordings=list(groups))
    proposal = dict(scientific_roles_assigned=False, predictive_admission=False,
        outcome_readout_allowed=False, outcome_or_support_counts_used_for_partition=False,
        alternatives=dict(excluded=empty, frozen_zero_shot=dict(calibration=empty, confirmation=whole),
                          split_calibration_confirmation=dict(calibration=a, confirmation=b)))
    return registry, groups, proposal


def test_new_calibration_conflicts_but_zero_shot_retains_whole_reservation():
    registry, groups, proposal = inputs()
    before = copy.deepcopy(registry)
    r = role_compatibility(registry, groups, proposal)
    assert r['split_calibration_confirmation']['conflicts_with_existing_reservation']
    assert not r['frozen_zero_shot']['conflicts_with_existing_reservation']
    assert r['frozen_zero_shot']['confirmation_recordings'] == 3
    assert registry == before


def test_reservation_is_not_a_predictive_admission():
    registry, groups, proposal = inputs()
    registry['records']['dronecrowd_00001']['predictive_admission'] = True
    with pytest.raises(ValueError, match='reservation'):
        role_compatibility(registry, groups, proposal)
