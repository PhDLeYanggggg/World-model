import pytest

from scripts.freeze_m3w_dronecrowd_calibration_roles import partition


def test_partition_uses_complete_groups_and_distinct_recordings():
    group = dict(a='g1', b='g1', c='g2')
    role = dict(calibration=dict(groups=['g1'], recordings=['a', 'b']),
                confirmation=dict(groups=['g2'], recordings=['c']))
    assert partition(group, role) == dict(a='calibration', b='calibration', c='confirmation')
    role['calibration']['recordings'].remove('a')
    role['confirmation']['recordings'].append('a')
    with pytest.raises(ValueError):
        partition(group, role)


def test_partition_does_not_drop_groups():
    with pytest.raises(ValueError):
        partition(dict(a='g1', b='g2', c='g3'),
            dict(calibration=dict(groups=['g1'], recordings=['a']),
                 confirmation=dict(groups=['g2'], recordings=['b'])))
