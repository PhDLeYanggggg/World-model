import numpy as np
import pytest

from scripts.audit_m3w_opendd_paired_window_budget import pairs
from src.world_model.m3w_paired_window_budget import allocate, scored_window


def test_two_decoders_have_same_choices_but_can_place_slots_differently():
    def left(r):
        return (r, r), np.asarray([sum(r), 3.*sum(r)])

    def right(r):
        if len(r) > 1:
            return None
        return (r, r), np.asarray([4.*sum(r), sum(r)])

    out = allocate(left, right, (1.,), (.5,))
    assert out['fixed'] == [((1.,), (.5,))]*2
    assert out['pooled'] == [((.5,), (1.,)), ((1., .5), ())]
    assert out['moved'] == [(1, 1), (1, 0)]
    assert out['predicted_gain'] == [1.5, 1.]


def test_ties_leave_budget_in_the_original_windows():
    def tied(r):
        return (r, r), np.zeros(2)

    a = allocate(tied, tied, (1.,), (.5,))
    assert a['pooled'] == a['fixed'] and a['moved'] == [(0, 0), (0, 0)]


def test_pairing_cannot_cross_recordings_or_allow_overlap():
    rec = np.array(['a', 'a', 'a', 'b', 'b'])
    groups = pairs(rec, np.array([0, 240, 480, 5, 245]), stride=12)
    assert [g.tolist() for g in groups] == [[0, 1], [2], [3, 4]]
    with pytest.raises(ValueError, match='Overlapping'):
        list(pairs(rec, np.array([0, 240, 480, 5, 15]), stride=12))


def test_illegal_slot_count_is_not_an_option():
    h = np.zeros((2, 8, 2))
    cv = np.zeros((2, 12, 2))
    assert scored_window(None, h, cv, cv, cv[:, None], (1., 2.)) is None
    assert scored_window(None, h, cv, cv, cv[:, None], (1.,)) is None
