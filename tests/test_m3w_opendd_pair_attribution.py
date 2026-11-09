import numpy as np
import pytest

from scripts import audit_m3w_opendd_pair_attribution as audit


def sample():
    cv = np.zeros((4, 3, 2))
    additive, joint = cv.copy(), cv.copy()
    additive[:2, :, 0] = 1.
    joint[0, :, 0] = -1.
    joint[1, :, 0] = 1.
    labels = joint.copy()
    valid = np.ones((4, 3), dtype=bool)
    return cv, additive, joint, labels, valid


def test_single_changed_actor_decomposes_relative_error_and_missing_debit():
    cv, a, b, labels, valid = sample()
    valid[0, -1] = False
    result = audit.query(cv, a, b, labels, valid)
    assert result['kind'] == 'path_only'
    assert result['known'] == 1.
    assert result['lower'] == .5
    assert result['edge']['changed_with_edited']['gain'] == 4.
    assert result['edge']['changed_with_cv']['gain'] == 8.
    assert result['edge']['both_changed']['pairs'] == 0
    stats = audit.summarize([audit.recording([result])])
    assert stats['geometry_total_native'] == .8
    assert stats['total']['debit_pp'] == 40.


def test_recipient_swap_keeps_count_but_changes_both_ends():
    cv, a, b, _, valid = sample()
    b[2], b[0] = b[0].copy(), cv[0]
    result = audit.query(cv, a, b, b, valid)
    assert result['kind'] == 'recipient_change'
    assert result['replaced_slots'] == 1
    assert result['edge']['both_changed']['pairs'] == 1
    assert result['edge']['changed_with_edited']['pairs'] == 2
    assert result['edge']['changed_with_cv']['pairs'] == 2
    assert result['edge']['unchanged']['pairs'] == 1


def test_inactive_recording_and_one_edit_null_control():
    cv, a, b, labels, valid = sample()
    inactive = audit.query(cv, cv, cv, labels, valid)
    assert audit.summarize([audit.recording([inactive])])['total']['known_pp'] is None
    a[1], b[1] = cv[1], cv[1]
    with pytest.raises(ValueError, match='outside two-edit'):
        audit.query(cv, a, b, labels, valid)


def test_hindsight_rejects_negative_gains_separately_at_each_query():
    a = dict(mass=2., edits=2, known=.2, lower=-.1)
    b = dict(mass=2., edits=2, known=.1, lower=.05)
    c = dict(mass=3., edits=2, known=.4, lower=.1)
    d = dict(mass=3., edits=2, known=.6, lower=-.1)
    assert audit.menu_hindsight([(a, b), (c, d)]) == pytest.approx(dict(known=16., lower=3.))
