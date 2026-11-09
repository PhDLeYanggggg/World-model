from scripts.audit_m3w_vehicle_release_transport import comparison, count_prefix, groups


PASS = (True, True, True, False, 1., 1.)
HARM = (False, False, False, False, -1., -1.)
MISSING = (False, False, False, True, 0., 0.)


def test_same_count_never_splits_ties():
    assert count_prefix(groups([.7, .9, .7, None]), 2) == [1]
    assert count_prefix(groups([.7, .9, .7, None]), 3) == [1, 0, 2]


def test_detects_scale_shift_without_using_held_labels_to_select():
    r = comparison([10., 9.], [PASS, HARM], [2., 1.], [PASS, HARM])
    assert r['held_below_numeric_cut']
    assert r['held_hindsight_rank_prefix'] == 1
    assert r['policies']['numeric']['released'] == 0
    assert r['policies']['matched_fraction']['full'] == 1
    q = comparison([10., 9.], [PASS, HARM], [2., 1.], [MISSING, PASS])
    assert q['policies']['matched_fraction']['released'] == 1
    assert q['policies']['matched_fraction']['full'] == 0
    assert q['policies']['matched_fraction']['hindsight_any_order_full'] == 1
    assert q['held_hindsight_rank_prefix'] == 0


def test_zero_calibration_support_does_not_turn_into_release():
    r = comparison([.8, .8], [PASS, HARM], [.9, .8], [PASS, PASS])
    assert r['calibration_prefix'] == 0
    assert all(v['released'] == 0 for v in r['policies'].values())
