import inspect
import numpy as np
import pytest
from src.world_model import m3w_component_calibration as api


def fixture():
    p = np.tile([.4, .002, 2., 2., .002], (6, 1))
    y = np.tile([0., .02, 1., 1., .02], (6, 1))
    return p, y, np.ones(6), np.ones(6, bool), np.ones(6, bool), np.array(['a','a','b','b','c','c'])


def test_record_unit_not_window_pseudoreplication():
    p, y, e, a, _, r = fixture()
    scores = api.record_scores(p, y, e, a, r)
    got = api.fit_margin(scores)
    np.testing.assert_allclose(got['margins'], [.018, .018, .5, .5])
    assert got['component_recording_counts'] == [3]*4
    twice = api.record_scores(np.repeat(p, 2, 0), np.repeat(y, 2, 0), np.repeat(e, 2), np.repeat(a, 2), np.repeat(r, 2))
    np.testing.assert_allclose(api.fit_margin(twice)['margins'], got['margins'])
    with pytest.raises(ValueError): api.fit_margin(scores + scores)


def test_oof_excludes_held_recording_and_never_uses_its_label_for_own_margin():
    p, y, e, a, s, r = fixture()
    first, _ = api.calibrate(p, y, e, a, s, r)
    y[r == 'a', 1] = .9; y[r == 'a', 4] = .9
    second, _ = api.calibrate(p, y, e, a, s, r)
    assert first['folds'][0] == second['folds'][0]
    assert first['folds'][1] != second['folds'][1]
    assert all(f['held_recording'] not in f['calibration']['recordings'] for f in first['folds'])


def test_adjustment_is_causal_and_conservative():
    p, y, e, a, s, r = fixture()
    c, _ = api.calibrate(p, y, e, a, s, r)
    for mode in api.MODES:
        q = api.adjust(p, e, c['final'], mode)
        assert (q[:, 1] >= p[:, 1]).all() and (q[:, 4] >= p[:, 4]).all()
        assert (q[:, 2] <= p[:, 2]).all() and (q[:, 3] <= p[:, 3]).all()
        assert not (api.eligible(q, a, s) & ~api.eligible(p, a, s)).any()
    assert set(inspect.signature(api.adjust).parameters) == {'p','env','calibration','mode'}


def test_unknown_and_zero_reference_not_pass_or_known_zero():
    p, y, e, a, s, r = fixture()
    y[:] = np.nan
    c, oof = api.calibrate(p, y, e, a, s, r)
    assert not c['final']['supported'] and sum(z['unknown_selected'] for z in c['record_scores']) == 6
    assert not any(v.any() for v in oof.values())
    for mode in api.MODES:
        assert not c['source'][mode]['oof']['finite_completion_supported']
    y[0] = [0, np.nan, 1, 1, 0]
    with pytest.raises(ValueError): api.record_scores(p, y, e, a, r)


def test_single_recording_cannot_validate_own_calibration():
    p, y, e, a, s, _ = fixture()
    c, oof = api.calibrate(p, y, e, a, s, np.array(['a']*6))
    assert c['final']['supported']
    assert not c['folds'][0]['calibration']['supported']
    assert not any(v.any() for v in oof.values())


def test_matched_count_and_deterministic_id_ties():
    a = np.array([True,True,True,False]);b=np.array([True,False,True,True]);u=np.ones(4)
    left,right = api.matched(a,b,u,u,['r']*4,[0,0,1,1],np.array([2,1,4,3]))
    np.testing.assert_array_equal(left,[False,True,True,False])
    np.testing.assert_array_equal(right,[True,False,False,True])
