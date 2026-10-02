import inspect

import numpy as np
import pytest

from src.world_model import m3w_selected_set_calibration as api


def example():
    p = np.tile([.4, .002, 2., 2., .002], (9, 1))
    y = np.tile([0., .015, 1., 1., .015], (9, 1))
    return p, y, np.ones(9), np.ones(9, bool), np.ones(9, bool), np.repeat(['a','b','c'],3)


def test_fixed_point_is_not_repeated_addition():
    p, y, e, m, s, r = example()
    c = api.fit(p, y, e, m, s, r, 'joint')
    assert c['status'] == 'selected_set_fixed_point'
    assert len(c['trace']) == 1
    np.testing.assert_allclose(c['margins'], [.013, .013, .5, .5])
    assert api.infer(p, e, m, s, c, 'joint')[1].all()


def test_held_labels_do_not_affect_their_own_calibrator_or_actions():
    p, y, e, m, s, r = example()
    a = api.evaluate(p, y, e, m, s, r)
    y[r == 'a'] = [0., .8, .5, .5, .8]
    b = api.evaluate(p, y, e, m, s, r)
    for x, z in zip(a['rows'][::2], b['rows'][::2]):
        assert x['folds'][0] == z['folds'][0]
        assert all(f['held_recording'] not in f['calibration']['recordings'] for f in x['folds'])
    assert set(inspect.signature(api.infer).parameters) == {'p','env','moving','support','fitted','mode'}


def test_unknown_rows_are_not_safe_zero_and_no_empty_pass():
    p, y, e, m, s, r = example()
    y[:] = np.nan
    result = api.evaluate(p, y, e, m, s, r)
    for row in result['rows']:
        assert row['selected'] == 0
        assert row['selected_set']['easy_selected_risk_upper'] is None
        assert not row['selected_set']['finite_completion_supported']


def test_single_recording_oof_unsupported():
    p, y, e, m, s, r = example()
    result = api.evaluate(p, y, e, m, s, np.array(['one']*len(r)))
    assert all(row['selected'] == 0 for row in result['rows'][::2])


def test_monotone_shrinkage_and_deterministic_replay():
    rng = np.random.default_rng(11)
    p, y, e, m, s, r = example()
    p[:,2:4] *= rng.uniform(.3, 3., (len(p),1))
    y[:,1] = y[:,4] = rng.uniform(0., .05, len(y))
    a = api.evaluate(p,y,e,m,s,r)
    assert a == api.evaluate(p,y,e,m,s,r)
    for row in a['rows']:
        assert row['selected'] <= row['parent_selected']
        cs = [f['calibration'] for f in row['folds']] if 'folds' in row else [row['final']]
        for c in cs:
            for step in c['trace']:
                assert step['next_selected'] <= step['selected']
                assert np.all(np.array(step['next_margins']) >= step['margins'])


def test_partial_nan_and_invalid_rounds_rejected():
    p, y, e, m, s, r = example()
    with pytest.raises(ValueError): api.fit(p,y,e,m,s,r,'joint',rounds=0)
    y[0,1] = np.nan
    with pytest.raises(ValueError): api.fit(p,y,e,m,s,r,'joint')


def test_original_raw_calibration_is_not_guaranteed_selected_calibration():
    # Raw-pool reference mass from soon-removed rows can conceal a retained
    # small-reference case. Recalibration must act on the latter population.
    p = np.array([[.8,.001,10,10,.001], [.001,.0009,1,1,.0009]]*3)
    y = np.array([[0,.05,.1,.1,.05], [.5,0,100,100,0]]*3)
    e = np.ones(6); m = s = np.ones(6,bool); r = np.repeat(['a','b','c'],2)
    c = api.fit(p,y,e,m,s,r,'joint')
    assert c['status'] == 'empty_selected_set'
    assert not api.infer(p,e,m,s,c,'joint')[1].any()
    assert len(c['trace']) >= 1
