import numpy as np
import pytest
from src.evaluation import m3w_selected_analogue as api


def data():
    x = np.arange(24, dtype=float).reshape(12,2)/24
    env = np.ones(12); rec = np.repeat(['a','b','c'],4)
    y = np.zeros((12,5)); y[:,2:4]=1
    y[:,1] = y[:,4] = np.linspace(0,.5,12)
    mask = np.ones(12,bool)
    return x,env,y,rec,mask,mask.copy()


def test_target_recording_labels_cannot_change_own_inference():
    args = data(); a = api.retrieve(*args,k=3)
    changed = list(args); changed[2] = args[2].copy(); changed[2][:4] = np.nan
    b = api.retrieve(*changed,k=3)
    for mode in api.MODES:
        np.testing.assert_array_equal(a['predictions'][mode][:4],b['predictions'][mode][:4])
    assert a['references'][0] == b['references'][0]
    assert all(r['same_recording_donors'] == 0 for r in a['references'])
    assert b['supported'][:4].all()


def test_unsupported_queries_are_retained_not_zero_risk():
    args = data(); out = api.retrieve(*args,k=9)
    assert len(out['indices']) == 12 and not out['supported'].any()
    p = args[2].copy(); p[:,[1,4]] = out['predictions']['nearest32']
    s = api.score(args[2],p,args[1],1.,out['supported'])
    assert s['rows'] == s['unsupported'] == 12
    assert s['easy']['full_cohort_MSE'] is None
    with pytest.raises(ValueError): api.summarize([],set())


def test_distance_and_mean_match_manual_reference_and_deterministic_replay():
    args = data(); out = api.retrieve(*args,k=3); again=api.retrieve(*args,k=3)
    np.testing.assert_array_equal(out['predictions']['nearest32'],again['predictions']['nearest32'])
    # Features are ordered and collinear: row0's three nearest cross-record rows are4,5,6.
    np.testing.assert_allclose(out['predictions']['nearest32'][0],args[2][4:7][:,[1,4]].mean(0))
    np.testing.assert_allclose(out['predictions']['source_mean'][:4],
                              np.tile(args[2][4:][:,[1,4]].mean(0),(4,1)))
    assert out['references'] == again['references']
