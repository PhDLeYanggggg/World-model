from itertools import product

import numpy as np

from src.evaluation.m3w_recording_switch_ceiling import free_count, inspect, matched_count, moments


def fixture():
    floor = np.zeros((6,12,2)); pred = floor.copy(); target = floor.copy()
    target[:,:,0] = np.array([1,1,5,5,1,5])[:,None]
    pred[:,:,0] = np.array([.5,-.01,4,-3,.25,-1])[:,None]
    valid = np.ones((6,12),bool); valid[5] = False
    return floor,pred,target,valid,np.repeat([1,2],3)


def test_both_exact_methods_match_enumeration_with_original_support():
    a,b,y,v,q = fixture(); m = moments(a,b,y,v,2.); w = np.full(6,1/3)
    free, _ = free_count(m,w); match, status = matched_count(m,q,w)
    def values(matched):
        good = []
        for take in map(np.asarray,product((False,True),repeat=6)):
            if matched and not (take[:3].sum()==take[3:].sum()==2):continue
            if not (m['ref_easy'][take]>0).any():continue
            if max(m[k][take].sum() for k in ('all_excess','easy_excess'))>0:continue
            if m['easy_signed'][take].sum()>.02*m['ref_easy'].sum():continue
            good.append(w[take]@m['utility'][take])
        return good
    np.testing.assert_allclose(w[free]@m['utility'][free],max(values(False)))
    if values(True):
        np.testing.assert_allclose(w[match]@m['utility'][match],max(values(True)))
        assert status['optimal']
    else:
        assert match is None and status['infeasible']


def test_missing_easy_reference_is_not_a_zero_risk_pass():
    a,b,y,v,q = fixture()
    result = inspect(a,b,y,v,q,'r',.5,4.)
    assert result['free_count'] is None and result['matched_count'] is None
    assert result['solver']['infeasible']


def test_one_low_harm_easy_anchor_can_support_a_beneficial_hard_action():
    a,b,y,v,q = fixture()
    b[[0,4],:,0] = -1.
    result = inspect(a,b,y,v,q,'r',2.,4.)
    assert result['free_count']['supported'] and result['free_count']['switched']==2
