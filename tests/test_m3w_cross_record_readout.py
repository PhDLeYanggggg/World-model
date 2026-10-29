import numpy as np
import pytest
from src.evaluation import m3w_cross_record_readout as api
from scripts.report_m3w_shared_train_risk import check_policy


def example():
    y = np.tile([.1, 0., 1., 1., 0.], (4,1))
    args = (np.zeros((4,380)), np.ones(4), y, None, None,
            np.array(['a','a','b','b']), np.arange(4), {})
    masks = {k:np.ones(4,bool) for k in api.POLICIES}
    masks['floor'][:] = False; masks['origin'][:] = False
    return api.evaluate(args, masks, {k:np.arange(4) for k in masks})


def test_empty_matched_risk_not_promoted():
    d = example()
    for p in d['policies'].values(): check_policy(p)
    c = d['comparisons']['cross_record_meta_minus_origin']
    assert c['matched_empty'] and c['matched_right']['easy_selected_risk_upper'] is None
    assert c['matched_risk_role'] == 'utility_only_not_safety_gate'
    assert c['empty_risk_status'] == 'not_estimable'
    assert c['full']['lower_pp'] > 0 and c['matched']['lower_pp'] == 0


def test_complete_cohort_and_no_confirmation():
    rows = [dict(context=str(c),source=str(s),seed=k,scope=scope,result=example())
            for scope in ('fit','inner_held') for c in range(6) for s in range(4) for k in range(3)]
    out = api.summarize(rows)
    assert not out['independent_confirmation'] and not out['deployment_changed']
    assert out['scopes']['fit']['safety']['floor']['budget_views'] == 0
    with pytest.raises(ValueError): api.summarize(rows[:-1])
