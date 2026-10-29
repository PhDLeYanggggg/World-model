import copy
import numpy as np
import pytest
from src.evaluation import m3w_whole_record_readout as api
from scripts.report_m3w_shared_train_risk import check_policy


def example():
    y = np.tile([.1, 0., 1., 1., 0.], (4, 1))
    args = (np.zeros((4, 380)), np.ones(4), y, None, None,
            np.array(['a', 'a', 'b', 'b']), np.arange(4), {})
    masks = {a: np.ones(4, bool) for a in api.POLICIES}
    masks['floor'][:] = False; masks['origin'][:] = False
    return api.evaluate(args, masks, {a: np.arange(4) for a in masks})


def test_policy_accounting_and_empty_controls_keep_prospective_contract():
    d = example()
    for p in d['policies'].values(): check_policy(p)
    c = d['comparisons']['exact_gradient_minus_origin']
    assert c['matched_empty'] and c['matched_right']['easy_selected_risk_upper'] is None
    assert c['full']['lower_pp'] > 0 and c['matched']['lower_pp'] == 0
    assert c['matched_risk_role'] == 'utility_only_not_safety_gate'


def test_fixed_complete_cohorts_and_no_deployment_promotion():
    rows = [dict(context=str(c), source=str(s), seed=k, scope=scope, result=example())
            for scope in ('fit', 'inner_held') for c in range(6) for s in range(4) for k in range(3)]
    summary = api.summarize(rows)
    assert not summary['independent_confirmation'] and not summary['deployment_changed']
    assert summary['scopes']['fit']['safety']['floor']['budget_views'] == 0
    with pytest.raises(ValueError): api.summarize(rows[:-1])
