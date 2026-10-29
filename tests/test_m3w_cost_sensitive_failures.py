import copy
import pytest

from scripts.analyze_m3w_cost_sensitive_failures import classify, dual_snapshot, run
from tests.test_m3w_cost_sensitive_readout import fixture


def test_undefined_easy_is_not_budget_pass():
    policy=fixture()['pairs']['cost_sensitive_minus_exact_gradient']['matched_right']
    assert policy['selected_count']==0
    actual=classify(policy)
    assert actual['supported']==0
    assert actual['easy_selected_risk_upper_undefined']==1


def test_missing_metric_is_error_not_undefined():
    policy=copy.deepcopy(fixture()['policies']['origin'])
    del policy['known_positive_risk']
    with pytest.raises(KeyError):classify(policy)


def test_dual_snapshot_distinguishes_cap_from_violation():
    torch=run.api.torch
    state=dict(step=2000,trace=[dict(step=2000,exact_record_risk=[[1.,-1.],[-1.,2.]])],
        dual=torch.tensor([[100.,100.],[0.,5.]]),record_scales=torch.ones((2,2)),
        settings=dict(dual_maximum=100.))
    value=dual_snapshot(state)
    assert value['all']['positive_risk_with_dual_at_cap']==1
    assert value['easy']['dual_at_cap']==1
    assert value['easy']['positive_risk_with_dual_at_cap']==0
    state['trace'][0]['step']=1999
    with pytest.raises(ValueError):dual_snapshot(state)
