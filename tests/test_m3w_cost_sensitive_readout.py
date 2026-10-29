import numpy as np
import pytest
from src.evaluation import m3w_cost_sensitive_readout as api
from scripts.report_m3w_shared_train_risk import check_policy


def fixture():
    y = np.array([[.1,0.,1.,1.,0.],[0.,.2,1.,1.,.2],[.1,0.,1.,1.,0.]])
    args=(np.zeros((3,380)),np.ones(3),y,None,None,np.array([1,1,2]),np.arange(3),{})
    masks=dict(origin=np.ones(3,bool),exact_gradient=np.zeros(3,bool),cost_sensitive=np.array([True,False,True]))
    ranks={a:np.arange(3,dtype=float) for a in api.ARMS}
    return api.evaluate(args,masks,ranks)


def test_accounting_includes_numeric_recordings_and_empty_controls():
    r=fixture()
    for a in api.ARMS:
        check_policy(r['policies'][a])
        assert sum(v[a]['known_rows'] for v in r['recordings'].values())==3
        for v in r['recordings'].values():check_policy(v[a])
    pair=r['pairs']['cost_sensitive_minus_exact_gradient']
    assert pair['matched_empty'] and pair['matched_left']['easy_selected_risk_upper'] is None
    assert pair['matched_role']=='utility_only' and pair['full']['lower_pp']>0


def test_complete_fit_roster_never_promoted_to_held_success():
    rows=[dict(context=str(c),source=str(s),seed=k,result=fixture())
          for c in range(6) for s in range(4) for k in (17,29,43)]
    d=api.summarize(rows)
    assert d['safety']['cost_sensitive']['budget_views']==72
    assert not d['inner_held_scored'] and not d['independent_confirmation'] and not d['deployment_changed']
    with pytest.raises(ValueError):api.summarize(rows[:-1])
