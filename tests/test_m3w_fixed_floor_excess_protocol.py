import inspect
import json
import numpy as np
from scripts import run_m3w_european_fixed_floor_excess as run


def test_fixed_objective_budget_and_no_new_role_access():
    cfg=json.loads((run.ROOT/run.CONFIG).read_text())
    assert cfg['new_heads']==108 and cfg['risk_budget']==.02
    assert cfg['head_training']['steps']==2000 and cfg['bootstrap_resamples']>=2000
    assert not any(cfg[k] for k in ('threshold_search','independent_roles_read','deployment_changed','stage5c_executed','smc_enabled'))


def test_actions_need_no_future_and_do_not_borrow_another_frame():
    data=dict(sites=np.array(['a']*4),recordings=np.array(['r']*4),frames=np.array([1,1,2,2]))
    c=dict(ids=np.arange(4),moving=np.ones(4,bool)); held=np.arange(4)
    old=dict(support=np.ones(4,bool),scores=np.tile([0]*5+[2,1,0,0,0],(4,1)),floor_safe=np.zeros(4,bool))
    mse=np.array([[10,.4,10,.4],[10,.1,10,.1],[10,0,10,0],[10,0,10,0]])
    new=np.array([[10,.1,10,.1],[10,.4,10,.4],[10,.4,10,.4],[10,.4,10,.4]])
    a=run.actions(c,data,held,old,mse,new)
    assert a['excess'].tolist()==[True,False,False,False]
    assert a['mse_matched_count'].tolist()==[False,True,False,False]
    assert not any(k in inspect.signature(run.api.predict).parameters for k in ('target','future','valid'))
