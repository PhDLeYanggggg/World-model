import json
import numpy as np
from scripts import run_m3w_european_causal_descriptor_refit as run


def test_same_loss_budget_and_all_six_causal_features():
    cfg = json.loads((run.ROOT/run.CONFIG).read_text())
    control = json.loads((run.ROOT/run.base.CONFIG).read_text())
    assert cfg['head_training'] == control['head_training'] and cfg['risk_budget'] == .02
    assert cfg['descriptor_features'] == list(run.api.FEATURES) and len(run.api.FEATURES) == 6
    assert cfg['groups'] == cfg['new_heads'] == 108
    assert not any(cfg[k] for k in ('threshold_search','independent_roles_read','deployment_changed','stage5c_executed','smc_enabled'))


def test_actions_match_current_query_counts_including_unknown_rows():
    data = dict(sites=np.array(['a']*4), recordings=np.array(['r']*4), frames=np.array([1,1,2,2]))
    c = dict(ids=np.arange(4),moving=np.ones(4,bool)); held = np.arange(4)
    old = dict(support=np.ones(4,bool),scores=np.tile([0]*5+[2,1,0,0,0],(4,1)),floor_safe=np.zeros(4,bool))
    control = np.array([[10,.4,10,.4],[10,.1,10,.1],[10,0,10,0],[10,0,10,0]])
    new = np.array([[10,.1,10,.1],[10,.4,10,.4],[10,.4,10,.4],[10,.4,10,.4]])
    a = run.actions(c,data,held,old,control,control,new)
    assert a['descriptor'].tolist() == [True,False,False,False]
    assert a['control_matched_count'].tolist() == [False,True,False,False]


def test_compressed_decisions_roundtrip_exact(tmp_path):
    path = tmp_path/'decisions.npz'; a = dict(ids=np.arange(10),take=np.arange(10)%2 == 0)
    run.write_arrays(path,a); run.write_arrays(path,a,True)
