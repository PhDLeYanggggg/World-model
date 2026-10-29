import numpy as np

from scripts.diagnose_m3w_policy_conditional import accounting


def test_removal_accounting_is_partitioned_and_unknown_is_not_harm():
    z=dict(prediction=np.tile([.2,.001,1.,1.,.001],(4,1)),envelope=np.ones(4),
        targets=np.array([[.3,0.,1.,1.,0.],[0.,.1,1.,1.,.1],[np.nan]*5,[.2,0.,1.,1.,0.]]),
        action_raw=np.ones(4,bool),action_support_only=np.array([0,1,1,1],bool),
        action_conditional=np.array([0,0,0,1],bool))
    cal=dict(table={str(i):dict(offset=[.01,.01]) for i in range(16)})
    r=accounting(z,cal)['partition']
    assert r['unsupported_removed']['rows']==1 and r['penalty_removed']['rows']==2
    assert r['penalty_removed']['unknown']==1
    assert r['penalty_removed']['known_harm_pp']==100*.1/3
    assert r['penalty_removed']['unknown_envelope_pp']==100/3
    assert r['raw']['rows']==sum(r[k]['rows'] for k in ('unsupported_removed','penalty_removed','retained'))
