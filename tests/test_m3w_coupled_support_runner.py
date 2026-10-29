import numpy as np
from scripts import run_m3w_coupled_support as run
from tests.test_m3w_local_cost_analogue import inputs


def test_diagnostic_writer_keeps_full_roster_and_partitions():
    b,q,p=inputs();out=run.api.retrieve(b,q,p,k=4)
    preds={'origin':p[q].copy()}
    for a in run.api.MODES:
        preds[a]=p[q].copy();preds[a][:,[1,4]]=out['predictions'][a]
    d=run.diagnostic(out,np.ones(q.sum(),bool),b['labels'][q],preds,b['envelope'][q],2.)
    assert d['local_supported']==q.sum()
    assert sum(d['overlap'].values())==q.sum()
    assert sum(d['row_leverage_overlap'].values())==q.sum()
    for a,c in (('alpha_zero','alpha_positive'),('one_reference_record','multiple_reference_records'),
                ('record_ESS_below2','record_ESS_atleast2')):
        assert d['slices'][a]['rows']+d['slices'][c]['rows']==q.sum()
    assert d['nested_loss_after'] <= d['nested_loss_before']+1e-12
    assert all(d['arms'][a]['easy_exceeds_all']==0 for a in run.api.JOINT_ARMS)
