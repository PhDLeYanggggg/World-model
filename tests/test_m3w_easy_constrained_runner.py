import numpy as np
from scripts import run_m3w_easy_constrained as run
from tests.test_m3w_local_cost_analogue import inputs


def test_channel_diagnostics_preserve_query_partition_and_constraints():
    b,q,p=inputs();out=run.api.retrieve(b,q,p,k=4);pred={'origin':p[q].copy()}
    for a in run.api.MODES:
        pred[a]=p[q].copy();pred[a][:,[1,4]]=out['predictions'][a]
    d=run.diagnostic(out,np.ones(q.sum(),bool),b['labels'][q],pred,2.)
    assert d['donor_easy_constraint_violations']==0
    assert d['channel_loss']['after'][1]<=d['channel_loss']['before'][1]+1e-12
    for a,c in (('donor_conflict','no_donor_conflict'),('reduced_alpha','unchanged_alpha'),('alpha_zero','alpha_positive')):
        assert d['strata'][a]['rows']+d['strata'][c]['rows']==q.sum()
