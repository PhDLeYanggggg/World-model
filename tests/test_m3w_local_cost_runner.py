import numpy as np
from scripts import run_m3w_local_cost_analogue as run
from tests.test_m3w_local_cost_analogue import inputs


def test_real_runner_diagnostic_path_handles_unknowns_and_masks():
    bank,q,frozen=inputs()
    bank['labels'][0]=np.nan
    out=run.api.retrieve(bank,q,frozen,k=4)
    ix=out['indices'];p={'origin':frozen[ix]}
    for arm in run.api.MODES:
        p[arm]=p['origin'].copy();p[arm][:,[1,4]]=out['predictions'][arm]
    d=run.diagnostic(out,np.ones(len(ix),bool),bank['labels'][ix],p,bank['envelope'][ix],2.)
    assert d['supported_queries']==30 and d['known_queries']==29
    assert set(d['arms'])==set(run.api.ARMS)
    assert np.all(np.array(d['donor_loss']['optimum'])<=np.array(d['donor_loss']['uniform'])+1e-12)
