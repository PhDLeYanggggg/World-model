import numpy as np
import torch
from src.world_model.m3w_signed_bias_probe import groups_and_weights, objective
from src.world_model.m3w_subset_excess import losses


def test_analytic_probe_uses_the_actual_parent_objective():
    sites=np.array(['a','a','a','b','b','b'])
    rec=np.array(['r']*6);frames=np.array([1,1,2,1,1,2]);known=np.array([True,True,True,True,False,True])
    bank=np.array([[True,True,False],[False,False,True],[True,True,False],
                   [False,True,False],[False,False,True],[False,True,False]])
    rng=np.random.default_rng(22);p=rng.normal(size=(6,2));y=rng.normal(size=(6,2))
    queries,_=groups_and_weights(sites,rec,frames,known,bank)
    def parts(q):
        x=np.ones((len(q),4))*1000
        x[:,1]=20+q[:,0];x[:,3]=20+q[:,1]
        return torch.tensor(x,dtype=torch.float64)
    for aggregate,arm in [(False,'subset_pointwise'),(True,'subset_aggregate')]:
        actual=0
        for ix,_,weight in queries:
            actual+=weight*float(losses(parts(p[ix]),parts(y[ix]),torch.zeros(len(ix),dtype=torch.int64),1,
                                      torch.tensor(bank[ix]))[arm])
        np.testing.assert_allclose(objective(y-p,queries,aggregate),actual,rtol=1e-12,atol=1e-12)
