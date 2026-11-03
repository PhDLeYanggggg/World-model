import numpy as np
import pytest
import torch

from scripts.run_m3w_bound_aware_cost import job_designs
from src.world_model import m3w_bound_aware_cost as api
from src.world_model.m3w_selection_cardinal_cost import objective as previous_objective


def test_nested_design_keeps_every_site_out_of_both_fits():
    names=np.array([f's{i}' for i in range(12)])
    src=dict(sites=np.repeat(names,4),recordings=np.repeat(np.arange(12),4),frames=np.tile(np.arange(4),12))
    producers=[dict(trial=f'single{f}_seed{s}',seed=s,fold=f,fit_sites=names[f*4:f*4+4].tolist(),
        baseline_index=1,easy_cut=2.,hard_cut=5.) for f in range(3) for s in (17,29,43)]
    cfg=dict(training=dict(query_batch_size=32,minimum_updates=8,largest_source_cycles=4))
    jobs=job_designs(producers,src,np.arange(48),cfg)
    assert len(jobs)==18
    for job in jobs:
        a,b,c=map(set,(job['producer_sites'],job['critic_sites'],job['evaluation_sites']))
        assert len(a)==len(b)==len(c)==4 and not(a&b or a&c or b&c)
    assert all(sum(site in job['evaluation_sites'] for job in jobs)==6 for site in names)
    for p in producers:
        if p['fold']==1:p['fit_sites']=names[:4].tolist()
    with pytest.raises(ValueError):job_designs(producers,src,np.arange(48),cfg)


def test_known_control_objective_equals_previous_absolute_cost_objective():
    rng=np.random.default_rng(4)
    p=torch.from_numpy(rng.random((6,7,5)).astype(np.float32))
    y=torch.from_numpy(rng.random((6,7,5)).astype(np.float32));known=np.array([True,False,True,True,True,False])
    y[~known]=0
    data=dict(causal=dict(allowed=np.ones((6,7),bool)),known=known,observed_future=known,
        defined_by_arm=dict(known_only=known),truth=y,truth_by_arm=dict(known_only=y),
        preprocess=dict(rms=np.array([1.,2.,3.,4.,5.],np.float32)))
    ix=np.arange(6);segments=torch.tensor([0,0,0,1,1,1])
    a=api.objective(p,data,ix,segments,2,'known_only')
    b=previous_objective(p,data,ix,segments,2,'mean_absolute')
    torch.testing.assert_close(a['total'],b['total'],rtol=0,atol=0)


def test_interpolation_endpoints_remain_exact_not_roundoff_interventions():
    rng=np.random.default_rng(2);b=rng.normal(size=(3,12,2))*1e9;p=rng.normal(size=(3,12,2))
    a=api.prediction(b,p,np.array([0.,1.,.25]))
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],p[1])
