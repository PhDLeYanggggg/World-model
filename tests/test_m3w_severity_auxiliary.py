import inspect
import numpy as np
import pytest
import torch
from src.world_model import m3w_severity_auxiliary as m
from src.world_model import m3w_membership_auxiliary as old
from tests.test_m3w_membership_cost import fixture


def test_weighted_log_loss_optimum_is_harm_ratio_not_prevalence():
    y=torch.tensor([[1.,1.,1.,1.],[1.,9.,0.,0.]],dtype=torch.float64,requires_grad=True)
    e=torch.tensor([1.,0.],dtype=torch.float64,requires_grad=True)
    q=torch.tensor(np.log(.1/.9),dtype=torch.float64,requires_grad=True)
    loss,parts=m.objective(y.detach(),q.expand(2),y,e,torch.ones(4),5.)
    loss.backward(); assert q.grad.item()==pytest.approx(0.,abs=1e-12)
    assert y.grad is None and e.grad is None
    assert parts['severity_BCE']!=parts['membership_BCE']


def test_unit_harm_recovers_ordinary_auxiliary():
    torch.manual_seed(2); y=torch.rand(12,4); y[:,1]=1.; y[:,3]=0
    e=torch.zeros(12); p=torch.rand(12,4); logit=torch.randn(12); scale=torch.ones(4)
    a,_=m.objective(p,logit,y,e,scale,1.); b,_=old.objective(p,logit,y,e,scale,'membership_aux')
    torch.testing.assert_close(a,b)
    for invalid in (0.,-1.,float('nan')):
        with pytest.raises(ValueError): m.objective(p,logit,y,e,scale,invalid)


def test_exact_resume_unknown_exclusion_and_matched_draws(tmp_path):
    torch.set_num_threads(1); x,y,e,s,d,p,c=fixture()
    kw=dict(seed=17,settings=c,identity={'test':1},heartbeat=lambda **_:None)
    a,fit=m.fit(x,y,e,s,d,p,directory=tmp_path/'a',**kw)
    m.fit(x,y,e,s,d,p,directory=tmp_path/'b',stop_at=4,**kw)
    b,_=m.fit(x,y,e,s,d,p,directory=tmp_path/'b',resume=True,**kw)
    for u,v in zip(m.predict(a,x,d,p),m.predict(b,x,d,p)): np.testing.assert_array_equal(u,v)
    old.fit(x,y,e,s,d,p,arm='membership_aux',directory=tmp_path/'old',**kw)
    sa=m.restore(tmp_path/'a')[1]; so=old.restore(tmp_path/'old')[1]
    for k in ('draws','fixed_ids','loss_scales'): np.testing.assert_array_equal(sa[k],so[k])
    assert torch.equal(sa['sampler_rng'],so['sampler_rng']) and fit['unknown_rows_sampled']==0
    assert sa['trace'][0]['cost_loss']==so['trace'][0]['cost_loss']
    assert sa['trace'][0]['membership_BCE']==so['trace'][0]['membership_BCE']
    with pytest.raises(ValueError): m.fit(x,y,e,s,d,p,directory=tmp_path/'a',resume=True,**dict(kw,identity={}))


def test_support_collapses_overlapping_windows_before_mass_ess():
    x,y,e,s,d,p,c=fixture(); records=np.repeat([0,1,2],10); agents=np.ones(len(x),int)
    result=m.support(y,e,p,s,records,agents)
    assert result['strata']['all']['tracks']['positive_groups']==3
    assert result['strata']['all']['tracks']['mass_ESS']<=3
    assert result['strata']['all']['windows']['positive_groups']>3
    assert result['ESS_is_mass_concentration_not_independent_sample_size']


def test_future_weight_is_not_an_inference_argument():
    assert m.predict is old.predict
    assert set(inspect.signature(m.predict).parameters)=={'model','x','env','pr'}
    assert 'target[:, 1].detach()' in inspect.getsource(m.objective)


def test_no_harm_is_not_silent_success(tmp_path):
    x,y,e,s,d,p,c=fixture(); known=np.isfinite(y).all(1); y[known,1]=0; y[known,3]=0
    with pytest.raises(ValueError,match='No harm support'):
        m.fit(x,y,e,s,d,p,seed=1,settings=c,identity={},directory=tmp_path,heartbeat=lambda **_:None)
