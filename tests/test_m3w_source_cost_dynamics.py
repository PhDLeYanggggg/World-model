import numpy as np
import pytest
import torch

from src.world_model.m3w_source_cost_dynamics import SourceDynamics, restore, dynamics_loss, fit_dynamics, forecast
from scripts.run_m3w_source_cost_dynamics import DynamicsCorpus, trajectory_metrics


def payload():
    torch.manual_seed(3)
    features=(torch.randn(7,6),torch.rand(7,8,3,32,32),torch.ones(7,8,1,32,32))
    frame=(torch.ones(7),torch.eye(2)[None].repeat(7,1,1),torch.ones(7,dtype=torch.bool))
    target=torch.randn(7,12,2)*.1
    return features,frame,target


def test_zero_initialization_matches_floor_and_output_is_bounded():
    x,frame,_=payload();model=SourceDynamics(6)
    assert torch.count_nonzero(model(*x,'past_rgb'))==0
    with torch.no_grad():
        model.head[-1].bias.fill_(100)
    delta=model(*x,'past_rgb')
    assert torch.all(torch.linalg.vector_norm(delta,dim=-1)<1)
    frame[2][0]=False
    assert torch.count_nonzero(restore(delta,*frame)[0])==0
    with pytest.raises(TypeError):
        model(*x,'past_rgb',future_endpoint=torch.ones(7,2))


def test_mask_control_and_unsupported_pixels_do_not_affect_prediction():
    x,_,_=payload();model=SourceDynamics(6)
    with torch.no_grad():
        model.head[-1].weight.fill_(.01)
    x[2][:,:2]=0;changed=x[1].clone();changed[:,:2]=999
    assert torch.equal(model(*x,'past_rgb'),model(x[0],changed,x[2],'past_rgb'))
    assert torch.equal(model(*x,'mask_only'),model(x[0],torch.rand_like(x[1]),x[2],'mask_only'))


def test_restore_matches_numpy_rotation_and_scale():
    from src.world_model.m3w_observed_unit_frame import restore_delta
    local=torch.randn(2,12,2);r=torch.tensor([3.,2.]);q=torch.tensor([[[0.,-1.],[1.,0.]],[[1.,0.],[0.,1.]]]);s=torch.tensor([True,False])
    np.testing.assert_allclose(restore(local,r,q,s).numpy(),restore_delta(local.numpy(),r.numpy(),q.numpy(),s.numpy()),atol=1e-6)


def test_linear_and_log_objectives_have_distinct_error_weighting():
    p=torch.zeros(2,12,2,requires_grad=True);y=torch.zeros_like(p);y[0,:,0]=1;y[1,:,0]=10
    l,a=dynamics_loss(p,y,2.,'ade');assert l.item()==pytest.approx(2.75)
    l.backward();g=p.grad.detach().clone();p.grad.zero_()
    l2,_=dynamics_loss(p,y,2.,'log_ade');l2.backward()
    assert l2.item()==pytest.approx(float((np.log1p(.5)+np.log1p(5))/2))
    assert p.grad[1,0,0].abs()<p.grad[0,0,0].abs()
    assert g[0,0,0]==g[1,0,0]


def test_exact_resume_and_four_way_matched_sampler(tmp_path):
    torch.set_num_threads(2);features,frame,y=payload();ids=np.arange(7);w=np.arange(1,8,dtype=float)/28
    cfg=dict(updates=7,batch_size=3,learning_rate=.001,weight_decay=.001,checkpoint_every=2)
    def inputs(i):return tuple(v[i] for v in features),tuple(v[i] for v in frame)
    def run(name,arm='past_rgb',objective='ade',stop=None):
        torch.manual_seed(17);model=SourceDynamics(6)
        fit=fit_dynamics(model,inputs,lambda i:y[i],ids,w,arm=arm,objective=objective,normalizer=1.,
            seed=17,config=cfg,identity={'v':1},checkpoint=tmp_path/name,heartbeat=lambda **_:None,stop_at=stop)
        return model,fit
    a,_=run('full.pt');run('resume.pt',stop=3);b,result=run('resume.pt');assert result['new_updates']==4
    np.testing.assert_array_equal(forecast(a,inputs,ids,'past_rgb'),forecast(b,inputs,ids,'past_rgb'))
    assert run('resume.pt')[1]['new_updates']==0
    base=torch.load(tmp_path/'full.pt',weights_only=False)
    for arm in ('mask_only','past_rgb'):
        for loss in ('ade','log_ade'):
            name=arm+loss+'.pt';run(name,arm,loss);cp=torch.load(tmp_path/name,weights_only=False)
            np.testing.assert_array_equal(base['draw_counts'],cp['draw_counts'])
            assert torch.equal(base['sampler_rng'],cp['sampler_rng'])
    with pytest.raises(ValueError,match='identity'):
        run('resume.pt',objective='log_ade')


def test_input_and_restoration_frame_do_not_depend_on_future_targets():
    data=object.__new__(DynamicsCorpus);data.nmain=1;data.y=np.zeros(4)
    data.radius=np.array([2,3,4],np.float32);data.rotation=np.repeat(np.eye(2,dtype=np.float32)[None],3,0)
    data.support=np.ones(3,bool);data.target=np.zeros((3,12,2),np.float32);data.allowed=np.array([False,True,False,True])
    def features(ids,training=False):
        if training and not data.allowed[ids].all():raise ValueError('Held')
        return (torch.ones(len(ids),480),torch.zeros(len(ids),8,3,32,32),torch.ones(len(ids),8,1,32,32))
    data.inputs=features
    before=data.dynamics_inputs(np.array([1,2]));data.target[:]=999;after=data.dynamics_inputs(np.array([1,2]))
    assert all(torch.equal(a,b) for x,z in zip(before,after) for a,b in zip(x,z))
    with pytest.raises(ValueError,match='Held'):
        data.loss_targets(np.array([2]))
    with pytest.raises(ValueError,match='Source-only'):
        data.dynamics_inputs(np.array([0]))


def test_zero_error_easy_percentage_is_not_invented():
    target=np.zeros((3,12,2));target[2,:,0]=2
    p=np.zeros_like(target);p[0,:,0]=.1
    m=trajectory_metrics(p,target,np.ones(3),np.ones(3,bool),1.)
    assert m['easy_rows']==2 and m['easy_percentage_degradation'] is None
    assert m['easy_absolute_harm']==pytest.approx(.05)
    assert m['easy_nonzero_predictions']==1
    with pytest.raises(ValueError):
        trajectory_metrics(p,target,np.zeros(3),np.ones(3,bool),1.)
