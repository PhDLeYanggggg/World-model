import copy
import numpy as np
import pytest
import torch
from src.evaluation.m3w_severity_transport import concentration, grouped_mass, error_accounting, radial_support, gradient_concentration
from src.world_model.m3w_membership_auxiliary import AuxiliaryCostHead


def test_signed_excess_keeps_cancellation_and_groups_before_concentration():
    new=np.array([3.,0.,2.,99.]); old=np.array([1.,2.,2.,0.]); target=np.array([0.,0.,0.,np.nan])
    r=error_accounting(new,old,target,np.array(['a','a','b','c']),np.array([1,1,1,1]),
                       np.array([1,0,1,np.nan]),np.array([True,False,False,False]))
    assert r['unknown_rows']==1
    assert r['partitions']['all']['positive_excess_mass']==8
    assert r['partitions']['all']['negative_excess_mass']==4
    assert r['partitions']['all']['signed_excess_MSE']==pytest.approx(4/3)
    assert r['groups']['track']['positive_row_excess']['positive_groups']==1
    assert r['partitions']['radial_outside']['positive_excess_share']==1


def test_zero_mass_is_undefined_not_independent_ess():
    r=concentration(np.zeros(3)); assert r['mass_ESS'] is None and r['top1_share'] is None
    assert concentration(grouped_mass(np.array([1.,2.,1.]),np.array(['a','a','b'])))['mass_ESS']==1.6
    with pytest.raises(ValueError): concentration([-1,1])


def test_radial_cut_does_not_depend_on_held_features_or_labels():
    x=np.arange(20,dtype=float)[:,None]; known=np.ones(20,bool)
    pr=dict(mean=np.array([0.]),std=np.array([1.]),known=known,weights=np.ones(20)/20,training_sites=['a'])
    mask,a=radial_support(x,np.array([[1.],[100.]]),pr,np.array(['a']*20),'b')
    _,b=radial_support(x,np.array([[1e10]]),pr,np.array(['a']*20),'b')
    assert a['training_radius_q95']==b['training_radius_q95'] and mask.tolist()==[False,True]
    with pytest.raises(ValueError): radial_support(x,x,pr,np.array(['a']*20),'a')


def test_unknown_fitting_features_do_not_set_radius_cut():
    x=np.array([[0.],[1.],[1000.]])
    pr=dict(mean=np.array([0.]),std=np.array([1.]),known=np.array([True,True,False]),
            weights=np.array([.5,.5,0.]),training_sites=['a'])
    _,r=radial_support(x,np.array([[2.]]),pr,np.array(['a']*3),'b')
    assert r['training_radius_q95']==1.


def test_group_gradients_add_and_do_not_update_model_or_rng():
    torch.manual_seed(17); model=AuxiliaryCostHead(3,4); before=copy.deepcopy(model.state_dict())
    z=torch.randn(8,3); env=torch.ones(8); y=torch.rand(8,4); easy=torch.arange(8)%2
    batch=(z,env,y,easy.float(),torch.ones(4)); rng=torch.get_rng_state().clone()
    r=gradient_concentration(model,batch,np.array(['a','a','b','b','c','c','c','c']),.5)
    for k in ('ordinary_BCE','weighted_BCE','easy_harm_cost'):
        assert r[k]['group_gradient_additivity_pass'] and r[k]['projection_sum']==pytest.approx(1.,abs=1e-5)
        assert r[k]['recording_gradient_norm_mass']['groups']==3
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in model.state_dict().items(): assert torch.equal(v,before[k])
    assert all(p.grad is None for p in model.parameters())


def test_unit_harm_weights_reproduce_ordinary_gradients():
    torch.manual_seed(3); model=AuxiliaryCostHead(2,3); y=torch.ones(4,4)
    r=gradient_concentration(model,(torch.randn(4,2),torch.ones(4),y,torch.zeros(4),torch.ones(4)),np.array([0,0,1,1]),1.)
    assert r['ordinary_BCE']==r['weighted_BCE']


def test_descriptive_flags_never_claim_causation_or_deployment():
    from scripts.run_m3w_european_severity_transport import summarize
    e=error_accounting(np.array([3.,1.]),np.zeros(2),np.zeros(2),np.array([0,1]),
                       np.array([0,0]),np.ones(2),np.array([True,False]))
    gradients={k:dict(recording_gradient_norm_mass=dict(top1_share=.6)) for k in
               ('ordinary_BCE','weighted_BCE','easy_harm_cost')}
    record=dict(gradients=gradients,error_accounting={c:dict(positive_disagreement=e) for c in ('original','ordinary_aux')})
    cfg=dict(pairs=['full','motion_only'],comparators=['original','ordinary_aux'])
    rows=[dict(pair=p,records=[record]) for p in cfg['pairs']]
    d=summarize(rows,cfg)['decision']
    assert d['recording_influence_investigation_motivated'] and d['radial_association_flag']
    assert not any(d[k] for k in ('causal_root_cause_proven','method_improvement_proven','deployment_changed'))
