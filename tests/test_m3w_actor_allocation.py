import copy
import itertools
import numpy as np
import pytest
from src.world_model import m3w_actor_allocation as api


def fixture(seed=1,n=3):
    rng=np.random.default_rng(seed);r=rng.uniform(.5,2,(n,7));easy=.3*r
    h=rng.uniform(0,.07,(n,7))*r;he=h*rng.uniform(0,.3,(n,7));p=np.stack((h+.2,h,r,easy,he),-1)
    ok=np.ones((n,7),bool);u=rng.uniform(-.1,1,(n,7))
    return p,ok,u


@pytest.mark.parametrize('seed',range(20))
def test_exact_actor_objective_matches_exhaustive_frozen_budget_solver(seed):
    p,ok,u=fixture(seed);parent=api.independent(p,ok,u);count=int((parent>=0).sum())
    choice,info=api.solve(p,ok,u,parent,count=count,seconds=2.)
    expected=max(api.objective(u,np.array(c)) for c in itertools.product(range(-1,7),repeat=3)
        if sum(k>=0 for k in c)==count and api.feasible(p,ok,u,np.array(c)))
    assert info['optimal'] and api.objective(u,choice)==pytest.approx(expected,rel=1e-10,abs=1e-10)
    assert api.feasible(p,ok,u,choice)


def test_negative_routing_utilities_fall_back_even_with_positive_frozen_gain():
    p,ok,u=fixture();u[:]=-1;parent=api.independent(p,ok,u)
    np.testing.assert_array_equal(parent,np.full(3,-1))
    choice,info=api.solve(p,ok,u,parent,count=0,seconds=2.)
    assert info['reason']=='zero_count_exact' and (choice==-1).all()


def test_allocation_zero_control_counts_budget_and_resume_are_exact():
    p,ok,u=fixture(5,n=6);rec=np.repeat('r',6);frames=np.repeat([0,1],3);ids=np.arange(6)
    z=p[:,:,0]-p[:,:,1];ind=api.independent(p,ok,z);joint=ind.copy()
    for g in api.base.query_groups(rec,frames,ids):joint[g]=api.base.solve_frame(p[g],ok[g],ind[g],exact_count=int((ind[g]>=0).sum()),seconds=2.)[0]
    control=dict(independent=ind,joint=joint);kwargs=dict(seconds=2.)
    direct=api.allocate(p,ok,dict(self_context=u,selected_context=u*.7),rec,frames,ids,control,**kwargs)
    saved=[]
    class Stop(Exception):pass
    def interrupt(s):saved.append(copy.deepcopy(s));raise Stop()
    with pytest.raises(Stop):api.allocate(p,ok,dict(self_context=u,selected_context=u*.7),rec,frames,ids,control,checkpoint=interrupt,**kwargs)
    resumed=api.allocate(p,ok,dict(self_context=u,selected_context=u*.7),rec,frames,ids,control,resume_state=saved[0],**kwargs)
    from src.world_model.m3w_multiaction_cost import exact
    exact(direct,resumed);np.testing.assert_array_equal(direct[0]['zero_joint'],joint)
    for g in api.base.query_groups(rec,frames,ids):
        counts=[int((direct[0][a+'_matched'][g]>=0).sum()) for a in ('zero','self','selected')];assert len(set(counts))==1
    assert not direct[2]['labels_used'] and direct[2]['frozen_moments']


def test_actor_cannot_spend_more_than_frozen_predicted_risk_budget():
    p=np.broadcast_to([1.,.1,1.,.5,.05],(2,7,5)).copy();ok=np.ones((2,7),bool);u=np.ones((2,7))*100
    parent=api.independent(p,ok,u);assert (parent==-1).all()
    choice,_=api.solve(p,ok,u,parent,count=0,seconds=2.);assert (choice==-1).all()


def test_decoder_accepts_no_future_labels():
    p,ok,u=fixture();parent=api.independent(p,ok,u)
    with pytest.raises(TypeError):api.solve(p,ok,u,parent,count=int((parent>=0).sum()),seconds=2.,future_endpoint=np.zeros((3,2)))
