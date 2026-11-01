import copy
import numpy as np
import pytest
from src.world_model import m3w_refit_cost_allocation as api


def inputs():
    rng=np.random.default_rng(20261101);p=np.zeros((6,7,5));p[:,:,0]=rng.uniform(.1,2,(6,7));p[:,:,2:4]=10
    q=p.copy();q[:,:,2:4]=20;q[:,:,0]=p[:,:,0][:,::-1]
    return dict(predictions=dict(locked_references=p,learned_references=q),allowed=np.ones((6,7),bool),
        recordings=np.repeat('r',6),frames=np.repeat([1,2],3),ids=np.arange(6),parent_joint=np.full(6,-1,int),seconds=2.)


def test_allocated_native_policies_keep_each_model_budget_and_match_counts():
    kw=inputs();choices,_,info=api.allocate(**kw)
    assert info['same_count_controls_every_query'] and not info['labels_used']
    for g in api.base.query_groups(kw['recordings'],kw['frames'],kw['ids']):
        for prefix,arm in zip(('locked','learned'),api.ARMS):
            for mode in ('independent','joint','matched'):assert api.base.feasible(kw['predictions'][arm][g],kw['allowed'][g],choices[prefix+'_'+mode][g])
        assert (choices['locked_matched'][g]>=0).sum()==(choices['learned_matched'][g]>=0).sum()


def test_actual_query_checkpoint_resume_exact_and_future_keys_rejected():
    kw=inputs();saved=[]
    direct=api.allocate(**kw)
    class Stop(Exception):pass
    def stop(v):saved.append(copy.deepcopy(v));raise Stop()
    with pytest.raises(Stop):api.allocate(**kw,checkpoint=stop)
    resumed=api.allocate(**kw,resume_state=saved[0])
    for a in direct[0]:np.testing.assert_array_equal(direct[0][a],resumed[0][a])
    assert direct[2]==resumed[2]
    bad=copy.deepcopy(kw);bad['predictions']['future_endpoint']=np.zeros((6,2))
    with pytest.raises(ValueError):api.allocate(**bad)
