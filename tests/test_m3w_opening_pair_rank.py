import numpy as np
import torch

from src.world_model import m3w_opening_pair_rank as model


def test_pair_loss_penalizes_wrong_within_query_order():
    right = torch.tensor([1.,-1.,2.,-2.])
    wrong = torch.tensor([-1.,1.,-2.,2.])
    sizes = [(1,1),(1,1)]
    increment = lambda v: model.objective(v,sizes,1.)-model.objective(v,sizes,0.)
    assert increment(right)<increment(wrong)


def test_sampling_is_replayable_and_preserves_positive_and_negative_identity():
    p = np.array([0,1])
    n = np.arange(2,21)
    h = n[:8]
    pools = [[(p,n,h),(p[:0],n,h)]]
    a,b = (model.batch_plan(pools,17,steps=2) for _ in range(2))
    for q,r in zip(a,b):
        for (i,j),(ii,jj) in zip(q,r):
            np.testing.assert_array_equal(i,ii)
            np.testing.assert_array_equal(j,jj)
            assert set(i)<=set(p) and set(j)<=set(n) and set(h)<=set(j)


def test_optimizer_checkpoint_resume_matches_continuous_updates(tmp_path):
    x = np.arange(20,dtype=np.float32).reshape(5,4)/20
    plan = [[(np.array([1]),np.array([2,3,4]))]]*4
    a = tmp_path/'resume.pt'
    b = tmp_path/'continuous.pt'
    model.train(x,plan[:2],17,.5,a,lambda **_:None)
    resumed = model.train(x,plan,17,.5,a,lambda **_:None)
    full = model.train(x,plan,17,.5,b,lambda **_:None)
    for v,w in zip(resumed.parameters(),full.parameters()):
        assert torch.equal(v,w)
