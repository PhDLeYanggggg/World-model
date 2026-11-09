import numpy as np
import torch

from scripts.audit_m3w_cv_budget_certificates import calculate
from scripts.run_m3w_cv_minimum_budget import plan
from src.world_model import m3w_cv_minimum_budget as model


def paths(x):
    out = np.zeros((len(x),12,2))
    out[:,:,0] = np.asarray(x)[:,None]
    return out


def test_count_solver_verifies_and_reduces_inherited_edits():
    cv,ols,y = paths([1]*4),paths([.95]*4),paths([0]*4)
    bank = paths([.5]*4)[:,None]
    q,v = np.repeat([0,1],2),np.ones((4,12),bool)
    result = plan(cv,ols,bank,y,v,q,(2.,4.),[[i,0] for i in range(4)])
    assert result['lower']==result['incumbent']['edits']==1
    assert min(result['incumbent']['gain_pp'])>0.
    reduced = calculate(cv,ols,bank,y,v,q,(2.,4.),[[i,0] for i in range(4)])
    assert reduced['lower']==reduced['upper']==1


def test_guard_and_exact_counts_are_distinct():
    pred = np.zeros((4,2,6))
    pred[...,0:2] = 1.
    pred[0,:,2:5] = 1.
    pred[0,:,0:2] = 5.
    q = [np.arange(4)]
    scale = np.ones(4)
    legal = np.ones((4,2),bool)
    args = pred,legal,q,scale,.5,1.,False
    assert (model.choose(*args)>=0).sum()==2
    assert (model.choose(*args,guard=True)>=0).sum()==0
    assert (model.choose(*args,match=np.array([-1,-1,1,-1]),guard=True)>=0).sum()==1


def test_teacher_loss_never_uses_targets_outside_verified_recordings():
    net = torch.nn.Linear(3,6)
    x = torch.randn(4,3)
    cost = torch.zeros((4,5))
    flag = torch.tensor([True,False,False,False])
    labels = torch.tensor([True,False,True,True])
    altered = torch.tensor([True,True,False,False])
    torch.testing.assert_close(model.objective(net,x,cost,labels,flag,True),
                               model.objective(net,x,cost,altered,flag,True))


def test_context_uses_only_its_query():
    q = np.array([1,1,2])
    x = np.array([[1.,2.],[3.,4.],[5.,6.]])
    base = model.query_context(x,q)
    x[-1] *= 100
    np.testing.assert_array_equal(model.query_context(x,q)[:2],base[:2])
