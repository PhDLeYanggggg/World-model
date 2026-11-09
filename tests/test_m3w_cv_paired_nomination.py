import numpy as np
import torch

from scripts import run_m3w_cv_paired_nomination as study


def test_pair_label_debits_missing_steps_and_is_antisymmetric():
    a = np.zeros((3,2,2))
    b = a.copy()
    b[:,0,0] = 2.
    b[:,1,0] = [1.,2.,4.]
    y = a.copy()
    y[:,1] = np.nan
    valid = np.tile([True,False],(3,1))
    scale = np.ones(3)
    preference = study.signed_preference(a,b,y,valid,scale)
    np.testing.assert_array_equal(preference,[.5,0.,0.])
    np.testing.assert_array_equal(study.signed_preference(b,a,y,valid,scale),-preference)


def test_ambiguous_pair_adds_no_preference_gradient():
    pred = torch.zeros((1,2,2),requires_grad=True)
    f = torch.nn.Identity()
    loss = study.objective(f,pred,torch.zeros_like(pred),torch.zeros(1),.25)
    loss.backward()
    assert torch.count_nonzero(pred.grad) == 0
    study.objective(f,pred,torch.zeros_like(pred),torch.ones(1),.25).backward()
    assert (pred.grad[:,0] < 0.).all() and (pred.grad[:,1] > 0.).all()


def test_rank_cannot_expand_frozen_eligibility():
    rank = np.array([[10.,-1.,20.]])
    frozen = (np.array([[1.,1.,-1.]]),np.array([[[1.,0.],[0.,0.],[0.,0.]]]),
              np.ones((1,3)),np.ones((1,3),bool))
    action, *_ = study.nomination(rank,frozen,(0.,.1))
    np.testing.assert_array_equal(action,[1])
