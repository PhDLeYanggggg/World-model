import numpy as np
import torch

from src.evaluation.m3w_opendd_query_rank import limited_queries, restrict, training_loss


def test_rank_loss_pushes_correct_edit_and_ignores_illegal_options():
    score = torch.zeros((1,3), requires_grad=True)
    legal = torch.tensor([[True, True, False]])
    target = torch.tensor([[0., 1., 4.]])
    training_loss(score, legal, target, target).backward()
    assert score.grad[0,0] > 0 and score.grad[0,1] < 0 and score.grad[0,2] == 0


def test_factorized_restrictions_preserve_queries_and_fixed_coverage():
    legal = np.array([[1,1],[1,1],[1,0],[1,1]],dtype=bool)
    scores = np.array([[1.,0.],[2.,3.],[5.,8.],[4.,0.]])
    offsets = [0,2,4]
    rec = restrict(legal,scores,offsets,'recipient')
    assert rec.tolist() == [[1,0],[0,1],[1,0],[1,0]]
    cand = restrict(legal,scores,offsets,'candidate')
    assert cand.tolist() == [[0,0],[1,1],[1,0],[0,0]]
    assert limited_queries(legal,scores,offsets,np.array([3.,1.])).tolist() == [True,False]
