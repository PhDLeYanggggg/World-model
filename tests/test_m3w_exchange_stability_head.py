import torch

from src.world_model.m3w_exchange_stability_head import loss, rank


def test_stability_target_requires_full_pass_and_support():
    z = torch.zeros(4, 8, requires_grad=True)
    loss(z, torch.ones(4, dtype=torch.long), torch.zeros(4), torch.tensor([2, 3, 3, 3]),
         torch.tensor([1, -1, 0, 1]), True).backward()
    assert z.grad[:, 7].tolist() == [0., 0., .25, -.25]


def test_stability_reranking_uses_a_conditional_score():
    scores = torch.zeros(2, 8)
    scores[0, 4:7] = 1.
    scores[0, 7] = -8.
    scores[1, 7] = 8.
    assert rank(scores, robust=False) == 0
    assert rank(scores, robust=True) == 1
