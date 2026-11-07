import numpy as np
import torch

from src.world_model.m3w_early_reference_listwise import batch, loss, sample_plan


def test_weighting_and_padding_do_not_create_candidates():
    z = torch.tensor([[2., 0., 100.], [0., 2., 100.]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    pos = torch.tensor([[True, False, False], [True, False, False]])
    early = torch.tensor([1., 0.])
    assert loss(z, mask, pos, early, 8.) < loss(z, mask, pos, early, 1.)
    z[:, -1] = -100.
    torch.testing.assert_close(loss(z, mask, pos, early, 1.), torch.tensor(1.126928))


def test_matched_plan_and_multiple_positives():
    a = (np.ones((3, 2)), np.array([True, True, False]), True)
    b = (np.ones((2, 2)), np.array([True, False]), False)
    sources = [[[a, b]], [[b]]]
    assert sample_plan(sources, 2) == sample_plan(sources, 2)
    x, m, p, e = batch([a, b])
    assert x.shape == (2, 3, 2) and p.sum() == 3 and m.sum() == 5 and e.sum() == 1
