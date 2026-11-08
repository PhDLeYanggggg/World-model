import numpy as np
import torch

from src.world_model.m3w_recipient_top2 import choose, top2_probability


def test_exact_top2_inclusion_matches_ordered_pair_enumeration():
    z = torch.tensor([[1.1, -.4, 2.0, 8.]], dtype=torch.double, requires_grad=True)
    valid = torch.tensor([[True, True, True, False]])
    p = top2_probability(z, valid)
    w = np.exp(z.detach().numpy()[0, :3])
    expected = np.zeros(4)
    for i in range(3):
        for j in range(3):
            if i != j:
                weight = (w[i] / w.sum()) * (w[j] / (w.sum() - w[i]))
                expected[[i, j]] += weight
    np.testing.assert_allclose(p.detach().numpy()[0], expected)
    (p @ torch.tensor([1., 2., 3., 4.], dtype=torch.double)).backward()
    assert torch.isfinite(z.grad).all()
    assert z.grad[0, 3] == 0


def test_small_or_padded_queries_and_stable_counts():
    z = torch.tensor([[12., -12., 3.], [1., 0., 3.], [0., 0., 1.]], requires_grad=True)
    valid = torch.tensor([[True, True, False], [False, False, True], [False, False, False]])
    p = top2_probability(z, valid)
    torch.testing.assert_close(p, valid.float())
    p.sum().backward()
    assert torch.isfinite(z.grad).all()
    picked = choose(np.ones(5), np.array([1, 0, 1, 1, 0], bool),
                    [np.arange(4), np.array([4])])
    np.testing.assert_array_equal(picked, [True, False, True, False, False])
