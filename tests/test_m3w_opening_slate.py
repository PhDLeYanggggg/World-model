import numpy as np
import torch

from src.world_model import m3w_opening_slate as m


def test_diverse_shortlists_respect_size_and_promote_new_groups():
    score = np.array([10., 9., 8., 7., 6.])
    rows = np.array([0, 0, 1, 1, 2])
    displacement = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [-1., 0.]])
    np.testing.assert_array_equal(m.shortlist(score, rows, displacement, 3, "flat"), [0, 1, 2])
    np.testing.assert_array_equal(m.shortlist(score, rows, displacement, 3, "row"), [0, 2, 4])
    np.testing.assert_array_equal(m.shortlist(score, rows, displacement, 3, "direction"), [0, 3, 4])


def test_listwise_loss_differentiable_and_permutation_invariant():
    x, y = torch.randn(3, 6, 5), torch.randn(3, 6)
    model = m.network(5)
    actual = m.listwise(model, x, y)
    order = torch.randperm(6)
    torch.testing.assert_close(actual, m.listwise(model, x[:, order], y[:, order]))
    actual.backward()
    assert torch.isfinite(actual) and all(p.grad is not None for p in model.parameters())
