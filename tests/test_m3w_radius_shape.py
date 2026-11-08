import numpy as np
import pytest
import torch

from src.world_model.m3w_radius_shape import at_radius, neighbors, shape


def test_shape_keeps_exact_budget_with_finite_gradient_and_identity_start():
    unit = torch.zeros(4, 12, 2)
    unit[:, -1, 0] = 1.
    assert torch.equal(shape(torch.zeros(4, 6), unit), unit)
    raw = torch.tensor([[-100., 100.]*3]*4, requires_grad=True)
    predicted = shape(raw, unit)
    torch.testing.assert_close(torch.linalg.vector_norm(predicted, dim=-1).amax(1), torch.ones(4))
    predicted.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_world_radius_keeps_zero_rows_unchanged():
    paths = np.zeros((2, 12, 2))
    paths[0, 3] = [3., 4.]
    out = at_radius(paths, np.array([2.5, 0.]))
    np.testing.assert_array_equal(out[0, 3], [1.5, 2.])
    np.testing.assert_array_equal(out[1], 0.)
    with pytest.raises(ValueError, match='No direction'):
        at_radius(paths, np.array([2.5, 1.]))


def test_neighbor_history_uses_current_separation_and_respects_queries():
    h = np.zeros((5, 8, 2))
    h[:, -1, 0] = [0., 3., 1., .01, 4.]
    context = neighbors(h, [0, 3, 5])
    np.testing.assert_array_equal(context[0, 0], h[[2, 1]])
    np.testing.assert_array_equal(context[3, 0, 0], h[4])
    assert np.isnan(context[3, 0, 1]).all()
    assert np.isnan(context[:, 1]).all()
