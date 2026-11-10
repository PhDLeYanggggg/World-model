import pytest
import torch

from scripts.run_m3w_opendd_listwise_exchange import padded
from src.world_model.m3w_exchange_listwise import loss


def test_padding_has_no_effect_or_gradient():
    z = torch.tensor([[1., -1., 99.], [-.2, 42., 30.]], requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    y = torch.tensor([[True, False, False], [False, False, False]])
    v = loss(z, y, valid, listwise=True)
    z2 = torch.where(valid, z.detach(), -z.detach())
    assert v.item() == pytest.approx(loss(z2, y, valid, listwise=True).item())
    v.backward()
    assert torch.isfinite(z.grad).all() and torch.count_nonzero(z.grad[~valid]) == 0
    assert z.grad[0, 0] < 0 < z.grad[0, 1]


def test_listwise_term_ignores_homogeneous_slates():
    z = torch.tensor([[1., 3.], [-2., -1.]])
    y = torch.tensor([[False, False], [True, True]])
    m = torch.ones_like(y)
    assert loss(z, y, m, listwise=True).item() == pytest.approx(loss(z, y, m, listwise=False).item())


def test_full_grid_positives_must_be_original_positives():
    x = torch.zeros(3, 2)
    orig = torch.tensor([True, True, False])
    full = torch.tensor([False, True, False])
    _, labels, valid = padded(x, orig, full, [1, 2])
    assert torch.equal(labels[valid], torch.stack((orig, full), -1))
    with pytest.raises(ValueError, match='must also pass'):
        padded(x, orig, ~orig, [1, 2])
