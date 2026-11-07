import torch

from src.world_model.m3w_relative_path_supervision import relative_path_loss


def test_relative_loss_uses_intersection_and_invariant_common_displacement():
    a = torch.tensor([[[3., 0.], [99., 9.]]], requires_grad=True)
    b = torch.tensor([[[1., 0.], [0., 0.]]], requires_grad=True)
    y = torch.tensor([[[0., 0.], [float('nan'), float('nan')]]])
    keep = torch.tensor([[True, False]])
    val = relative_path_loss(a, b, y, y, keep, ~torch.zeros_like(keep), torch.tensor([2.]))
    torch.testing.assert_close(val, torch.tensor(1.))
    torch.testing.assert_close(val, relative_path_loss(a+100, b+100, y, y, keep, keep, torch.tensor([2.])))
    val.backward()
    torch.testing.assert_close(a.grad, torch.tensor([[[.5, 0.], [0., 0.]]]))
    torch.testing.assert_close(b.grad, -a.grad)


def test_empty_pair_support_has_zero_loss_and_gradient():
    p = torch.ones(2, 3, 2, requires_grad=True)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    loss = relative_path_loss(p, 2*p, 10*p.detach(), p.detach(), mask, ~mask, torch.ones(2))
    assert loss.item() == 0.
    loss.backward()
    torch.testing.assert_close(p.grad, torch.zeros_like(p))
