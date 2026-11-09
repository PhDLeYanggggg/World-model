import torch

from src.world_model import m3w_av2_harm_quantile as model


def test_tail_is_one_sided_and_does_not_backpropagate_into_point_estimate():
    net = model.Harm(6)
    x, ctx = torch.randn(6, 6), torch.randn(3, 21)
    opts, row, u = torch.tensor([[0, 2], [1, 4]]), torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([1., 2., 3.])
    point, tail = net(x, ctx, opts, row, u)
    assert (tail >= point).all()
    model.quantile_loss(tail, torch.zeros_like(tail)).backward()
    assert all(v.grad is None for v in (*net.point.parameters(), *net.cap.parameters()))
    assert any(v.grad is not None for v in net.margin.parameters())


def test_tail_loss_penalizes_underprediction_more():
    y = torch.tensor([0.])
    assert model.quantile_loss(y-1., y) > model.quantile_loss(y+1., y)


def test_pair_margins_invariant_to_slot_order():
    net = model.Harm(6)
    x, ctx = torch.randn(3, 6), torch.randn(3, 21)
    opts, row, u = torch.tensor([[0, 1], [1, 2]]), torch.arange(3), torch.tensor([1., 2., 3.])
    for a, b in zip(net(x, ctx, opts, row, u), net(x, ctx, opts.flip(1), row, u), strict=True):
        torch.testing.assert_close(a, b)
