import torch

from src.world_model.m3w_completion_motion_blend import CompletionMotionBlend, objectives


def test_completion_debits_use_exact_missing_support():
    ref = torch.zeros(2, 2, 2)
    pred = ref.clone()
    pred[0, :, 0] = torch.tensor([2., 3.])
    pred[1, :, 0] = torch.tensor([1., 2.])
    valid = torch.tensor([[True, False], [False, False]])
    y = ref.clone()
    y[~valid] = float('nan')
    l, r = torch.tensor([0]), torch.tensor([1])
    a, b, c, d = objectives(pred, ref, y, valid, torch.ones(2), l, r)
    for actual, expected in zip((a, b, c, d), (1., 0., 1., 1.)):
        torch.testing.assert_close(actual, torch.tensor(expected))
    valid[1, 0] = True
    y[1, 0] = 0.
    a, b, c, d = objectives(pred, ref, y, valid, torch.ones(2), l, r)
    for actual, expected in zip((a, b, c, d), (1.5, .5, 0., .5)):
        torch.testing.assert_close(actual, torch.tensor(expected))


def test_initial_blend_and_rotation_invariance():
    net = CompletionMotionBlend()
    h = torch.randn(3, 8, 2)
    a, b = torch.randn(3, 12, 2), torch.randn(3, 12, 2)
    torch.testing.assert_close(net(h, a, b), torch.full((3, 1), .5))
    net.head[-1].weight.data.normal_()
    rot = torch.tensor([[0., -1.], [1., 0.]])
    torch.testing.assert_close(net(h@rot+5, a@rot, b@rot), net(h, a, b), atol=1e-6, rtol=1e-6)
