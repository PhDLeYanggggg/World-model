import torch

from src.world_model.m3w_reference_bridge import Forecast, loss


def case():
    hist = torch.zeros((2, 7, 6))
    peers = torch.zeros((2, 4, 6))
    mask = torch.zeros((2, 4))
    frame = torch.eye(2).expand(2, -1, -1)
    scale = torch.ones(2)
    velocity = torch.zeros((2, 2))
    cv = torch.zeros((2, 12, 2))
    ols = torch.ones_like(cv)
    lead = torch.linspace(1., 0., 12)
    return hist, peers, mask, frame, scale, velocity, cv, ols, lead


def test_bridge_has_no_extra_dual_reference_disagreement():
    args = case()
    cv, ols = args[-3:-1]
    net = Forecast()
    with torch.no_grad():
        net.motion.bias.fill_(.7)
    p, _ = net(*args, 'bridge')
    d = lambda x: torch.linalg.vector_norm(x, dim=-1)
    torch.testing.assert_close(d(p-cv)+d(p-ols), d(ols-cv))


def test_support_gate_not_trained_to_game_trajectory_gradient():
    args = case()
    net = Forecast()
    with torch.no_grad():
        net.motion.bias.fill_(.7)
    p, logits = net(*args, 'causal_support')
    assert torch.autograd.grad(p.sum(), logits, allow_unused=True, retain_graph=True)[0] is None
    valid = torch.ones((2, 12), dtype=torch.bool)
    valid[0] = False
    cv, ols = args[-3:-1]
    l = loss(p, logits, cv, ols, torch.where(valid[..., None], ols, float('nan')), valid, torch.ones(2), 2.)
    assert l.isfinite()
    l.backward()
    assert net.support.bias.grad.abs().sum() > 0.
