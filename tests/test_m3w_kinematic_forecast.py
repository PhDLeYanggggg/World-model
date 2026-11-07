import torch

from src.world_model.m3w_kinematic_forecast import KinematicForecaster


def geometry():
    rng = torch.Generator().manual_seed(413)
    return dict(history=torch.randn(4, 8, 3, generator=rng),
                neighbors=torch.randn(4, 8, 8, 3, generator=rng),
                neighbor_mask=torch.rand(4, 8, 1, generator=rng).gt(.3).expand(-1, -1, 8))


def test_rigid_motion_and_peer_permutation_equivariance():
    m = KinematicForecaster(True)
    x = geometry()
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    shift = torch.tensor([10., -2.])
    y = {k: v.clone() for k, v in x.items()}
    for key in ('history', 'neighbors'):
        y[key][..., :2] = x[key][..., :2] @ rotation + shift
    y['neighbors'] = y['neighbors'].flip(1)
    y['neighbor_mask'] = y['neighbor_mask'].flip(1)
    torch.testing.assert_close(m(y), m(x) @ rotation)


def test_absent_or_disabled_peers_cannot_influence_forecast():
    a, b = KinematicForecaster(False), KinematicForecaster(True)
    b.load_state_dict(a.state_dict())
    x = geometry()
    y = dict(x, neighbor_mask=torch.zeros_like(x['neighbor_mask']))
    torch.testing.assert_close(a(x), b(y), rtol=0., atol=0.)
    torch.testing.assert_close(a(x), a(dict(x, neighbors=x['neighbors'] * 12)), rtol=0., atol=0.)


def test_output_has_bounded_step_speed_and_finite_gradient():
    model, x = KinematicForecaster(True), geometry()
    p = model(x)
    assert p.shape == (4, 12, 2)
    v = torch.cat((p[:, :1], p.diff(dim=1)), dim=1)
    max_speed = x['history'][..., :2].diff(dim=1).norm(dim=-1).amax(1)
    assert (v.norm(dim=-1) <= 2 * max_speed[:, None] + 1.e-6).all()
    p.square().mean().backward()
    assert all(x.grad is not None and x.grad.isfinite().all() for x in model.parameters())
    still = dict(x, history=x['history'][:, -1:].expand(-1, 8, -1))
    torch.testing.assert_close(model(still), torch.zeros_like(p))
