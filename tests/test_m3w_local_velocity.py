import numpy as np
import torch

from src.world_model import m3w_local_velocity as m


def geometry():
    g = np.zeros((3, 476), np.float32)
    g[:, :16:2] = np.arange(-7, 1)
    n = g[:, 38:166].reshape(3, 8, 8, 2)
    n[..., 0] = np.arange(-7, 1)
    n[..., 1] = np.arange(1, 9)[None, :, None]
    v = g[:, 230:294].reshape(3, 8, 8)
    v[0] = 1.
    v[1, :5] = 1.
    v[2, :1] = 1.
    return g


def test_disjoint_matched_context_and_no_future():
    g = geometry()
    x, p, mask, _, _, paths = m.inputs(g)
    np.testing.assert_array_equal(mask["near"].sum(1), [4, 2, 0])
    np.testing.assert_array_equal(mask["near"].sum(1), mask["outer"].sum(1))
    assert not (mask["near"] & mask["outer"]).any()
    g[:, 332:] = 42
    xx, pp, _, _, _, _ = m.inputs(g)
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(p, pp)
    np.testing.assert_array_equal(paths["cv"], paths["ols4"])


def test_velocity_bound_and_ego_ignores_peer():
    x, p, mask, b, s, _ = m.inputs(geometry())
    x, p, b, s = map(torch.from_numpy, (x, p, b, s))
    net = m.VelocityModel()
    torch.testing.assert_close(net(x, p, torch.from_numpy(mask["near"]), b, s),
                               torch.zeros(3, 12, 2))
    with torch.no_grad():
        net.readout[-1].bias.fill_(1000.)
        empty = torch.from_numpy(mask["ego"])
        d = net(x, p, empty, b, s)
        torch.testing.assert_close(d, net(x, -p, empty, b, s))
        v = torch.diff(d, dim=1, prepend=torch.zeros_like(d[:, :1]))
        assert (torch.linalg.vector_norm(v, dim=-1) <= .5 * s[:, None] + 1e-5).all()


def test_stepwise_unknown_debit_is_conservative_and_finite():
    r = np.zeros((2, 12, 2))
    p = r.copy()
    p[..., 0] = 1.
    y = p.copy()
    v = np.zeros((2, 12), bool)
    v[0, :9] = True
    y[~v] = np.nan
    s = m.scores(p, r, y, v)
    np.testing.assert_allclose(s["observed"], [1., 0.])
    np.testing.assert_allclose(s["full_grid_lower"], [.5, -1.])
    np.testing.assert_allclose(m.lower(*map(torch.from_numpy, (p, r, y, v))), [-.5, 1.])
