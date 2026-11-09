import numpy as np
import torch

from scripts import run_m3w_opendd_matched_spline as study
from src.world_model import m3w_opendd_spline_frames as model


def test_comoving_restores_common_velocity_boost():
    rng = np.random.default_rng(17)
    h = rng.normal(size=(4, 8, 2)).cumsum(1)
    t = np.arange(-7., 1.)[None, :, None]
    velocity = np.array([.4, -.2])
    a, _ = model.designs(h)
    b, _ = model.designs(h+t*velocity+6.)
    net = model.Spline()
    with torch.no_grad():
        net.head[-1].bias[:] = .2
        pa = net(*(torch.from_numpy(v) for v in a['comoving_spline']), False).numpy()
        pb = net(*(torch.from_numpy(v) for v in b['comoving_spline']), False).numpy()
    np.testing.assert_allclose(pb-pa, np.broadcast_to(np.arange(1., 13.)[:, None]*velocity, pa.shape),
                               atol=2e-6)


def test_draw_matches_eqmotion_rows_with_missing_outcomes():
    a, b = [], {}
    for site in ('a', 'b'):
        h = np.random.default_rng(ord(site)).normal(size=(6, 8, 2))
        y = torch.arange(6.)[:, None, None].repeat(1, 12, 2) + ord(site)
        valid = torch.ones(6, 12, dtype=torch.bool)
        valid[:2] = False
        x, scale = model.designs(h)
        b[site] = dict(inputs={k: tuple(torch.from_numpy(z) for z in v) for k, v in x.items()},
                       scale=torch.from_numpy(scale), target=y, valid=valid)
        a.append((None, None, None, (y, y, y, y), y, valid))
    new = study.sampler(b, ('a', 'b'), 'comoving_spline')
    old = study.control.batcher(a)
    r, q = np.random.default_rng(12), np.random.default_rng(12)
    for _ in range(4):
        *_, y, v = new(r)
        _, y0, v0 = old(q)
        torch.testing.assert_close(y, y0)
        torch.testing.assert_close(v, v0)
        assert r.bit_generator.state == q.bit_generator.state
