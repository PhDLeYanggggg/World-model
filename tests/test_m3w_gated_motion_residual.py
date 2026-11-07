import numpy as np
import torch

from src.world_model.m3w_gated_motion_residual import GatedMotionResidual


def test_equal_initialization_causal_equivariance_and_static():
    a, b = GatedMotionResidual(False), GatedMotionResidual(True)
    b.load_state_dict(a.state_dict())
    t = torch.arange(8.)[None, :, None]
    h = torch.cat((t, .1*t.square()), -1)
    rot = torch.tensor([[0., -1.], [1., 0.]])
    for m in (a, b):
        y = m({'history': h})
        assert y.shape == (1, 12, 2)
        torch.testing.assert_close(m({'history': h @ rot+20.}), y @ rot, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(m({'history': h*0.+20.}), torch.zeros_like(y))
    assert not torch.equal(a({'history': h}), b({'history': h}))


def test_zero_residual_is_causal_four_point_least_squares():
    m = GatedMotionResidual(True)
    for p in m.parameters():
        p.data.zero_()
    x = np.arange(8, dtype=np.float32)**2
    history = torch.from_numpy(np.stack((x, x+3.), axis=-1)[None])
    v = np.polyfit(np.arange(4), x[-4:], 1)[0]
    pred = np.arange(1, 13)[:, None]*np.array([[v, v]])
    np.testing.assert_allclose(m({'history': history}).detach().numpy()[0], pred, atol=1e-5)
