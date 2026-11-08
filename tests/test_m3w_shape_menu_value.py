import numpy as np
import torch

from src.world_model import m3w_shape_menu_value as m


def test_missing_motion_labels_are_masked_and_shape_output_is_registered():
    t = np.arange(8, dtype=float)
    h = np.stack((np.c_[t, .04*t*t], np.c_[t, -.03*t*t]))
    x, c, _, bank, geom, rot, s = m.features(h, np.full(len(h), .5))
    y = c.copy()
    y[0, -3:] = np.nan
    valid = np.isfinite(y).all(2)
    a, b = m.targets(c, bank, y, valid, .5, rot, s)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert not b[0, -3:].any() and not a[:, 0, 0].any()
    for kind in ("value", "dynamics"):
        net = m.Head(x.shape[1], kind)
        loss = m.objective(net, *(torch.from_numpy(v) for v in (x, a, b, valid)))
        loss.backward()
        assert torch.isfinite(loss)
    assert m.choose(a, geom[:, 5], geom)["learned_dynamics"].tolist() == [5, 5]
