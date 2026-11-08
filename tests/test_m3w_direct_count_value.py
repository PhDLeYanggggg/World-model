import numpy as np
import torch

from src.world_model import m3w_direct_count_value as m


def test_relative_peer_histories_ignore_common_rigid_transform_and_mask_empty_slot():
    t = np.arange(8, dtype=float)
    h = np.stack([np.c_[t, .2*t], np.c_[.7*t, 1+.2*t], np.c_[t, 6+t]])
    row = np.array([[0, -1]])
    groups = [(slice(0, 3), 3)]
    out = m.contexts(h, row, groups)
    rot = np.array([[0., 1.], [-1., 0.]])
    np.testing.assert_allclose(out, m.contexts(h@rot+[34., 56.], row, groups), atol=1e-6)
    assert out[0, 16] == out[0, 33] == 1.
    assert not out[0, 34:].any()


def test_same_menu_yields_exact_zero_for_identity_and_risk_penalizes_harm():
    cv = np.zeros((2, 2, 2))
    b = np.ones((2, 1, 2, 2))
    ols = b[:, 0]
    y = np.zeros_like(cv)
    y[0] = .1
    t = m.targets(cv, ols, b, y, np.ones((2, 2), dtype=bool), .2, np.array([[0, 1]]), np.array([[0, 0]]), np.ones(1))
    assert not t[:, 0].any()
    assert t[0, 1, 3] > 0. and t[0, 1, 8] == 1.
    assert m.score(t)[0, 1] < m.score(t, risk=False)[0, 1]


def test_unused_count_is_excluded_from_supervision():
    net = m.Value(5)
    legal = torch.tensor([[True, True, False], [True, True, False]])
    p = net(torch.zeros((2, 5)))
    a = torch.zeros((2, 3, 10))
    b = a.clone()
    b[:, 2] = 8.
    torch.testing.assert_close(m.objective(p, a, legal), m.objective(p, b, legal))
