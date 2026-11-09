import numpy as np
import torch

from src.world_model.m3w_vehicle_pair_geometry import GeometryValue, shuffle_origins, spacing


def fixture():
    t = np.arange(-7, 1)
    v = np.array([[1., 0.], [-1., 0.]])
    return dict(speed=v, units=np.ones(2)*12, history=v[:, None]*t[None, :, None],
                paths=v[:, None, None]*np.arange(1, 13)[None, None, :, None])


def test_spacing_is_symmetric_and_translation_rotation_invariant():
    m = fixture()
    o = np.array([[0., 0.], [10., 0.]])
    a, b, k = [0], [1], [0]
    x = spacing(m, o, a, b, k, k)
    np.testing.assert_array_equal(x, spacing(m, o+100., b, a, k, k))
    rot = np.array([[0., -1.], [1., 0.]])
    changed = {k: v @ rot if k != 'units' else v for k, v in m.items()}
    np.testing.assert_array_equal(x, spacing(changed, o @ rot, a, b, k, k))
    assert x[0, 2] == 1. and x[0, 5] == 0.
    np.testing.assert_allclose(x[0, 3], 5/12)


def test_permutation_stays_inside_query_and_replays():
    x = np.arange(16).reshape(8, 2)
    out = shuffle_origins(x, np.array([3, 5]), 112)
    np.testing.assert_array_equal(out, shuffle_origins(x, np.array([3, 5]), 112))
    np.testing.assert_array_equal(np.sort(out[:3], axis=0), x[:3])
    np.testing.assert_array_equal(np.sort(out[3:], axis=0), x[3:])


def test_shared_model_supports_both_geometry_and_zero_channels():
    m = GeometryValue()
    a, b = torch.randn(9, 45), torch.randn(9, 45)
    r = torch.randn(9, 12)
    torch.testing.assert_close(m(a, b, r), m(b, a, r))
    assert m(a, b, torch.zeros_like(r)).shape == (9, 4)
