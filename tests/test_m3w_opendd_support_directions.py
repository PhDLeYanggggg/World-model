import numpy as np
import torch

from src.world_model import m3w_opendd_support_directions as m


def test_no_amplification_or_duplicate_directions():
    cv = np.zeros((2, 12, 2))
    native = cv+np.array([1., 0.])
    alt = cv+np.array([[[0., 2.]], [[0., .5]]])
    b, legal, radius = m.match(cv, native, (alt, alt.copy()))
    np.testing.assert_array_equal(legal, [[True, True], [True, False], [False, False]])
    np.testing.assert_allclose(np.linalg.norm(b[1, 0], axis=-1), radius[0])


def test_earliest_query_precedes_later_higher_score():
    a = m.first_release(np.array([2, 2, 3]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                        np.array([.2, .3, .99]), np.ones(3, dtype=bool), .1)
    assert a == (2, 1)


def test_finite_multitask_loss():
    net = m.network(3)
    loss = m.loss(net, torch.zeros((2, 3)), torch.zeros((2, 2)))
    assert torch.isfinite(loss)
    loss.backward()
