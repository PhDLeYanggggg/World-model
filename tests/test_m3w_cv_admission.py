import numpy as np
import torch

from src.world_model import m3w_cv_admission as m


def test_factorized_easy_mass_and_probability():
    g, a, e, p = m.decode(torch.randn(23, 8))
    assert g.shape == a.shape == e.shape == p.shape == (23, 2)
    assert (e <= a).all() and (e >= 0).all() and (p.sum(-1) <= 1).all()
    s, cost, easy = m.predicted(g.numpy(), a.numpy(), e.numpy(), p.numpy(), np.ones(23))
    assert s.shape == easy.shape == (23,) and cost.shape == (23, 2)


def test_gate_cannot_admit_unsupported_or_nonpositive_choice():
    ok = m.admissible(np.array([1., 2., 3., -1.]), np.array([[-1., -1.], [-1., 1.], [-1., -1.], [-1., -1.]]),
                      np.array([.7, .9, .01, .9]), np.ones(4, bool), 0., .1)
    np.testing.assert_array_equal(ok, [True, False, False, False])
    assert m.select(np.arange(4), np.zeros(4, bool), 8).size == 0
    assert m.choose_setting([dict(edits=0, passes=2, gain_pp=[1., 1.])], 2) is None
    assert m.choose_setting([dict(edits=3, passes=1, gain_pp=[1., 1.])], 2) is None
    assert m.choose_setting([dict(edits=3, passes=2, gain_pp=[1., -1.])], 2) is None
    assert m.choose_setting([dict(edits=3, passes=2, gain_pp=[1., 1.], tolerance=0., min_easy=.1)], 2) == (0., .1)


def test_joint_gate_retains_count_and_legality():
    s = np.array([3., 2., 1., 8.])
    e = np.array([True, True, True, False])
    edges = np.array([[0, 0], [1, 2]])
    p = np.array([[0., 0., -10.], [0., 0., 10.]])
    np.testing.assert_array_equal(m.select(s, e, 2), [0, 1])
    np.testing.assert_array_equal(m.select(s, e, 2, edges, p, 1.), [0, 2])
