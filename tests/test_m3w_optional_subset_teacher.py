import numpy as np
import torch

from scripts.run_m3w_opendd_subset_student import Student
from src.evaluation.m3w_optional_subset_teacher import build, labels


def test_teacher_distinguishes_optimum_and_forced_feasibility():
    cv = np.zeros((3, 12, 2))
    y = cv.copy()
    y[..., 0] = 1.
    p = cv.copy()
    p[:2, :, 0] = np.array([1., -.025])[:, None]
    d = build(cv, cv, p, y, np.ones((3, 12), bool), np.zeros(3, int), [1.5, 2.])
    z, info = labels(d)
    np.testing.assert_array_equal(z, [[1, 1, 1], [0, 0, 1]])
    assert info["forced_optimal"] == 1
    p[1, :, 0] = -.5
    d = build(cv, cv, p, y, np.ones((3, 12), bool), np.zeros(3, int), [1.5, 2.])
    z, info = labels(d)
    np.testing.assert_array_equal(z, [[1, 1, 1], [0, 0, 0]])
    assert info["forced_infeasible"] == 1


def test_teacher_keeps_unidentified_local_gain_missing():
    z, info = labels(dict(chosen=np.array([2]), no_reference=True))
    assert np.isnan(z[0, 0])
    np.testing.assert_array_equal(z[:, 1:], [[0., 0.]])
    assert info == {"no_reference": 1}


def test_students_have_identical_initialization_but_separate_weights():
    net = Student()
    x = torch.ones(4, 46)
    pred = net(x)
    torch.testing.assert_close(pred, pred[:, :1].expand_as(pred))
    assert next(net.heads[0].parameters()).data_ptr() != next(net.heads[1].parameters()).data_ptr()
