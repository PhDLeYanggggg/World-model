import numpy as np
import torch

from src.evaluation.m3w_opendd_location import paired
from src.world_model import m3w_tumdot_incumbent_selector as m


def test_incumbent_pair_target_counts_only_actual_disagreement_on_missing_steps():
    cv = np.zeros((2, 4, 2))
    b = np.stack([cv+[1., 0.], cv+[-1., 0.]], axis=1)
    valid = np.array([[True, True, False, False], [False]*4])
    y = np.where(valid[..., None], np.broadcast_to([2., 0.], cv.shape), np.nan)
    z = m.labels(cv, b, y, valid, np.ones(2), 2., 0, paired)
    np.testing.assert_array_equal(z["pair_target"][:, 0], 0.)
    np.testing.assert_allclose(z["pair_target"][:, 1, 0], [-2., -2.])
    np.testing.assert_array_equal(z["pair_target"][:, 1, 2], [1., 0.])


def test_guard_falls_back_without_changing_recipient_budget():
    score = np.array([[[-.2, -.1, -.2], [.1, .1, 0.], [0., 0., 0.]],
                      [[.4, -.1, 0.], [0., 0., .1], [0., 0., 0.]]])
    np.testing.assert_array_equal(m.choices(score, 2), [1, 0])
    np.testing.assert_array_equal(m.choices(score, 2, True), [2, 0])
    a = torch.randn(2, 3, 3)
    np.testing.assert_array_equal(m.centered(torch.nn.Identity(), a, 2)[:, 2], 0.)
