import numpy as np
import torch

from src.world_model import m3w_selected_positive_harm as m


def test_event_and_severity_loss_have_finite_gradients_for_empty_positive_batch():
    for arm in (True, False):
        logits = torch.zeros((4, 2), requires_grad=True)
        m.loss(logits, torch.zeros(4), arm).backward()
        assert logits.grad.isfinite().all()


def test_positive_severity_and_event_gradients():
    logits = torch.zeros((3, 2), requires_grad=True)
    m.loss(logits, torch.tensor([0., .25, 1.]), True).backward()
    assert logits.grad[0, 0] > 0
    assert logits.grad[1, 1] > 0
    assert logits.grad[2, 1] < 0


def test_fixed_counts_within_unchanged_candidate_pool():
    score = np.array([[.8, .7], [.6, .9], [.1, .9]])
    eligible = np.array([[False, True], [True, True], [True, True]])
    choices = m.matched([np.arange(3)], np.array([0, -1, -1]), score, eligible)
    np.testing.assert_array_equal(choices, [-1, 1, -1])


def test_same_sampling_stream_and_only_fitting_query_members():
    groups = [[np.array([10, 11]), np.array([20])], [np.array([30, 31])]]
    a = m.sampled_batches(groups, 4, 64, 17)
    np.testing.assert_array_equal(a, m.sampled_batches(groups, 4, 64, 17))
    assert set(a.ravel()) == {10, 11, 20, 30, 31}
