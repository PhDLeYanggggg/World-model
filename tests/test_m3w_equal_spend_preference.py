import pytest
import torch

from src.world_model.m3w_equal_spend_preference import objective, plan_values


def test_plan_values_sum_only_selected_increments_by_recording():
    scores = torch.tensor([[.1, 2.1, 9.], [3., 4., 5.], [8., 9., 11.]])
    count = torch.tensor([[1, 0], [0, 1], [2, 2]])
    group = torch.tensor([0, 0, 1])
    scale = torch.tensor([1., 2.])
    torch.testing.assert_close(plan_values(scores, count, group, scale), torch.tensor([[2., 1.], [1.5, 1.5]]))
    torch.testing.assert_close(plan_values(scores + torch.tensor([[10.], [-1.], [8.]]), count, group, scale),
                               plan_values(scores, count, group, scale))


def test_pairwise_loss_is_invariant_to_per_edit_bias_at_equal_spend():
    s = torch.tensor([[0., 1., 5.], [0., 2., 3.]], requires_grad=True)
    choices = torch.tensor([[0, 1], [2, 1]])
    g = torch.tensor([0, 0])
    scale = torch.tensor([2. ** .5])
    a = objective(s, choices, g, scale, pairwise=True)
    b = objective(s + torch.tensor([0., 4., 8.]), choices, g, scale, pairwise=True)
    torch.testing.assert_close(a, b)
    assert a.item() == pytest.approx(.69314718)
    a.backward()
    assert s.grad[1, 2] < 0. < s.grad[1, 1]
