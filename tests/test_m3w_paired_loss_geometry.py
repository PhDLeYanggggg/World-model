import pytest
import torch

from src.world_model.m3w_paired_decision_risk import contrast


@pytest.mark.parametrize('weight', [.1, 1., 2.])
def test_null_curvature_matches_difference_not_common_mode(weight):
    scores = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    w = torch.tensor(weight, dtype=torch.float64)
    hessians = {}
    for arm in ('paired_point', 'paired_rank'):
        fn = lambda s: contrast(s[0], s[1], w, arm)
        hessians[arm] = torch.autograd.functional.hessian(fn, scores)
    point, rank = hessians['paired_point'], hessians['paired_rank']
    torch.testing.assert_close(point, w/2*torch.eye(2, dtype=scores.dtype))
    torch.testing.assert_close(rank, w/4*torch.tensor([[1., -1.], [-1., 1.]], dtype=scores.dtype))
    difference = torch.tensor([1., -1.], dtype=scores.dtype)
    common = torch.ones(2, dtype=scores.dtype)
    torch.testing.assert_close(difference @ point @ difference, difference @ rank @ difference)
    torch.testing.assert_close(common @ rank @ common, torch.zeros((), dtype=scores.dtype))
    assert common @ point @ common > 0


def test_rank_order_and_loss_do_not_identify_acceptance_boundary():
    w = torch.tensor(1., dtype=torch.float64)
    harmful = torch.tensor(2., dtype=torch.float64)
    beneficial = torch.tensor(-2., dtype=torch.float64)
    reference = contrast(harmful, beneficial, w, 'paired_rank')
    for shift in (-10., 0., 10.):
        torch.testing.assert_close(contrast(harmful+shift, beneficial+shift, w, 'paired_rank'), reference)
        assert harmful+shift > beneficial+shift
    assert harmful-10 <= 0 and beneficial-10 <= 0
    assert harmful+10 > 0 and beneficial+10 > 0
