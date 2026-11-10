import torch

from src.world_model.m3w_exchange_angular_incumbent import gain, objective


def test_incumbent_tail_uses_worst_two_donor_means():
    cv = torch.zeros(3, 1, 2)
    pred = cv.clone()
    pred[:, 0, 0] = torch.arange(3)
    args = (pred, cv, cv, cv, cv, torch.ones(3, 1, dtype=torch.bool), torch.ones(3), 1)
    assert objective(*args, 'dual_reference').item() == 1.
    assert objective(*args, 'incumbent_mean').item() == 2.
    assert objective(*args, 'incumbent_tail').item() == 2.25


def test_direct_incumbent_debit_is_not_relative_to_cv():
    ref = torch.tensor([[[1., 0.], [1., 0.]]])
    pred = torch.tensor([[[0., 1.], [0., 1.]]])
    valid = torch.tensor([[True, False]])
    y = pred.clone()
    # Observed gain and unobserved disagreement have the same magnitude.
    torch.testing.assert_close(gain(pred, ref, y, valid, torch.ones(1)), torch.zeros(1))
    changed = y.clone()
    changed[:, 1] = -300.
    torch.testing.assert_close(gain(pred, ref, changed, valid, torch.ones(1)), torch.zeros(1))
