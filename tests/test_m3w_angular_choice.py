import torch

from src.world_model.m3w_angular_choice import DEGREES, choose, targets


def test_keep_has_zero_utility_and_wins_ties():
    s = torch.zeros(2, 2, len(DEGREES))
    assert choose(s, False).tolist() == [0, 0]
    s[0, 1, 3] = .1
    assert choose(s, False).tolist() == [4, 0]
    assert choose(s, True).tolist() == [0, 0]
    assert choose(s, True, force=True).tolist() == [1, 1]


def test_missing_debit_counteracts_apparent_observed_gain():
    cv = torch.zeros(1, 2, 2)
    p = cv.clone()
    p[0, :, 0] = 1.
    y = cv.clone()
    y[0, :, 1] = 1.
    valid = torch.tensor([[True, False]])
    observed, debit = targets(cv, p, y, valid, torch.ones(1))
    i = DEGREES.index(90)
    assert observed[0, i] > 0.
    torch.testing.assert_close(observed[0, i], debit[0, i])
    y[0, 1] = -100.
    a, b = targets(cv, p, y, valid, torch.ones(1))
    torch.testing.assert_close(a, observed)
    torch.testing.assert_close(b, debit)
