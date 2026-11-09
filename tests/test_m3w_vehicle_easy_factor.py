import numpy as np
import torch

from scripts import run_m3w_vehicle_easy_factor as run


def test_factor_predicts_conditional_support_and_separate_unknown_debit():
    model = run.model.FactorValue()
    x = torch.zeros((2, 45))
    g, b, logit, cond = model.parts(x)
    out = model(x)
    torch.testing.assert_close(out[:, 3], logit.sigmoid()*cond[:, 0]+cond[:, 2])
    torch.testing.assert_close(out[:, 5], logit.sigmoid()*cond[:, 1])
    y = torch.zeros_like(out)
    u = torch.zeros(2)
    loss = run.model.factor_loss(model, x, y, u)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_calibration_rejects_entire_failed_score_tie_and_abstains_when_top_fails():
    threshold, n = run.calibration(np.array([3., 4., 3., 1.]), np.array([True, True, False, True]))
    assert (threshold, n) == (4., 1)
    assert run.calibration(np.array([2., 1.]), np.array([False, True])) == (None, 0)


def test_fixed_slot_uses_largest_supported_query_then_first_tie():
    off = np.array([0, 2, 5, 8, 10])
    rec = np.array(['A']*8 + ['B']*2)
    legal = np.ones((10, 3), dtype=bool)
    assert run.fixed_slots(off, rec, legal) == {'A': (2, 5), 'B': (8, 10)}
