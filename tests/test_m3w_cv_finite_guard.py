import numpy as np
import torch

from scripts.audit_m3w_opendd_finite_guard_capacity import best_risk
from src.world_model.m3w_cv_finite_guard import objective, terms


def test_favorable_tangent_can_still_increase_finite_easy_error():
    cv = torch.tensor([[[1., 0.], [1., 0.]]])
    target = torch.zeros_like(cv)
    offset = torch.tensor([[[-3., 0.], [-.5, 0.]]])
    valid = torch.tensor([[True, False]])
    e, c = terms(offset, torch.zeros_like(offset), cv, target, valid)
    torch.testing.assert_close(e, torch.tensor([1.]))
    torch.testing.assert_close(c, torch.tensor([4.]))
    e, c = terms(offset, offset, cv, target, valid)
    torch.testing.assert_close(e, torch.tensor([0.]))
    torch.testing.assert_close(c, torch.tensor([0.]))


def test_hidden_positions_do_not_affect_finite_or_curvature_terms():
    cv = torch.zeros(1, 12, 2)
    valid = torch.zeros(1, 12, dtype=torch.bool)
    offset = torch.ones_like(cv, requires_grad=True)
    unknown = torch.full_like(cv, float("nan"))
    e, c = terms(offset, cv, cv, unknown, valid)
    assert (e == 0.).all() and (c == 0.).all()
    v = objective(offset, torch.zeros(1), cv, cv, unknown, valid,
                  torch.tensor([False]), torch.ones(1), "curvature_guard")
    v.backward()
    assert torch.isfinite(offset.grad).all()


def test_risk_menu_selects_lower_observed_error_without_filling_gaps():
    cv = np.zeros((2, 2, 2))
    a = np.array([[[1., 0.], [0., 0.]]]*2)
    b = np.array([[[-1., 0.], [1., 0.]]]*2)
    y = np.array([[[-2., 0.], [np.nan, np.nan]], [[np.nan, np.nan]]*2])
    valid = np.isfinite(y).all(2)
    picked = best_risk(cv, [a, b], y, valid)
    np.testing.assert_array_equal(picked, [b[0], a[1]])
