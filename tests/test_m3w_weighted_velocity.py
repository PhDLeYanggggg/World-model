import torch

from scripts.audit_m3w_weighted_velocity_debt import decompose
from src.world_model.m3w_weighted_velocity import (
    WeightedVelocity, anchored_path, line_fit, objective, robust_path,
)


def test_line_fit_anchored_to_last_observation_not_free_intercept():
    h = torch.arange(8, dtype=torch.float64)[None, :, None]*torch.tensor([[[2., -1.]]])
    h += 100.
    v, a = line_fit(h, torch.arange(1, 9, dtype=h.dtype)[None])
    torch.testing.assert_close(v, torch.tensor([[2., -1.]], dtype=h.dtype))
    torch.testing.assert_close(a, h[:, -1])
    torch.testing.assert_close(anchored_path(v)[:, 0], v)


def test_fixed_robust_weights_reject_one_older_observation_outlier():
    h = torch.arange(8, dtype=torch.float64)[None, :, None]*torch.tensor([[[2., 1.]]])
    h[:, 2] += 20.
    wanted = torch.tensor([[2., 1.]], dtype=h.dtype)
    flat = line_fit(h, torch.ones(1, 8, dtype=h.dtype))[0]
    robust = robust_path(h)[:, 0]
    assert (robust-wanted).norm() < (flat-wanted).norm()


def test_weighted_forecaster_is_equivariant_in_native_coordinate_plane():
    h = torch.randn(4, 8, 2)
    q = torch.tensor([[0., -1.], [1., 0.]])
    model = WeightedVelocity()
    with torch.no_grad():
        model.score[-1].weight.fill_(.15)
    a, w = model(h)
    b, z = model(h@q+8.)
    torch.testing.assert_close(a@q, b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(w, z, rtol=1e-5, atol=1e-5)
    assert (w > 0.).all()


def test_future_mask_ignores_padding_but_unknown_debit_keeps_gradient():
    p = torch.ones(2, 12, 2, requires_grad=True)
    cv, ols = torch.zeros_like(p), torch.ones_like(p)*.25
    valid = torch.zeros(2, 12, dtype=torch.bool)
    y = torch.full_like(p, float("nan"))
    scale = torch.ones(2)
    easy = torch.zeros(2, dtype=torch.bool)
    assert objective(p, cv, ols, y, valid, scale, easy, False) == 0.
    value = objective(p, cv, ols, y, valid, scale, easy, True)
    value.backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0.


def test_dual_reference_triangle_gap_keeps_pair_contrasts_distinct():
    def pair(known, lower):
        return {"known_pp": {"point": known}, "completion_lower_pp": {"point": lower}}
    row = decompose(pair(5., 2.), pair(2., -3.), 6.)
    assert row["excess_sum_debit"] == 2.
    assert row["joint_lower_sum"] == -1.
    assert row["triangle_optimistic_joint_sum"] == 1.
