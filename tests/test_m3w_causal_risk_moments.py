import numpy as np
import torch

from scripts.report_m3w_opendd_risk_recipient_guard import guard
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model.m3w_causal_risk_moments import MomentHead, choose, excess, target


def test_moments_reconstruct_original_risk_including_missing_and_signed_easy():
    cv = np.zeros((4, 12, 2))
    y = cv.copy()
    y[:, :, 0] = np.array([1., 1., 3., 4.])[:, None]
    p = cv.copy()
    p[:, :, 0] = np.array([-.05, .10, -1., 0.])[:, None]
    mask = np.ones((4, 12), dtype=bool)
    mask[2] = False
    z = target(cv, p[:, None], y, mask, 2.)[:, 0]
    chosen = np.linalg.norm(p-cv, axis=-1).max(-1) > 0.
    a, b = excess(z[chosen].sum(0), z[:, 4].sum())
    r = policy(p, cv, cv, y, mask, ["r"]*len(y), easy_cut=2., hard_cut=3.)["recordings"]["r"]
    np.testing.assert_allclose(a/b+.02, [r[k] for k in ("completion_all_risk_upper",
                               "completion_easy_risk_upper", "known_easy_net_degradation")])
    np.testing.assert_array_equal(z[2], [0., 0., 0., 0., 0., 1.])


def test_head_is_bounded_and_reference_does_not_depend_on_replacement():
    x = torch.randn(5, 45)
    y = x.clone()
    y[:, 20:] = torch.randn_like(y[:, 20:])
    r = torch.rand(5)
    net = MomentHead()
    z = net(x, r)
    torch.testing.assert_close(net(y, r*2)[:, 3:5], z[:, 3:5])
    assert torch.all(z[:, 0] + z[:, 5] <= r)
    assert torch.all(z[:, 1] <= z[:, 0])
    assert torch.all(z[:, 2] <= z[:, 1])
    assert torch.all(z[:, 4] <= z[:, 3])
    assert torch.all(net(y, r*0)[:, [0, 1, 2, 5]] == 0.)


def test_only_prefix_can_spend_earlier_predicted_reference_credit():
    z = np.array([[.03, .03, .03, 1., 1., 0.], [0., 0., 0., 1., 1., 0.]])
    values = np.array([1., 0.])
    earlier = np.array([0., 0., 0., 3., 3., 0.])
    assert choose(values, z, 1., earlier, 3., "local") == 1
    assert choose(values, z, 1., earlier, 3., "prefix") == 0
    assert choose(values, z, 1., np.zeros(6), 0., "prefix") == 1


def test_guard_keeps_same_recipient_shape_but_rejects_recipient_change():
    cv = np.zeros((5, 12, 2))
    q = np.array([0, 0, 1, 1, 1])
    old = cv.copy()
    old[[0, 2, 3], :, 0] = np.array([1., 2., 3.])[:, None]
    alt = old.copy()
    alt[[0, 1]] = old[[1, 0]]
    alt[[2, 3]] = -old[[2, 3]]
    out, stats = guard(q, cv, old, alt, alt)
    np.testing.assert_array_equal(out["prefix_same_recipient"][:2], old[:2])
    np.testing.assert_array_equal(out["prefix_same_recipient"][2:], alt[2:])
    np.testing.assert_array_equal(out["prefix_single_only"][:2], alt[:2])
    np.testing.assert_array_equal(out["prefix_single_only"][2:], old[2:])
    assert stats["prefix_same_recipient_recipient_rejects"] == 1
