import numpy as np
import torch

from src.world_model.m3w_recipient_feasibility import RecipientHead, attainable, loss, nominate
from scripts.audit_m3w_recipient_relaxed_search import relaxed
from scripts.run_m3w_recipient_query_fallback import local_choices


def test_attainable_risk_excludes_identity_and_selects_unknown_debit():
    gain = np.array([[0., -.2, -.3], [0., 0., 0.], [0., 0., 0.]])
    radius = np.array([[0., .5, .4], [0., .5, .2], [0., 0., 0.]])
    a, u, ok = attainable(gain, np.array([True, False, False]), radius)
    np.testing.assert_allclose(a, [-.2, 0., 0.])
    np.testing.assert_allclose(u, [0., .2, 0.])
    np.testing.assert_array_equal(ok, [True, True, False])


def test_recipient_head_uses_coherent_bounded_moments():
    net = RecipientHead(4)
    b = torch.tensor([.2, 1.])
    ref, cost, logits = net(torch.ones((2, 4)), b)
    assert torch.all(ref[:, 1] <= ref[:, 0])
    assert torch.all(cost[:, 1] <= b)
    assert torch.all(cost[:, 2] <= cost[:, 1])
    observed = torch.tensor([True, False])
    obj = loss((ref, cost, logits), b*observed, torch.tensor([0.1, 0.]), torch.tensor([0., .4]),
               observed, observed, True)
    obj.backward()
    assert torch.isfinite(obj)
    assert all(p.grad is not None and p.grad.isfinite().all() for p in net.parameters())


def test_common_nominator_does_not_prefer_illegal_identity():
    cost = np.array([[[1., 0., 0.], [.1, 0., 0.], [1., .2, .2]]])
    np.testing.assert_array_equal(nominate(cost, np.array([[False, True, True]])), [1])


def test_relaxation_distinguishes_mass_certificate_from_witness():
    ok = np.ones(4, bool)
    gain = np.array([4., 3., 2., 1.])
    a = np.array([[-1., .8], [-1., .8], [.2, -.1], [.2, -.1]])
    assert relaxed(gain, a, ok, 3) == ("easy_mass_lower_bound", None)
    a[1, 1] = -.5
    status, on = relaxed(gain, a, ok, 2)
    assert status == "integer_witness"
    assert len(on) == 2 and (a[on].sum(0) <= 0.).all()


def test_local_fallback_ignores_other_queries_and_matches_counts():
    inv = np.array([0, 0, 1, 1])
    cost = np.array([[3., 0., 0.], [2., 0., 0.], [5., .5, 0.], [4., 0., .5]])
    args = (inv, np.ones(2, int), np.ones(4, bool), np.ones((4, 2)), cost, np.arange(4))
    a, b = local_choices(*args)
    np.testing.assert_array_equal(a, [False, True, False, False])
    np.testing.assert_array_equal(b, [True, False, False, False])
    cost[2:] = [500., 0., 0.]
    c, d = local_choices(*args)
    np.testing.assert_array_equal(c[:2], a[:2])
    np.testing.assert_array_equal(d[:2], b[:2])
