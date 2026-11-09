from math import log

import pytest

from scripts import audit_m3w_av2_paired_release_capacity as audit
from scripts import run_m3w_av2_supported_gain as run

torch = run.torch


def test_conditional_magnitude_ignores_nonqualifying_examples():
    z = torch.zeros(3, requires_grad=True)
    b = torch.ones(3)
    g = torch.tensor([.25, 2., -.9])
    q = torch.tensor([1., 0., 0.])
    loss = run.gain_loss(z, b, g, q, "conditional")
    assert loss.item() == pytest.approx(.0625)
    loss.backward()
    torch.testing.assert_close(z.grad, torch.tensor([.125, 0., 0.]))
    assert run.gain_loss(z, b, g, torch.zeros(3), "conditional").item() == 0.


def test_censored_value_targets_zero_on_rejected_slate():
    z = torch.zeros(2, requires_grad=True)
    b, g, q = torch.ones(2), torch.tensor([.5, .9]), torch.tensor([1., 0.])
    loss = run.gain_loss(z, b, g, q, "censored")
    assert loss.item() == pytest.approx(.125)
    loss.backward()
    torch.testing.assert_close(z.grad, torch.tensor([0., .125]))


def test_score_factorizes_only_the_conditional_head():
    z = torch.tensor([[0., log(5.), 0.], [0., log(5.), 50.], [0., 0., 0.]])
    b = torch.tensor([2., 0., 1.])
    torch.testing.assert_close(run.scores(z, b, "conditional"), torch.tensor([.5, 0., 1./12.]))
    torch.testing.assert_close(run.scores(z, b, "censored"), torch.tensor([1., 0., .5]))


def test_calibration_cannot_accept_unsupported_additive_or_empty_policy():
    table = run.calibration([dict(score=.4, strict=True, add_ok=False)])
    assert table["strict_with_add"] == 0
    assert run.winner(dict.fromkeys(run.METHODS, table)) is None
    good = run.calibration([dict(score=.3, strict=True, add_ok=True)])
    assert good["threshold"] == .3
    assert run.winner(dict.fromkeys(run.METHODS, good)) == "conditional"


def test_relaxed_reference_requires_two_distinct_eligible_forecasts():
    np = run.np
    y = np.zeros((3, 12, 2))
    cv = y.copy()
    cv[:, :, 0] = np.array([.1, 2., 1.])[:, None]
    v = np.ones((3, 12), dtype=bool)
    d = dict(cv=cv, ols4=cv+.1)
    a, j = cv.copy(), cv.copy()
    a[:2] = y[:2]
    a[1, :, 0] = -.2
    j[:2] = y[:2]
    assert audit.alternative_capacity(d, np.stack((a, j)), y, v, (.2, 1.)) == (True, True)
    assert audit.alternative_capacity(d, np.stack((a, a)), y, v, (.2, 1.)) == (True, False)
    v[0] = False
    assert audit.alternative_capacity(d, np.stack((a, j)), y, v, (.2, 1.)) == (False, False)
