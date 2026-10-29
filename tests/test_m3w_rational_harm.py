import numpy as np
import pytest

from src.world_model import m3w_rational_harm as api
from src.world_model import m3w_full_range_harm as clipped
from tests.test_m3w_full_range_harm import TinyBase, fit
from tests.test_m3w_isolated_harm_fp64 import anchor
from tests.test_m3w_shared_train_risk import packet


def test_matched_initialization_resume_and_protection(tmp_path):
    a = anchor(tmp_path)
    direct = fit(tmp_path/'direct.pt.gz', a, implementation=api)
    fit(tmp_path/'resume.pt.gz', a, implementation=api, stop_at=3)
    resumed = fit(tmp_path/'resume.pt.gz', a, implementation=api, resume=True)
    for k in direct:
        if k != 'seconds':
            api.core.exact(direct[k], resumed[k])
    control = fit(tmp_path/'control.pt.gz', a, implementation=clipped)
    api.assert_control_match(direct, control)
    for k in ('loss', 'primary_loss', 'regret', 'mean_soft_action'):
        api.core.exact(direct['trace'][0][k], control['trace'][0][k])
    x, env = packet()[:2]
    p = api.predict(direct, 'a', x, env)
    teacher = api.previous.predict(a, 'a', x, env)
    np.testing.assert_array_equal(p[:, [0, 2, 3]], teacher[:, [0, 2, 3]])
    assert (p >= 0).all() and (p[:, :2].sum(1) <= env+1e-6).all()
    assert (p[:, 4] <= p[:, 1]).all()


@pytest.mark.parametrize('raw', [-1e6, -12., -.5, -1e-9, 0., 1e-9, .5, 12., 1e6])
def test_bounded_finite_map_including_extreme_anchors(raw):
    q = api.torch.tensor([0., 1e-83, 1e-30, .01, .5, .99, 1-1e-15, 1.], dtype=api.torch.float64)
    z = api.torch.full_like(q, raw, requires_grad=True)
    out = api.bounded_fraction(q, z)
    out.sum().backward()
    assert api.torch.isfinite(out).all() and api.torch.isfinite(z.grad).all()
    assert (out >= 0).all() and (out <= 1).all()
    if raw == 0:
        api.torch.testing.assert_close(out, q, atol=0, rtol=0)
        api.torch.testing.assert_close(z.grad[1:-1], api.torch.ones_like(q[1:-1]), atol=1e-14, rtol=1e-14)
    else:
        assert (z.grad[1:-1] > 0).all()


def test_clipped_zero_gradient_repaired_for_interior_anchor():
    for module, expected_nonzero in ((clipped, False), (api, True)):
        model = module.HarmBranch(TinyBase(), 2, True)
        with api.torch.no_grad():
            model.risk_output.bias.fill_(-.5)
        out = model(api.torch.zeros(1, 2), api.torch.ones(1))
        out[:, 4].sum().backward()
        assert bool(model.risk_output.bias.grad.abs().sum() > 0) == expected_nonzero


def test_zero_envelope_and_full_reach():
    model = api.HarmBranch(TinyBase(), 2, True)
    model(api.torch.ones(2, 2), api.torch.zeros(2)).sum().backward()
    assert all(p.grad is None or api.torch.isfinite(p.grad).all() for p in model.parameters())
    with api.torch.no_grad():
        model.risk_output.bias.fill_(1e6)
    p = model(api.torch.zeros(1, 2), api.torch.ones(1))
    assert p[0, 1] > .899 and p[0, 4] > .899
